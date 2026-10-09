"""Bounded request scheduling for four-stage, forward-only decision inference."""
import asyncio
import atexit
from collections import deque
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass
import json
import queue
import threading
import time

from fastapi import HTTPException

from .cuda_graphs import bucket
from .model import admit, SERVE_MAX_STATE_8K, SERVE_MAX_BRANCH_8K, ROW_PASS_TOKENS
from .pp4_engine import Engine, plan_layout
from .serve import DecisionResponses, prepare
from .api import to_record, SystemOneRequest


@dataclass(frozen=True)
class PipelineOptions:
    max_batch: int = 8
    inflight: int = 8
    batch_wait_ms: float = 2
    capacity: int = 256
    request_timeout: float = 120
    state_budget: int = ROW_PASS_TOKENS
    row_budget: int = ROW_PASS_TOKENS
    max_state: int = SERVE_MAX_STATE_8K
    max_row: int = SERVE_MAX_BRANCH_8K
    graphs: bool = True

    def __post_init__(self):
        if not (1 <= self.max_batch <= 8 and 1 <= self.inflight <= 8):
            raise ValueError('PP4 batch size and inflight must be in 1..8')
        if self.capacity < 1 or self.request_timeout <= 0 or self.batch_wait_ms < 0:
            raise ValueError('invalid PP4 queue or timeout settings')
        if self.state_budget != ROW_PASS_TOKENS or self.row_budget != ROW_PASS_TOKENS:
            raise ValueError('PP4 buffers use the canonical row-pass token budget')


@dataclass
class Pending:
    enc: dict
    done: Future
    queued: float


class PipelineServer(DecisionResponses):
    """Shares the standard TypeSafe response code; owns a bounded pipeline scheduler."""
    def __init__(self, run, options=PipelineOptions(), *, tokenizer_path=None, engine=None):
        self.options = options
        self.engine = engine or Engine({**vars(options), 'model': run, 'tokenizer_path': tokenizer_path})
        self.checkpoint, self.tok = self.engine.checkpoint, self.engine.tok
        self.release_date = self.checkpoint.release_date()
        self.truncate_states = False
        self.ready = False
        self.batches = self.batched_requests = 0
        self.queue = queue.Queue()
        self.condition = threading.Condition()
        self.accepted = 0
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._dispatch, name='kev-pp4-batcher', daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def _encode(self, rec):
        return admit(self.engine, self.tok, rec, max_state=self.options.max_state, max_branch=self.options.max_row)

    def _fits(self, encs):
        try:
            plan_layout(encs, vars(self.options))
            return True
        except ValueError:
            return False

    def submit(self, rec):
        with self.condition:
            if self.stopping.is_set() or self.engine.failed:
                raise HTTPException(503, 'pipeline is unavailable')
            if self.accepted >= self.options.capacity:
                raise HTTPException(503, 'request capacity is full')
            self.accepted += 1
        try:
            enc = self._encode(rec)
            if not self._fits([enc]):
                raise ValueError(f'request exceeds PP4 padded state/question-output/workspace budget ({self.options.row_budget} tokens); split questions across requests')
        except Exception as exc:
            self._release()
            if isinstance(exc, ValueError): raise HTTPException(422, str(exc)) from None
            raise
        done = Future()
        # Admission and enqueue are atomic with shutdown; no request can appear
        # after the dispatcher drained its final queue.
        with self.condition:
            if self.stopping.is_set() or self.engine.failed:
                self.accepted -= 1
                self.condition.notify_all()
                raise HTTPException(503, 'pipeline is unavailable')
            self.queue.put(Pending(enc, done, time.monotonic()))
        return done

    def _release(self):
        with self.condition:
            self.accepted -= 1
            self.condition.notify_all()

    def _finish_request(self, item, result=None, error=None):
        # Cancellation and completion race on a Future. RUNNING claims it only
        # when it has not already been cancelled; active GPU work still drains.
        if item.done.set_running_or_notify_cancel():
            if error is not None: item.done.set_exception(error)
            else: item.done.set_result(result)
        self._release()

    def _dispatch(self):
        pending = deque()
        batch = []
        candidate = None
        try:
            while not self.stopping.is_set() and not self.engine.failed:
                if not pending:
                    try: pending.append(self.queue.get(timeout=.05))
                    except queue.Empty: continue
                first = pending.popleft()
                if first.done.cancelled():
                    self._release(); continue
                if time.monotonic() - first.queued >= self.options.request_timeout:
                    self._finish_request(first, error=HTTPException(504, 'request deadline exceeded')); continue
                batch = [first]
                target = bucket(first.enc['seg'].count(0))
                deadline = time.monotonic() + self.options.batch_wait_ms / 1000
                while len(batch) < self.options.max_batch:
                    candidate = None
                    for i, item in enumerate(pending):
                        if bucket(item.enc['seg'].count(0)) == target and self._fits([b.enc for b in batch] + [item.enc]):
                            candidate = item; del pending[i]; break
                    if candidate is None:
                        try: candidate = self.queue.get(timeout=max(0, deadline - time.monotonic()))
                        except queue.Empty: break
                    if candidate.done.cancelled():
                        self._release(); candidate = None; continue
                    if bucket(candidate.enc['seg'].count(0)) == target and self._fits([b.enc for b in batch] + [candidate.enc]):
                        batch.append(candidate)
                    else:
                        pending.append(candidate)
                    candidate = None
                    if time.monotonic() >= deadline: break
                try:
                    result = self.engine.submit([item.enc for item in batch])
                    result.add_done_callback(lambda f, b=batch: self._complete(f, b))
                except Exception:
                    for item in batch:
                        self._finish_request(item, error=HTTPException(503, 'pipeline is unavailable'))
                batch = []
        except Exception as exc:
            self.engine.fail(exc)
        finally:
            pending.extend(batch)
            if candidate is not None: pending.append(candidate)
            with self.condition:
                self.stopping.set()
                while True:
                    try: pending.append(self.queue.get_nowait())
                    except queue.Empty: break
            for item in pending:
                self._finish_request(item, error=HTTPException(503, 'pipeline stopped'))

    def _complete(self, future, batch):
        try:
            ps, info = future.result()
        except Exception:
            for item in batch:
                self._finish_request(item, error=HTTPException(503, 'pipeline is unavailable'))
            return
        self.batches += 1
        self.batched_requests += len(batch)
        for item, probabilities in zip(batch, ps):
            stats = {'tokens': len(item.enc['ids']), 'state_tokens': item.enc['seg'].count(0),
                     'latency_ms': (time.monotonic() - item.queued) * 1000}
            if stats['latency_ms'] > self.options.request_timeout * 1000:
                self._finish_request(item, error=HTTPException(504, 'request deadline exceeded'))
            else:
                self._finish_request(item, result=(probabilities, stats))

    def probs(self, rec):
        done = self.submit(rec)
        try: return done.result(timeout=self.options.request_timeout)
        except TimeoutError:
            done.cancel()
            raise HTTPException(504, 'request deadline exceeded') from None

    async def answer_async(self, req):
        rec, meta = to_record(prepare(req))
        done = self.submit(rec)
        try:
            result = await asyncio.wait_for(asyncio.wrap_future(done), self.options.request_timeout)
        except asyncio.TimeoutError:
            raise HTTPException(504, 'request deadline exceeded') from None
        return self._body(req, meta, *result)

    def wait_idle(self, timeout=120):
        with self.condition:
            if not self.condition.wait_for(lambda: self.accepted == 0, timeout=timeout):
                raise TimeoutError('pipeline did not drain')
        self.engine.wait_idle()

    def warmup(self):
        # A finite graph set; all other shapes run eager with bounded scratch.
        self.engine.configure(inflight=1, capture=True)
        try:
            tokens = self.tok('context information. ' * 4096, add_special_tokens=False).input_ids
            for state_tokens, words, questions in ((128, 12, 1), (128, 48, 1), (1024, 12, 1), (128, 12, 8)):
                rec, _ = to_record(SystemOneRequest(
                    state=self.tok.decode(tokens[:state_tokens - 1]),
                    questions={f'warm-{i}': {'type': 'noul', 'instructions': 'question detail. ' * words}
                               for i in range(questions)}))
                enc = self._encode(rec)
                for n in (1, 2, 4, 8):
                    encs = [enc] * n
                    if self._fits(encs): self.engine.submit(encs).result(timeout=120)
        finally:
            self.engine.configure(inflight=self.options.inflight, capture=False)
        self.ready = True

    def healthy(self):
        return not self.stopping.is_set() and not self.engine.failed and self.thread.is_alive()

    def model_card(self):
        meta = self.checkpoint.meta
        manifest = self.checkpoint.file('artifact-manifest.json')
        identity = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
        return {'description': f'Kev pointer head on {meta.base}, four-stage BF16 pipeline',
                'release_date': self.release_date, 'run': self.checkpoint.requested,
                'base': meta.base, 'base_revision': meta.base_revision, 'revision': identity.get('revision'),
                'lora': meta.lora, 'device': 'cuda:0,1,2,3', 'backend': 'torch-pp4', 'dtype': 'bfloat16',
                'temperature': meta.temperature, 'world_size': 4, 'gpus': [0, 1, 2, 3],
                'layers_per_stage': [16, 16, 16, 16], 'max_state_tokens': self.options.max_state,
                'max_row_tokens': self.options.max_row, 'truncate_states': False,
                'padded_state_budget': self.options.state_budget, 'padded_question_output_budget': self.options.row_budget,
                'request_capacity': self.options.capacity, 'request_timeout_seconds': self.options.request_timeout,
                'max_batch': self.options.max_batch, 'inflight': self.options.inflight,
                'batch_wait_ms': self.options.batch_wait_ms, 'prefix_cache': {'size': 0},
                'cuda_graphs': {'enabled': self.options.graphs, 'capture_enabled': False},
                'batches': {'count': self.batches, 'requests': self.batched_requests, 'accepted': self.accepted}}

    def close(self):
        with self.condition:
            self.stopping.set(); self.ready = False
        self.engine.close()
        self.thread.join(timeout=5)
