"""Forward-only PP4: stage-local prefix forks, graph compute, bounded host relay.

No global model synchronization occurs on the serving path. Each stage has one compute
worker; event completion owns transport slots and request futures.
"""
from collections import Counter, deque
from concurrent.futures import Future
import copy
from dataclasses import dataclass, field
import os
import queue
import threading
import time

import torch
from torch import nn
from transformers import DynamicCache
from .checkpoint import Checkpoint, LoadOptions
from .device import empty_cache, allocated_bytes
from .cuda_graphs import BufferKV, bucket, count_bucket, set_linear
from .model import encode, rows_of


@dataclass
class Layout:
    encs: list
    states: torch.Tensor
    state_positions: torch.Tensor
    state_lengths: torch.Tensor
    rows: torch.Tensor
    row_positions: torch.Tensor
    row_lengths: torch.Tensor
    owners: torch.Tensor
    picks: list
    state_bucket: int
    row_bucket: int
    prefix_bucket: int
    row_chunks: list
    real_state_tokens: int
    real_row_tokens: int
    decide_picks: torch.Tensor
    option_picks: torch.Tensor
    option_counts: torch.Tensor


def plan_layout(encs, config):
    if not 1 <= len(encs) <= 8:
        raise ValueError("microbatch must contain 1..8 requests")
    splits = [rows_of(enc) for enc in encs]
    nb, sb = count_bucket(len(encs)), bucket(max(len(s[0]) for s in splits))
    if nb * sb > config["state_budget"]:
        raise ValueError("microbatch exceeds padded state token budget")
    allrows = [(i, row) for i, (_, _, rows) in enumerate(splits) for row in rows]
    lb = bucket(max(len(row["ids"]) for _, row in allrows))
    if len(allrows) * lb > config["row_budget"]:
        raise ValueError("microbatch exceeds padded question-output token budget")
    sr = 1 << (max(16, sb) - 1).bit_length()
    if sr+lb > config["row_budget"]:
        raise ValueError("one padded question row exceeds row token budget")
    limit = min(32, max(1, config["row_budget"] // (sr + lb)))
    limit = max(n for n in range(1, limit + 1) if count_bucket(n) <= limit)
    return splits, nb, sb, allrows, lb, sr, limit


def make_layout(encs, pad_id, config):
    splits, nb, sb, allrows, lb, sr, limit = plan_layout(encs, config)
    states = torch.full((nb, sb), pad_id, dtype=torch.long)
    state_pos = torch.zeros_like(states)
    lengths = torch.zeros(nb, dtype=torch.long)
    for i, (ids, pos, _) in enumerate(splits):
        states[i, sb - len(ids):] = torch.tensor(ids)
        state_pos[i, sb - len(ids):] = torch.tensor(pos)
        lengths[i] = len(ids)
    rows = torch.full((len(allrows), lb), pad_id, dtype=torch.long)
    row_pos = torch.zeros_like(rows)
    row_len, owners, picks = [], [], []
    for i, (owner, row) in enumerate(allrows):
        n = len(row["ids"])
        rows[i, :n] = torch.tensor(row["ids"])
        row_pos[i, :n] = torch.tensor(row["pos"])
        row_len.append(n); owners.append(owner)
        picks.append((owner, row["decide"], row["opts"]))
    max_opts = max(len(p[2]) for p in picks)
    decide_picks = torch.tensor([r * lb + p[1] for r, p in enumerate(picks)])
    option_picks = torch.tensor([[r * lb + o for o in p[2]] + [r * lb] * (max_opts-len(p[2])) for r, p in enumerate(picks)])
    option_counts = torch.tensor([len(p[2]) for p in picks])
    return Layout(encs, states.pin_memory(), state_pos.pin_memory(), lengths.pin_memory(),
                  rows.pin_memory(), row_pos.pin_memory(), torch.tensor(row_len).pin_memory(),
                  torch.tensor(owners).pin_memory(), picks, sb, lb, sr,
                  [(start, min(len(allrows), start + limit)) for start in range(0, len(allrows), limit)],
                  sum(len(s[0]) for s in splits), sum(len(r["ids"]) for _, r in allrows),
                  decide_picks.pin_memory(), option_picks.pin_memory(), option_counts.pin_memory())


@dataclass
class Job:
    id: int
    layout: Layout
    future: Future
    created_ns: int
    hstate: object = None
    hrows: object = None
    ready: object = None
    relay: object = None
    metrics: list = field(default_factory=list)
    graphs: bool = True


class PrefixBuffers:
    """Only the current stage's layers have storage; global layer IDs stay intact."""
    def __init__(self, config, layer_ids, device, entries, tokens, bank=False):
        self.config, self.layer_ids, self.device = config, layer_ids, device
        self.bank = bank
        self.tensors = {}
        c = config
        for i in layer_ids:
            if c.layer_types[i] == "full_attention":
                shape = (entries, c.num_key_value_heads, tokens // entries, c.head_dim) if bank else (tokens, c.num_key_value_heads * c.head_dim)
                self.tensors[i] = tuple(torch.zeros(shape, device=device, dtype=torch.bfloat16) for _ in range(2))
            else:
                channels = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim
                self.tensors[i] = (
                    torch.zeros((entries, channels, c.linear_conv_kernel_dim), device=device, dtype=torch.bfloat16),
                    torch.zeros((entries, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim), device=device, dtype=torch.float32),
                )

    def views(self, n, length, offset=0):
        views = {}
        for i, tensors in self.tensors.items():
            if self.config.layer_types[i] == "full_attention":
                if self.bank:
                    views[i] = tuple(t[:n, :, offset:offset + length] for t in tensors)
                else:
                    shape = (n, self.config.num_key_value_heads, length, self.config.head_dim)
                    views[i] = tuple(t.flatten()[:n * length * self.config.num_key_value_heads * self.config.head_dim].view(shape) for t in tensors)
            else:
                views[i] = tuple(t[:n] for t in tensors)
        return views

    def cache(self, views, filled, previous):
        cache = DynamicCache(config=self.config)
        for i, v in views.items():
            if self.config.layer_types[i] == "full_attention":
                cache.layers[i] = BufferKV(*v, filled)
            else:
                set_linear(cache.layers[i], *v, previous)
        return cache

    def bytes(self):
        return sum(t.numel() * t.element_size() for ts in self.tensors.values() for t in ts)


class Stage(nn.Module):
    def __init__(self, lm, head, index, config, device):
        from .fused_qwen35 import fuse
        super().__init__()
        self.index, self.device, self.options = index, device, config
        self.config = lm.config
        self.ids = list(range(index * 16, (index + 1) * 16))
        self.layers = nn.ModuleList(list(lm.layers)[index * 16:(index + 1) * 16]).to(device)
        self.rotary = copy.deepcopy(lm.rotary_emb).to(device)
        self.embedding = lm.embed_tokens.to(device) if index == 0 else None
        self.norm = lm.norm.to(device) if index == 3 else None
        self.head = head.to(device) if index == 3 else None
        self.eval()
        fuse(self)
        self.compute = torch.cuda.Stream(device=device)
        self.h2d = torch.cuda.Stream(device=device)
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs, self.pending, self.failures = {}, {}, {}
        self.counters = Counter()
        self.capture_enabled = False
        self.last_issue_end_ns = None
        # One bank per worker, reused only after its current question chunks finish.
        self.bank_width = config["max_state"]
        with torch.cuda.device(device):
            self.bank = PrefixBuffers(self.config, self.ids, device, config["max_batch"], config["max_batch"] * self.bank_width, bank=True)
            self.rowbuf = PrefixBuffers(self.config, self.ids, device, 32, config["row_budget"])
            self.compute.wait_stream(torch.cuda.default_stream(device))
        self.weights_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        # Fused projection tensors are intentionally plain attributes upstream.
        self.weights_bytes += sum(t.numel() * t.element_size() for layer in self.layers for mod in layer.modules()
                                  for name, t in vars(mod).items() if isinstance(t, torch.Tensor) and name in ("in_proj", "qkv", "gate_up"))

    def forward_layers(self, hidden, ids, positions, full_mask, linear_mask, cache, question):
        if self.index == 0:
            hidden = self.embedding(ids)
        rope = self.rotary(hidden, positions[None].expand(3, -1, -1))
        for layer in self.layers:
            mask = linear_mask if layer.block_type == "linear_attention" else full_mask
            hidden = layer(hidden, position_embeddings=rope, attention_mask=mask,
                           position_ids=positions, past_key_values=cache, use_cache=True)
        return self.norm(hidden) if question and self.norm is not None else hidden

    def mask(self, allow):
        return torch.zeros(allow.shape, device=self.device, dtype=torch.bfloat16).masked_fill(~allow, torch.finfo(torch.bfloat16).min)[:, None]

    def get_state_pass(self, nb, sb, unpadded=False):
        key = ("state", nb, sb, unpadded)
        if key in self.pending:
            return key, self.pending[key]
        ids = torch.zeros((nb, sb), dtype=torch.long, device=self.device)
        pos = torch.zeros_like(ids)
        lens = torch.zeros(nb, dtype=torch.long, device=self.device)
        hidden = torch.zeros((nb, sb, self.config.hidden_size), dtype=torch.bfloat16, device=self.device) if self.index else None
        output = torch.empty((nb, sb, self.config.hidden_size), dtype=torch.bfloat16, device=self.device)
        views = self.bank.views(nb, sb, self.bank_width - sb)
        def body():
            # Fresh Python metadata for every eager/warm/capture invocation;
            # state passes overwrite rather than continue a previous prefix.
            cache = self.bank.cache(views, 0, False)
            full_mask = linear_mask = None
            if not unpadded:
                pad = sb - lens[:, None]
                t = torch.arange(sb, device=self.device)
                real = t[None] >= pad
                q, k = t[None, :, None], t[None, None, :]
                allow = ((k <= q) & (k >= pad[:, :, None])) | (k == q)
                full_mask, linear_mask = self.mask(allow), real.long()
            h = self.forward_layers(hidden, ids, pos, full_mask, linear_mask, cache, False)
            output.copy_(h)
        data = {"ids": ids, "pos": pos, "lengths": lens, "hidden": hidden, "output": output, "body": body}
        # Captured graphs own persistent inputs. Unseen eager shapes keep only
        # one scratch entry per pass type; arbitrary clients cannot grow a cache.
        for old in list(self.pending):
            if old[0] == key[0] and old not in self.graphs:
                del self.pending[old]
        self.pending[key] = data
        return key, data

    def get_row_pass(self, nb, lb, sr):
        key = ("row", nb, lb, sr)
        if key in self.pending:
            return key, self.pending[key]
        ids = torch.zeros((nb, lb), dtype=torch.long, device=self.device)
        pos = torch.zeros_like(ids)
        lens = torch.zeros(nb, dtype=torch.long, device=self.device)
        plen = torch.zeros_like(lens)
        owners = torch.zeros_like(lens)
        hidden = torch.zeros((nb, lb, self.config.hidden_size), dtype=torch.bfloat16, device=self.device) if self.index else None
        output = torch.empty((nb, lb, self.config.hidden_size), dtype=torch.bfloat16, device=self.device)
        views = self.rowbuf.views(nb, sr + lb)
        cache = self.rowbuf.cache(views, sr, True)

        def body():
            for i, dst in views.items():
                src = self.bank.tensors[i]
                for d, s in zip(dst, src):
                    if self.config.layer_types[i] == "full_attention":
                        d[:, :, :sr].copy_(s[:, :, -sr:].index_select(0, owners))
                    else:
                        d.copy_(s.index_select(0, owners))
            q = torch.arange(lb, device=self.device)[None, :, None]
            k = torch.arange(sr + lb, device=self.device)[None, None, :]
            allow = ((k < sr) & (k >= sr - plen[:, None, None])) | ((k >= sr) & (k - sr <= q) & (k - sr < lens[:, None, None])) | (k - sr == q)
            linear = (torch.arange(lb, device=self.device)[None] < lens[:, None]).long()
            output.copy_(self.forward_layers(hidden, ids, pos, self.mask(allow), linear, cache, True))
        data = {"ids": ids, "pos": pos, "lengths": lens, "prefix_lengths": plen, "owners": owners,
                "hidden": hidden, "output": output, "body": body}
        # Captured graphs own persistent inputs. Unseen eager shapes keep only
        # one scratch entry per pass type; arbitrary clients cannot grow a cache.
        for old in list(self.pending):
            if old[0] == key[0] and old not in self.graphs:
                del self.pending[old]
        self.pending[key] = data
        return key, data

    def replay(self, key, data, enabled):
        if enabled and key in self.graphs:
            self.graphs[key].replay(); self.counters["graph_replays"] += 1
        elif enabled and self.capture_enabled and (key[0] == "row" or key[0] == "state" and key[2] <= 1024):
            # Warm/capture only during an explicit idle, serial preparation phase.
            for _ in range(2): data["body"]()
            self.compute.synchronize()
            empty_cache("cuda")
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, pool=self.pool, stream=self.compute, capture_error_mode="thread_local"):
                    data["body"]()
                self.graphs[key] = graph
                graph.replay()
                self.counters["captures"] += 1
            except Exception as exc:
                self.failures[str(key)] = repr(exc)
                raise
        else:
            data["body"](); self.counters["eager_passes"] += 1
            if key[0] == "state" and key[2] > 1024:
                self.counters["expected_long_state_eager"] += 1

    def run_job(self, job):
        l = job.layout
        nb = l.states.shape[0]
        unpadded = l.state_bucket > 1024 and bool(torch.all(l.state_lengths == l.state_bucket))
        state_key, state = self.get_state_pass(nb, l.state_bucket, unpadded)
        state["ids"].copy_(l.states, non_blocking=True)
        state["pos"].copy_(l.state_positions, non_blocking=True)
        state["lengths"].copy_(l.state_lengths, non_blocking=True)
        if self.index:
            state["hidden"].copy_(job.hstate)
        start = torch.cuda.Event(enable_timing=True)
        state_end = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(self.compute)
        self.replay(state_key, state, job.graphs)
        # Snapshot before another graph may reuse the shared scratch pool.
        hstate = state["output"].clone() if self.index < 3 else None
        state_end.record(self.compute)
        hrows = torch.empty((l.rows.shape[0], l.row_bucket, self.config.hidden_size), device=self.device, dtype=torch.bfloat16)
        for a, b in l.row_chunks:
            nr = count_bucket(b - a)
            key, row = self.get_row_pass(nr, l.row_bucket, l.prefix_bucket)
            row["ids"].zero_(); row["pos"].zero_(); row["lengths"].zero_(); row["owners"].zero_(); row["prefix_lengths"].zero_()
            row["ids"][:b - a].copy_(l.rows[a:b], non_blocking=True)
            row["pos"][:b - a].copy_(l.row_positions[a:b], non_blocking=True)
            row["lengths"][:b - a].copy_(l.row_lengths[a:b], non_blocking=True)
            row["owners"][:b - a].copy_(l.owners[a:b], non_blocking=True)
            row["prefix_lengths"][:b - a].copy_(l.state_lengths[l.owners[a:b]], non_blocking=True)
            if self.index:
                row["hidden"].zero_()
                row["hidden"][:b - a].copy_(job.hrows[a:b])
            self.replay(key, row, job.graphs)
            hrows[a:b].copy_(row["output"][:b - a])
        job.metrics.append({"stage": self.index, "start": start, "state_end": state_end, "end": end,
                            "state_key": state_key, "row_chunks": len(l.row_chunks)})
        if self.index < 3:
            end.record(self.compute)
            return hstate, hrows, end
        # One batched pointer readout; no pageable H2D copy or synchronizing
        # torch.tensor(..., device=...) per question in the worker hot path.
        di = l.decide_picks.to(self.device, non_blocking=True)
        oi = l.option_picks.to(self.device, non_blocking=True)
        counts_gpu = l.option_counts.to(self.device, non_blocking=True)
        q, k = oi.shape
        hidden = hrows.flatten(0, 1)
        hd = hidden.index_select(0, di).float()
        ho = hidden.index_select(0, oi.flatten()).float()
        owner = torch.arange(q, device=self.device).repeat_interleave(k)
        z = self.head.many(hd, ho, owner).view(q, k)
        z.masked_fill_(torch.arange(k, device=self.device)[None] >= counts_gpu[:, None], -torch.inf)
        probabilities = torch.softmax(z, -1)
        end.record(self.compute)
        cpu = torch.empty((q, k), dtype=torch.float32, pin_memory=True)
        cpu.copy_(probabilities, non_blocking=True)
        done = torch.cuda.Event(); done.record(self.compute)
        return cpu, l.option_counts.tolist(), done

    def status(self):
        with torch.cuda.device(self.device):
            peak_allocated = allocated_bytes("cuda")
        return {"stage": self.index, "device": str(self.device), "layers": self.ids,
                "weights_bytes": self.weights_bytes, "bank_bytes": self.bank.bytes(), "row_buffer_bytes": self.rowbuf.bytes(),
                "graph_keys": [list(k) for k in self.graphs], "counters": dict(self.counters), "failures": self.failures,
                "allocated": torch.cuda.memory_allocated(self.device), "reserved": torch.cuda.memory_reserved(self.device),
                "peak_allocated": peak_allocated, "peak_reserved": torch.cuda.max_memory_reserved(self.device)}


class HostSlot:
    def __init__(self):
        self.state = self.rows = None

    def allocate(self, state, rows):
        if self.state is None or self.state.numel() < state.numel():
            self.state = torch.empty(state.numel(), dtype=state.dtype, pin_memory=True)
        if self.rows is None or self.rows.numel() < rows.numel():
            self.rows = torch.empty(rows.numel(), dtype=rows.dtype, pin_memory=True)
        return self.state[:state.numel()].view_as(state), self.rows[:rows.numel()].view_as(rows)


class Link:
    def __init__(self, engine, index):
        self.engine, self.index = engine, index
        self.free = queue.Queue()
        self.slots = [HostSlot(), HostSlot()]
        for slot in self.slots: self.free.put(slot)
        self.stream = torch.cuda.Stream(device=index)

    def send(self, job, state, rows, ready):
        while not self.engine.stopping.is_set():
            try:
                slot = self.free.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        else:
            raise RuntimeError("pipeline stopped during transfer")
        hs, hq = slot.allocate(state, rows)
        with torch.cuda.device(self.index), torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(self.stream)
            hs.copy_(state, non_blocking=True); hq.copy_(rows, non_blocking=True)
            b.record(self.stream)
            state.record_stream(self.stream); rows.record_stream(self.stream)
        job.hstate, job.hrows, job.ready = hs, hq, b
        job.relay = (self, slot, state, rows, a, b)
        self.engine.queues[self.index + 1].put(job)


class Engine:
    def __init__(self, config):
        self.config = config
        self.inflight = config["inflight"]
        self.graphs_enabled = config["graphs"]
        self.capture_enabled = False
        self.stopping = threading.Event()
        self.condition = threading.Condition()
        self.active = 0; self.next_id = 0; self.jobs = {}
        self.history = deque(maxlen=256)
        self.failed = None
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(8)
        if torch.cuda.device_count() != 4:
            raise ValueError("PP4 needs exactly four visible CUDA GPUs")
        self.checkpoint = Checkpoint(config["model"])
        meta = self.checkpoint.meta
        if meta.weights_dtype != "bf16" or meta.option_isolation or meta.special_embeddings:
            raise ValueError("PP4 needs a bf16 dense hybrid checkpoint without option isolation or special embeddings")
        self.tok, model = self.checkpoint.load(
            "cpu", LoadOptions(dtype=torch.bfloat16, backend="torch", attn="sdpa", fused=True, cuda_graphs=False),
            tokenizer_path=config.get("tokenizer_path"))
        if not model.hybrid or len(model.lm.layers) != 64:
            raise ValueError("PP4 needs a 64-layer dense Qwen3.5 text backbone")
        self.pad_id = model.pad_id
        self.stages = []
        for i in range(4):
            with torch.cuda.device(i):
                self.stages.append(Stage(model.lm, model.head, i, config, torch.device(f"cuda:{i}")))
        del model
        torch.set_num_threads(1)
        self.queues = [queue.Queue(maxsize=8) for _ in range(4)]
        self.links = [Link(self, i) for i in range(3)]
        self.retired, self.completed = queue.Queue(), queue.Queue()
        self.threads = [threading.Thread(target=self.worker, args=(i,), name=f"pp4-stage-{i}", daemon=True) for i in range(4)]
        self.reaper = threading.Thread(target=self.finish, name="pp4-completion", daemon=True)
        for thread in self.threads: thread.start()
        self.reaper.start()

    def encode(self, tok, rec, **kwargs):
        return encode(tok, rec, option_isolation=self.checkpoint.meta.option_isolation, **kwargs)

    def configure(self, inflight=None, graphs=None, capture=False):
        self.wait_idle()
        if inflight is not None: self.inflight = int(inflight)
        if graphs is not None: self.graphs_enabled = bool(graphs)
        self.capture_enabled = capture
        for stage in self.stages: stage.capture_enabled = capture

    def submit(self, encs, layout=None):
        layout = layout or make_layout(encs, self.pad_id, self.config)
        with self.condition:
            self.condition.wait_for(lambda: self.active < self.inflight or self.failed or self.stopping.is_set())
            if self.failed or self.stopping.is_set(): raise RuntimeError(self.failed or "engine stopped")
            self.active += 1; self.next_id += 1
            job = Job(self.next_id, layout, Future(), time.monotonic_ns(), graphs=self.graphs_enabled)
            job.future.set_running_or_notify_cancel()
            self.jobs[job.id] = job
        self.queues[0].put(job)
        return job.future

    def worker(self, index):
        stage = self.stages[index]
        try: os.sched_setaffinity(0, {index * 2})
        except OSError: pass
        with torch.cuda.device(index), torch.inference_mode(), torch.cuda.stream(stage.compute):
            while not self.stopping.is_set():
                try: job = self.queues[index].get(timeout=.1)
                except queue.Empty: continue
                try:
                    if index:
                        relay = job.relay
                        with torch.cuda.stream(stage.h2d):
                            stage.h2d.wait_event(job.ready)
                            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                            a.record(stage.h2d)
                            hs = torch.empty(job.hstate.shape, dtype=torch.bfloat16, device=stage.device)
                            hq = torch.empty(job.hrows.shape, dtype=torch.bfloat16, device=stage.device)
                            hs.copy_(job.hstate, non_blocking=True); hq.copy_(job.hrows, non_blocking=True)
                            b.record(stage.h2d)
                        stage.compute.wait_event(b)
                        hs.record_stream(stage.compute); hq.record_stream(stage.compute)
                        job.metrics.append({"link": index - 1, "d2h_start": relay[4], "d2h_end": relay[5],
                                            "h2d_start": a, "h2d_end": b,
                                            "bytes": (hs.numel() + hq.numel()) * 2})
                        self.retired.put((b, relay))
                        job.hstate, job.hrows, job.relay = hs, hq, None
                    started = time.monotonic_ns()
                    torch.cuda.nvtx.range_push(f"PP4 stage={index} job={job.id}")
                    outputs = stage.run_job(job)
                    torch.cuda.nvtx.range_pop()
                    job.metrics[-1]["host_issue_ns"] = time.monotonic_ns() - started
                    job.metrics[-1]["cpu_worker_gap_ms"] = (started-stage.last_issue_end_ns)/1e6 if stage.last_issue_end_ns else None
                    stage.last_issue_end_ns = time.monotonic_ns()
                    if index < 3:
                        self.links[index].send(job, *outputs)
                    else:
                        self.completed.put((job, *outputs))
                except Exception as exc:
                    self.fail(exc)
                finally:
                    self.queues[index].task_done()

    def fail(self, exc):
        with self.condition:
            self.failed = repr(exc)
            jobs = list(self.jobs.values())
            self.jobs.clear(); self.active = 0
            self.stopping.set(); self.condition.notify_all()
        for job in jobs:
            if not job.future.done(): job.future.set_exception(exc)

    def finish(self):
        try:
            self._finish()
        except Exception as exc:
            self.fail(exc)

    def _finish(self):
        retired, completed = [], []
        while not self.stopping.is_set() or retired or completed:
            try:
                while True: retired.append(self.retired.get_nowait())
            except queue.Empty: pass
            try:
                while True: completed.append(self.completed.get_nowait())
            except queue.Empty: pass
            pending = []
            for event, relay in retired:
                if event.query(): relay[0].free.put(relay[1])
                else: pending.append((event, relay))
            retired = pending
            pending = []
            for job, cpu, counts, event in completed:
                if not event.query():
                    pending.append((job, cpu, counts, event)); continue
                metrics = []
                for m in job.metrics:
                    if "stage" in m:
                        metrics.append({k: v for k, v in m.items() if k not in ("start", "state_end", "end")} | {
                            "state_ms": m["start"].elapsed_time(m["state_end"]),
                            "question_ms": m["state_end"].elapsed_time(m["end"]),
                            "compute_ms": m["start"].elapsed_time(m["end"])})
                    else:
                        metrics.append({"link": m["link"], "bytes": m["bytes"],
                                        "d2h_ms": m["d2h_start"].elapsed_time(m["d2h_end"]),
                                        "h2d_ms": m["h2d_start"].elapsed_time(m["h2d_end"])})
                rows = [cpu[r, :count].tolist() for r, count in enumerate(counts)]
                per_request = [[] for _ in job.layout.encs]
                for p, (owner, _, _) in zip(rows, job.layout.picks): per_request[owner].append(p)
                info = {"job": job.id, "batch": len(job.layout.encs), "created_ns": job.created_ns,
                        "completed_ns": time.monotonic_ns(), "metrics": metrics,
                        "state_tokens": job.layout.real_state_tokens, "row_tokens": job.layout.real_row_tokens,
                        "padded_state_tokens": job.layout.states.numel(),
                        "padded_row_tokens": sum(count_bucket(b-a)*job.layout.row_bucket for a,b in job.layout.row_chunks),
                        "prefix_copy_tokens": sum(count_bucket(b-a)*job.layout.prefix_bucket for a,b in job.layout.row_chunks)}
                self.history.append(info)
                with self.condition:
                    owned = self.jobs.pop(job.id, None)
                    if owned is not None: self.active -= 1
                    self.condition.notify_all()
                # Completion and failure claim a job under the same lock. The
                # winner alone resolves its Future, outside the scheduler lock.
                if owned is not None: job.future.set_result((per_request, info))
            completed = pending
            time.sleep(.0002)

    def wait_idle(self):
        with self.condition:
            self.condition.wait_for(lambda: not self.active or self.failed)
            if self.failed: raise RuntimeError(self.failed)

    def status(self):
        return {"pid": os.getpid(), "inflight": self.inflight, "active": self.active,
                "graphs_enabled": self.graphs_enabled, "capture_enabled": self.capture_enabled,
                "temperature": self.stages[3].head.temperature,
                "stages": [s.status() for s in self.stages], "failed": self.failed,
                "host_slots_bytes": sum((slot.state.numel() * 2 if slot.state is not None else 0) +
                                         (slot.rows.numel() * 2 if slot.rows is not None else 0)
                                         for link in self.links for slot in link.slots)}

    def close(self):
        self.stopping.set()
        if self.active:
            self.fail(RuntimeError("pipeline stopped"))
        with self.condition: self.condition.notify_all()
        for thread in self.threads: thread.join(timeout=5)
        self.reaper.join(timeout=5)
