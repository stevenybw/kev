"""Pipeline scheduling errors must not strand clients or exceed admission capacity."""
import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
import threading
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from kev.pp4 import PipelineOptions, PipelineServer
from kev.pp4_engine import plan_layout
from kev.api import SystemOneRequest
from kev import serve


class FakeEngine:
    def __init__(self):
        self.failed = None
        self.tok = None
        self.checkpoint = SimpleNamespace(meta=SimpleNamespace(option_isolation=False), release_date=lambda: '2026-10-01')
        self.jobs = []
        self.lock = threading.Lock()
        self.closed = False

    def submit(self, encs):
        done = Future()
        with self.lock:
            if self.closed: raise RuntimeError('engine stopped')
            self.jobs.append((encs, done))
        return done

    def wait_idle(self): pass

    def close(self):
        with self.lock:
            self.closed = True
            for _, done in self.jobs:
                if not done.done(): done.set_exception(RuntimeError('stopped'))

    def fail(self, exc):
        self.failed = str(exc)
        self.close()


def encoding(marker=1, state=20, branch=10, questions=1):
    ids = list(range(state))
    seg = [0] * state
    opt, decide = [], []
    for i in range(questions):
        start = len(ids)
        ids.extend([marker] * branch); seg.extend([i + 1] * branch)
        opt.append([start + branch - 3, start + branch - 2]); decide.append(start + branch - 1)
    return {'ids': ids, 'seg': seg, 'pos': list(range(len(ids))), 'opt_idx': opt, 'decide_idx': decide}


def service(monkeypatch, **settings):
    engine = FakeEngine()
    monkeypatch.setattr(PipelineServer, '_encode', lambda self, rec: rec)
    return PipelineServer('fake', PipelineOptions(**settings), engine=engine)


def wait_jobs(s, n=1):
    deadline = time.monotonic() + 2
    while len(s.engine.jobs) < n and time.monotonic() < deadline: time.sleep(.002)
    assert len(s.engine.jobs) >= n


def finish(s, index=0):
    encs, done = s.engine.jobs[index]
    done.set_result(([[[.25, .75]] for _ in encs], {}))


def test_capacity_includes_active_and_releases_cancelled_requests(monkeypatch):
    s = service(monkeypatch, capacity=1, batch_wait_ms=0)
    try:
        done = s.submit(encoding()); wait_jobs(s)
        done.cancel()
        with pytest.raises(HTTPException) as error: s.submit(encoding())
        assert error.value.status_code == 503
        finish(s); s.wait_idle()
        assert s.accepted == 0
        next_done = s.submit(encoding()); wait_jobs(s, 2); finish(s, 1)
        assert next_done.result(timeout=1)[0] == [[.25, .75]]
    finally: s.close()


def test_shutdown_completes_active_and_pending(monkeypatch):
    s = service(monkeypatch, batch_wait_ms=0)
    done = [s.submit(encoding(state=n)) for n in (20, 100, 300)]
    wait_jobs(s)
    s.close()
    for f in done:
        with pytest.raises(HTTPException) as error: f.result(timeout=1)
        assert error.value.status_code == 503
    assert s.accepted == 0
    with pytest.raises(HTTPException): s.submit(encoding())


def test_admission_rejects_many_questions_before_gpu_work(monkeypatch):
    s = service(monkeypatch)
    try:
        with pytest.raises(HTTPException) as error: s.submit(encoding(branch=400, questions=100))
        assert error.value.status_code == 422
        assert s.accepted == 0 and not s.engine.jobs
    finally: s.close()


def test_timeout_keeps_active_capacity_until_gpu_completion(monkeypatch):
    s = service(monkeypatch, capacity=1, request_timeout=.01, batch_wait_ms=0)
    monkeypatch.setattr('kev.pp4.to_record', lambda req: (encoding(), None))
    req = SystemOneRequest(state='x', questions={'a': {'type': 'noul'}})
    try:
        with pytest.raises(HTTPException) as error: asyncio.run(s.answer_async(req))
        assert error.value.status_code == 504
        assert s.accepted == 1
        finish(s); s.wait_idle()
        assert s.accepted == 0
    finally: s.close()


def test_stage_failure_resolves_all_clients(monkeypatch):
    s = service(monkeypatch, batch_wait_ms=0)
    try:
        futures = [s.submit(encoding()) for _ in range(3)]
        wait_jobs(s)
        s.engine.fail(RuntimeError('CUDA error'))
        for done in futures:
            with pytest.raises(HTTPException) as error: done.result(timeout=2)
            assert error.value.status_code == 503
        assert not s.healthy()
    finally: s.close()


def test_health_readiness_and_auth_do_not_expose_key(monkeypatch):
    monkeypatch.setattr(serve, 'API_KEY', 'test-key')
    monkeypatch.setattr(serve.app.state, 'server', SimpleNamespace(healthy=lambda: True, ready=False), raising=False)
    with TestClient(serve.app) as client:
        assert client.get('/healthz').status_code == 200
        assert client.get('/readyz').status_code == 503
        response = client.get('/v1/models')
        assert response.status_code == 401
        assert 'test-key' not in response.text
        assert response.headers['x-typesafe-request-id']
        assert client.get('/__benchmark/status').status_code == 404


def test_batcher_failure_releases_its_unsubmitted_batch(monkeypatch):
    s = service(monkeypatch, batch_wait_ms=100)
    original = s._fits
    calls = 0
    def fits(encs):
        nonlocal calls
        calls += 1
        if len(encs) > 1: raise RuntimeError('batch planning failed')
        return original(encs)
    monkeypatch.setattr(s, '_fits', fits)
    try:
        futures = [s.submit(encoding()) for _ in range(2)]
        for done in futures:
            with pytest.raises(HTTPException) as error: done.result(timeout=2)
            assert error.value.status_code == 503
        assert s.accepted == 0
    finally: s.close()


def test_batch_shape_counts_all_rows_and_padding():
    _, nb, sb, rows, lb, sr, _ = plan_layout([encoding(state=100, branch=200, questions=8)] * 2, vars(PipelineOptions()))
    assert nb * sb == 224 and len(rows) * lb == 3584 and sr + lb == 352


def test_default_capacity_refuses_request_257(monkeypatch):
    s = service(monkeypatch)
    try:
        futures = [s.submit(encoding()) for _ in range(256)]
        assert s.accepted == 256
        with pytest.raises(HTTPException) as error: s.submit(encoding())
        assert error.value.status_code == 503
        s.close()
        assert s.accepted == 0 and all(f.done() for f in futures)
    finally: s.close()


def test_http_deadline_and_capacity_statuses(monkeypatch):
    s = service(monkeypatch, capacity=1, request_timeout=.01, batch_wait_ms=0)
    monkeypatch.setattr('kev.pp4.to_record', lambda req: (encoding(), None))
    monkeypatch.setattr(serve, 'API_KEY', 'test-key')
    monkeypatch.setattr(serve.app.state, 'server', s, raising=False)
    body = {'state': 'x', 'questions': {'a': {'type': 'noul'}}}
    try:
        with TestClient(serve.app) as client:
            headers = {'Authorization': 'Bearer test-key'}
            assert client.post('/v1/systemone', json=body, headers=headers).status_code == 504
            assert client.post('/v1/systemone', json=body, headers=headers).status_code == 503
        finish(s); s.wait_idle()
        assert s.accepted == 0
    finally: s.close()
