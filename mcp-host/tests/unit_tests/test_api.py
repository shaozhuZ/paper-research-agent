import json

import pytest
from fastapi.testclient import TestClient

import api.apiserver as server


class FakeToolbox:
    async def healthy(self):
        return True

    async def call(self, name, **kw):
        if name == "get_stats":
            return {"AI": 3, "Security": 2, "Other": 1}
        if name == "index_paper":
            return {"filename": kw["filename"], "chunks_indexed": 5}
        raise AssertionError(name)


class FakeAgent:
    def __init__(self):
        self.toolbox = FakeToolbox()
        self.calls = []

    async def call_tool(self, name, **kw):
        return await self.toolbox.call(name, **kw)

    async def run(self, query, language, domain):
        self.calls.append((query, language, domain))
        if query == "boom":
            raise RuntimeError("vector_search failed: 503 from embedding API")
        return {
            "answer": "ok",
            "papers": [{"title": "p1", "url": "u1"}],
            "recommended_papers": [],
            "language": language,
        }

    async def stream(self, query, language, domain):
        if query == "boom":
            yield {"type": "delta", "text": "partial"}
            raise RuntimeError("model connection reset")
        for piece in ("Hel", "lo"):
            yield {"type": "delta", "text": piece}
        yield {"type": "done", "result": {"answer": "Hello", "papers": [], "language": language, "_private": 1}}

    async def close(self):
        pass


@pytest.fixture()
def client(monkeypatch, tmp_path):
    fake = FakeAgent()
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path)

    async def create(*a, **k):
        return fake

    monkeypatch.setattr(server.ResearchAgent, "create", staticmethod(create))
    with TestClient(server.app) as c:
        c.fake = fake
        c.upload_dir = tmp_path
        yield c


def test_invoke_contract_and_request_id(client):
    r = client.post("/invoke", json={"query": "q", "language": "French", "domain": "AI"},
                    headers={"x-request-id": "abc"})
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "abc"
    assert r.json()["papers"] == [{"title": "p1", "url": "u1"}]
    assert client.fake.calls == [("q", "French", "AI")]


def test_invoke_rejects_unknown_domain(client):
    assert client.post("/invoke", json={"query": "q", "domain": "Bio"}).status_code == 422


def test_upload_validates_domain(client):
    r = client.post("/upload", files={"file": ("a.pdf", b"%PDF")}, data={"domain": "Bio"})
    assert r.status_code == 400
    r = client.post("/upload", files={"file": ("a.pdf", b"%PDF")}, data={"domain": "AI"})
    assert r.status_code == 200 and r.json()["detail"]["chunks_indexed"] == 5
    # the PDF is saved under its content hash and passed to the tool by path
    saved = list(client.upload_dir.glob("*.pdf"))
    assert len(saved) == 1 and saved[0].read_bytes() == b"%PDF"


def test_stats_and_ok(client):
    assert client.get("/stats").json() == {"AI": 3, "Security": 2, "Other": 1}
    assert client.get("/ok").json() == {"status": "ok", "mcp": True}


def test_invoke_failure_reports_reason(client):
    r = client.post("/invoke", json={"query": "boom", "domain": "AI"})
    assert r.status_code == 502
    assert r.json()["detail"] == "agent failed: RuntimeError: vector_search failed: 503 from embedding API"


def _events(resp):
    return [json.loads(line[len("data: "):]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_invoke_stream_sends_deltas_then_result(client):
    r = client.post("/invoke/stream", json={"query": "q", "domain": "AI"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _events(r)
    assert [e["type"] for e in events] == ["delta", "delta", "done"]
    assert "".join(e["text"] for e in events[:2]) == "Hello"
    # the final event has the /invoke response shape (defaults filled, private keys dropped)
    assert events[-1]["result"]["answer"] == "Hello"
    assert events[-1]["result"]["contexts"] == [] and "_private" not in events[-1]["result"]


def test_invoke_stream_reports_errors_as_an_event(client):
    events = _events(client.post("/invoke/stream", json={"query": "boom", "domain": "AI"}))
    assert [e["type"] for e in events] == ["delta", "error"]
    assert events[-1]["detail"] == "agent failed: RuntimeError: model connection reset"
