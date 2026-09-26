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

    async def run(self, query, language, domain):
        self.calls.append((query, language, domain))
        return {
            "answer": "ok",
            "papers": [{"title": "p1", "url": "u1"}],
            "recommended_papers": [],
            "language": language,
        }

    async def close(self):
        pass


@pytest.fixture()
def client(monkeypatch):
    fake = FakeAgent()

    async def create(*a, **k):
        return fake

    monkeypatch.setattr(server.ResearchAgent, "create", staticmethod(create))
    with TestClient(server.app) as c:
        c.fake = fake
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


def test_stats_and_ok(client):
    assert client.get("/stats").json() == {"AI": 3, "Security": 2, "Other": 1}
    assert client.get("/ok").json() == {"status": "ok", "mcp": True}
