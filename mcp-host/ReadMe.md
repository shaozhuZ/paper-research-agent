# mcp-host: FastAPI + LangGraph agent

- `src/agent/graph.py` — the agent graph and `ResearchAgent` (owns the MCP session and compiled graph for the process lifetime)
- `src/agent/mcp_tools.py` — long-lived MCP session, direct tool calls, health check, reconnect support
- `src/agent/parsing.py` — pure helpers that turn model output into the API contract
- `src/api/apiserver.py` — HTTP endpoints, request-id middleware, JSON access logs

Run locally:

```bash
pip install -r requirements.txt
MCP_TOOL_URL=http://localhost:9000/mcp GOOGLE_API_KEY=... PYTHONPATH=src uvicorn api.apiserver:app --port 8000
```

Tests (no network, no API key):

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
