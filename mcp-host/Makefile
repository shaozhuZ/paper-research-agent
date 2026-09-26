.PHONY: test unit integration run

# Unit + integration tests. Integration tests start a local fake MCP server; no network or API key needed.
test:
	python -m pytest -q

unit:
	python -m pytest -q tests/unit_tests

integration:
	python -m pytest -q tests/integration_tests

run:
	PYTHONPATH=src uvicorn api.apiserver:app --host 0.0.0.0 --port 8000
