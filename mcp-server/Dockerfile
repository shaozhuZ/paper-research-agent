FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy server code
COPY mcp_tool_server.py chunking.py retrieval.py migrate_to_hybrid.py /app/

# Expose MCP HTTP port
EXPOSE 9000

# Run MCP server (HTTP transport)
# Endpoint will be: http://<host>:8000/mcp
CMD ["fastmcp", "run", "mcp_tool_server.py:mcp", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "9000"]