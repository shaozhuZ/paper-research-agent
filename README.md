# AI Research Assistant — RAG Pipeline with LangGraph & Milvus

> **Course Project — Option #1:** Building a Research Assistant Agent for Paper Recommendations with RAG Pipelines

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Component Descriptions](#3-component-descriptions)
4. [Indexed Paper Collection](#4-indexed-paper-collection)
5. [RAG Pipeline & Vector Database](#5-rag-pipeline--vector-database)
6. [Algorithm Comparison: HNSW vs IVF\_PQ vs DiskANN](#6-algorithm-comparison-hnsw-vs-ivf_pq-vs-diskann)
7. [Anti-Hallucination Strategy](#7-anti-hallucination-strategy)
8. [GKE Deployment & Auto-Scaling](#8-gke-deployment--auto-scaling)
9. [Local Setup & Running the Application](#9-local-setup--running-the-application)
10. [Environment Variables Reference](#10-environment-variables-reference)
11. [API Endpoints](#11-api-endpoints)
12. [Limitations & Assumptions](#12-limitations--assumptions)

---

## 1. Project Overview

This project implements a multilingual, AI-powered academic research assistant that combines:

- A **LangGraph-based AI agent** as the central orchestrator
- A **FastMCP tool server** exposing vector search, paper indexing, translation, and URL lookup
- A **Milvus vector database** for storing and querying paper embeddings
- A **translation microservice** (English ↔ Spanish ↔ French ↔ Italian)
- A **Streamlit web application** for paper upload and interactive querying

Users can upload research PDFs, classify them by domain (AI / Security / Other), and ask questions in any of four supported languages. The system retrieves relevant context from indexed papers, generates a grounded answer using Google Gemini, and returns the response — along with two reference papers and two recommended papers — in the same language as the query.

---

## Video Demonstrations

| Video | Link |
|-------|------|
| 🎬 Functional Demo | [Watch on Google Drive](https://drive.google.com/file/d/1WAJR-9c0TGYAs5xYudgf5YG1yrNYmmVG/view?usp=sharing) |
| 💻 Code Walkthrough | [Watch on Google Drive](https://drive.google.com/file/d/1IWi8nq0XTAKhtk53FBtC684qcUNB0XFm/view?usp=sharing) |
---

## 2. System Architecture

```

┌───────────────────────────────────────────────────────────────┐
│                        User (Browser)                         │
│                               │                               │
│                      ┌────────▼────────┐                      │
│                      │  Streamlit App  │  (GKE Pod, :8080)    │
│                      └────────┬────────┘                      │
│              HTTP POST /invoke, /upload  |  GET /stats        │
│                               │                               │
│              ┌────────────────▼────────────────┐              │
│              │  mcp-host (FastAPI + LangGraph)  │ (:8000)     │
│              │  • /invoke  → runs agent loop    │             │
│              │  • /upload  → indexes PDF        │             │
│              │  • /stats   → queries DB         │             │
│              └────────────────┬────────────────┘              │
│                    MCP streamable-http                        │
│                               │                               │
│              ┌────────────────▼──────────────────┐            │
│              │     mcp-server (FastMCP :9000)    │            │
│              │  • index_paper  • vector_search   │            │
│              │  • translate    • search_paper_url│            │
│              └──────┬──────────────┬─────────────┘            │
│                     │              │                          │
│            ┌────────▼──────┐  ┌───▼──────────────────┐        │
│            │  Translation  │  │   Milvus Vector DB   │        │
│            │  Service      │  │   (GKE Pod, :19530)  │        │
│            │  (GKE, :7000) │  │   HPA: 1–5 pods      │        │
│            └───────────────┘  └──────────────────────┘        │
│                                                               │
│              Semantic Scholar API  (external HTTPS)           │
│              └── called by mcp-server / search_paper_url      │
└───────────────────────────────────────────────────────────────┘

```

**Data Flow — User Query:**

1. User enters a query (any supported language) and selects a domain in Streamlit.
2. Streamlit sends `POST /invoke` to the FastAPI server.
3. FastAPI calls `run_agent_async()`, launching the LangGraph agent loop.
4. The agent:
   a. Detects the query language; if not English, calls `translate()` via MCP.
   b. Calls `vector_search()` with the English query and domain filter.
   c. Generates a grounded answer using only retrieved context (Gemini, temperature=0.2).
   d. Calls `search_paper_url()` for each recommended paper to resolve Semantic Scholar URLs.
   e. If the target language is not English, calls `translate()` to convert the answer.
5. The structured JSON response (answer + 2 reference papers + 2 recommended papers) is returned to Streamlit and rendered.

**Data Flow — Paper Upload:**

1. User selects PDF(s) and a research domain in the Streamlit sidebar.
2. Streamlit sends `POST /upload` (multipart form) to FastAPI.
3. FastAPI base64-encodes the PDF and calls `index_paper()` via MCP.
4. The MCP server extracts text, chunks it, generates Gemini embeddings, and stores vectors + metadata in Milvus.

---

## 3. Component Descriptions

### 3.1 Streamlit Web App (`streamlit/`)

- **File:** `app.py`
- **Port:** 8080
- **Dockerfile:** `python:3.10`, installs `streamlit` and `requests`
- **Features:**
  - Sidebar: multi-file PDF uploader, domain selector (AI / Security / Other), upload button, live index statistics (total and per-domain paper counts)
  - Main area: language selector (English / Spanish / French / Italian), domain filter for queries (including "All"), query text area
  - Response panel: answer block, reference papers with clickable Semantic Scholar URLs, recommended papers with URLs
  - Session-state upload history to prevent duplicate indexing

### 3.2 FastAPI Agent Server (`mcp-host/`)

- **File:** `src/api/apiserver.py`
- **Port:** 8000
- **Dockerfile:** `python:3.11-slim`, sets `PYTHONPATH=/app/src`
- **Key endpoints:** `GET /ok`, `POST /invoke`, `POST /upload`, `GET /stats`
- **Startup:** Pre-warms the LangGraph graph via `lifespan` hook before accepting requests

### 3.3 LangGraph AI Agent (`mcp-host/src/agent/graph.py`)

- **Framework:** LangGraph `StateGraph` with `AgentState` (message list)
- **LLM:** `ChatGoogleGenerativeAI` (Gemini) bound with MCP tools, temperature=0.2
- **Graph structure:** `llm` node → conditional edge (`tools_condition`) → `tools` node → back to `llm`; terminates when no more tool calls are needed
- **MCP transport:** `streamable_http` via `langchain-mcp-adapters`
- **System prompt:** Defines the multilingual research assistant role, explicit step-by-step workflow, strict JSON output format, and anti-hallucination rules

### 3.4 FastMCP Tool Server (`mcp-server/`)

- **File:** `mcp_tool_server.py`
- **Port:** 9000
- **Dockerfile:** `python:3.11-slim`, runs via `fastmcp run ... --transport streamable-http`
- **Exposed tools:**
  | Tool | Description |
  |------|-------------|
  | `translate` | Proxy to the translation service; no-op if source == target |
  | `index_paper` | Decodes base64 PDF, chunks text, embeds with Gemini, stores in Milvus |
  | `vector_search` | Cosine similarity search with optional domain filter |
  | `get_stats` | Returns per-domain unique paper counts from Milvus |
  | `search_paper_url` | Queries Semantic Scholar API (with retry on 429) for paper URLs |

### 3.5 Translation Service (`translation-service/`)

- **File:** `app.py`
- **Port:** 7000
- **Dockerfile:** `python:3.11-slim`, runs `uvicorn`
- **Library:** `deep-translator` (`GoogleTranslator`)
- **Supported languages:** English (`en`), Spanish (`es`), French (`fr`), Italian (`it`)
- **Endpoint:** `POST /translate` — accepts `{text, source_lang, target_lang}`, returns `{translated_text}`; short-circuits when source == target

### 3.6 Index Script (`index/milvus_index.py`)

A standalone batch script used exclusively for the algorithm comparison experiments (HNSW / IVF_PQ / DiskANN). It supports configurable index type via the `INDEX_TYPE` environment variable and outputs timing and storage metrics. All 30 papers for the main system were indexed through the Streamlit application.

---

## 4. Indexed Paper Collection

All 30 papers were uploaded through the Streamlit application and are stored with domain metadata in Milvus.

### AI Domain (10 papers)

| Paper |
|-------|
| Accelerating Large Language Model Decoding with Speculative Sampling |
| AN OPEN LARGE LANGUAGE MODEL FOR (OpenLLaMA) |
| A Survey on Large Language Model based Autonomous Agents |
| DeepSeek-Coder: When the Large Language Model Meets Programming |
| Distilling the Knowledge in a Neural Network |
| Efficient Memory Management for Large Language Models |
| EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks |
| Fault Detection Method based on Artificial Neural Networks |
| Machine Learning for Threat Intelligence |
| OpenAssistant Conversations: Democratizing Large Language Model Alignment |

### Security Domain (10 papers)

| Paper |
|-------|
| A Brick Wall, a Locked Door, and a Bandit (Adversarial Robustness) |
| An Orchestrator-Based Architecture For Multi-Agent Systems |
| Asleep at the Keyboard: Assessing the Security of GitHub Copilot's Code |
| Cyber-Security in Smart Grid |
| Firewall Security Policies, Testing and Performance Evaluation |
| Full Duplex Secrecy |
| Improving Cloud Network Security Using the Tree-Rule Firewall |
| Obfuscated Gradients Give a False Sense of Security |
| Off-Path TCP Sequence Number Inference Attack |
| Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic |

### Other Domain (10 papers)

| Paper |
|-------|
| Assessing the Environmental Impacts of Renewable Energy Sources |
| Davidson: Mindfulness on Brain and Immune Function |
| Efficacy of the Mindfulness Meditation Mobile App |
| Meditation States and Traits: EEG, ERP, and Neuroimaging Studies |
| The Benefits of Meditation and Mindfulness Practices |
| The Role of Renewable Energy in the Global Energy Transformation |
| The Socio-Economic Implications of the Coronavirus Pandemic |
| Updated World Map of the Köppen-Geiger Climate Classification |
| What Drives Housing Price Dynamics: Cross-Country Evidence |
| Zurich Toolbox for Ready-Made Economic Indicators |

---

## 5. RAG Pipeline & Vector Database

### Embedding Model
- **Model:** `gemini-embedding-001` (3072-dimensional vectors)
- **Provider:** Google Generative AI via `langchain-google-genai`

### Chunking Strategy
- **Library:** `RecursiveCharacterTextSplitter` from `langchain-text-splitters`
- **Chunk size:** 800 characters
- **Chunk overlap:** 120 characters
- **Separators:** `["\n\n", "\n", " ", ""]`

### Milvus Collection Schema
| Field | Type | Details |
|-------|------|---------|
| `id` | INT64 | Primary key, auto-assigned |
| `text` | VARCHAR(65535) | Chunk content |
| `filename` | VARCHAR(512) | Source PDF filename |
| `domain` | VARCHAR(64) | AI / Security / Other |
| `embedding` | FLOAT_VECTOR(3072) | Gemini embedding |

### Similarity Metric
All index types use **cosine similarity** (`metric_type: COSINE`).

### Domain-Aware Filtering
`vector_search()` accepts an optional `domain` parameter. When set, a Milvus `expr` filter (`domain == "<value>"`) is applied before the ANN search, ensuring retrieved context is domain-relevant.

---

## 6. Algorithm Comparison: HNSW vs IVF\_PQ vs DiskANN

### Benchmark Setup

- **Dataset:** 30 Research Papers (10 AI, 10 Security, 10 Other)
- **Total Chunks:** 2,998
- **Milvus Instance:** Milvus standalone on GKE
- **Collection Name:** `research_papers`
- **Embedding Model:** `gemini-embedding-001` (3072-dimensional vectors)
- **Similarity Metric:** Cosine similarity (all index types)

### Index Configuration Parameters

**HNSW**
```python
{"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
```
**IVF_PQ**
```python
{"index_type": "IVF_PQ", "metric_type": "COSINE", "params": {"nlist": 128, "m": 16, "nbits": 8}}
```
**DiskANN**
```python
{"index_type": "DISKANN", "metric_type": "COSINE", "params": {}}
```

### Results

| Index Type | Chunk Time (s) | Embedding Time (s) | Index Build Time (s) | Total Processing Time (s) | Index Size |
|------------|---------------:|-------------------:|---------------------:|--------------------------:|----------:|
| **HNSW**   | 17.75 | 46.77 | 4.42 | 68.94 | ~33.3 MiB |
| **IVF_PQ** | 24.62 | 47.60 | 6.19 | 78.41 | ~3.6 MiB  |
| **DiskANN**| 26.44 | 46.24 | 26.58 | 99.26 | ~39.7 MiB |

### Observations

- **Embedding time dominates** across all three algorithms (~47 s), as the 2,998 chunks must be sent to the Gemini API over the network. Index type has no effect on this phase.
- **HNSW** achieves the fastest index build time (4.42 s) and a moderate storage footprint (~33.3 MiB). It offers high recall with low query latency and is the default for live query serving in this system.
- **IVF_PQ** uses product quantization to compress vectors, yielding by far the smallest index (~3.6 MiB — roughly 9× smaller than HNSW). The trade-off is a modest reduction in recall accuracy due to lossy compression, making it ideal for memory-constrained deployments.
- **DiskANN** takes the longest to build (26.58 s) and produces the largest index (~39.7 MiB). It is designed for disk-resident billion-scale datasets; at 2,998 chunks the disk I/O overhead is apparent without a recall advantage over HNSW. It demonstrates the pipeline's extensibility to large corpora.

---

## 7. Anti-Hallucination Strategy

The system employs several complementary techniques to minimize hallucinations:

1. **Retrieval-grounded generation:** The system prompt explicitly instructs Gemini to generate answers using *only* the retrieved context chunks from `vector_search()`. The instruction "Never fabricate information. Only use what is in the retrieved context" is enforced as a hard rule.

2. **Low temperature:** The LLM is initialized with `temperature=0.2`, reducing output randomness and keeping the model closer to the retrieved evidence.

3. **Structured JSON output:** The agent is constrained to return a strict JSON schema `{answer, papers, recommended_papers}`. This prevents free-form narrative that might drift from the sources.

4. **Deterministic fallback for recommendations:** `fallback_recommendations()` performs a direct `vector_search()` call (bypassing the LLM) when the agent fails to return enough unique recommended papers, ensuring the final paper list is always database-derived.

5. **URL verification:** Paper URLs are resolved via the Semantic Scholar API rather than generated by the LLM. If no match is found, a search URL is returned instead of a hallucinated link.

6. **Domain filtering:** Retrieval is scoped to the user-selected domain, reducing the chance of retrieving off-topic chunks that could confuse the model.

---

## 8. GKE Deployment & Auto-Scaling

All services are containerized and deployed on Google Kubernetes Engine (GKE).

### Services and Ports

| Service | Container Port | GKE Service Type |
|---------|---------------|-----------------|
| `streamlit` | 8080 | LoadBalancer |
| `mcp-host` (FastAPI + LangGraph agent) | 8000 | ClusterIP |
| `mcp-server` (FastMCP tools) | 9000 | ClusterIP |
| `translation-service` | 7000 | ClusterIP |
| `milvus` | 19530 | ClusterIP |

### Milvus Horizontal Pod Autoscaler (HPA)

The Milvus vector database (`milvus-standalone`, namespace `milvus`) is configured with HPA using:

```bash
kubectl autoscale deployment milvus-standalone -n milvus --min=1 --max=5 --cpu-percent=70
```

Verified configuration:

```
NAMESPACE   NAME                REFERENCE                      TARGETS        MINPODS   MAXPODS   REPLICAS
milvus      milvus-standalone   Deployment/milvus-standalone   cpu: 29%/70%   1         5         1
```

### Agent Horizontal Pod Autoscaler

The LangGraph agent (`mco-host`, namespace `default`) also has HPA enabled (configured via GKE deployment):

```
NAMESPACE   NAME               REFERENCE             TARGETS       MINPODS   MAXPODS   REPLICAS
default     mco-host-hpa-eoeu  Deployment/mco-host   cpu: 0%/80%   1         5         1
```

### Building and Pushing Docker Images

```bash
# Set your GCP project
export PROJECT_ID=<your-gcp-project-id>
export REGION=us-central1

# Build and push each image
docker build -t gcr.io/$PROJECT_ID/streamlit:latest ./streamlit
docker push gcr.io/$PROJECT_ID/streamlit:latest

docker build -t gcr.io/$PROJECT_ID/mcp-host:latest ./mcp-host
docker push gcr.io/$PROJECT_ID/mcp-host:latest

docker build -t gcr.io/$PROJECT_ID/mcp-server:latest ./mcp-server
docker push gcr.io/$PROJECT_ID/mcp-server:latest

docker build -t gcr.io/$PROJECT_ID/translation-service:latest ./translation-service
docker push gcr.io/$PROJECT_ID/translation-service:latest
```

---

## 9. Local Setup & Running the Application

### Quick start (Docker Compose)

```bash
cp .env.example .env          # set GOOGLE_API_KEY
docker compose up -d --build  # Milvus + translation + mcp-server + mcp-host + streamlit
# put PDFs under papers/AI, papers/Security, papers/Other, then:
python scripts/ingest_papers.py --dir papers
```

UI at http://localhost:8080, API docs at http://localhost:8000/docs.

### Tests

```bash
cd mcp-host
pip install -r requirements-dev.txt
python -m pytest -q   # unit + integration; integration spins up a local fake MCP server, no API key needed
```

### Latency benchmark

```bash
python scripts/bench_latency.py --n 30 --concurrency 1 --label after
```

### Manual setup (without Docker Compose)

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- A Google AI Studio API key with access to `gemini-embedding-001` and `gemini-3-flash-preview` (or equivalent)
- A running Milvus instance (local or cloud)

### Step 1 — Clone the Repository

```bash
git clone <your-github-classroom-repo-url>
cd <repo-directory>
```

### Step 2 — Start Milvus Locally

```bash
# Using the official Milvus standalone Docker Compose
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
# Milvus will be available at http://localhost:19530
```

### Step 3 — Configure Environment Variables

Create a `.env` file (or export directly) with the required variables. **Replace all placeholder values:**

```bash
# Required for all services
export GOOGLE_API_KEY=<your-google-ai-studio-api-key>

# MCP Server
export MILVUS_URI=http://localhost:19530
export MILVUS_TOKEN=          # leave empty for local Milvus
export MILVUS_COLLECTION=research_papers
export INDEX_TYPE=HNSW        # HNSW | IVF_PQ | DISKANN
export TRANSLATION_SERVICE_URL=http://localhost:7000

# MCP Host (Agent)
export MCP_TOOL_URL=http://localhost:9000/mcp
export GEMINI_MODEL=gemini-3-flash-preview

# Streamlit
export LANGGRAPH_API_BASE=http://localhost:8000
```

### Step 4 — Start the Translation Service

```bash
cd translation-service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7000
```

### Step 5 — Start the MCP Tool Server

```bash
cd mcp-server
pip install -r requirements.txt
fastmcp run mcp_tool_server.py:mcp --transport streamable-http --host 0.0.0.0 --port 9000
```

### Step 6 — Start the FastAPI Agent Server

```bash
cd mcp-host
pip install -r requirements.txt
PYTHONPATH=src uvicorn api.apiserver:app --host 0.0.0.0 --port 8000
```

### Step 7 — Start the Streamlit App

```bash
cd streamlit
pip install -r requirements.txt
streamlit run app.py --server.port 8080
```

Open your browser at **http://localhost:8080**.

### Step 8 — Run Algorithm Comparison Experiments

To reproduce the HNSW / IVF_PQ / DiskANN comparison experiments, run the index script with the desired index type:

```bash
cd index
pip install langchain-google-genai langchain-text-splitters pymilvus pypdf

# Run with desired index type (change INDEX_TYPE to IVF_PQ or DISKANN for comparison)
GOOGLE_API_KEY=<key> MILVUS_URI=http://localhost:19530 INDEX_TYPE=HNSW PAPERS_DIR=papers python milvus_index.py
```

### Verifying Services

```bash
# Health checks
curl http://localhost:7000/ok           # Translation service
curl http://localhost:8000/ok           # Agent API
curl http://localhost:8000/stats        # Paper counts in Milvus

# Test translation directly
curl -X POST http://localhost:7000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source_lang": "English", "target_lang": "Spanish"}'

# Test agent invocation
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "What is HNSW?", "language": "English", "domain": "AI"}'
```

---

## 10. Environment Variables Reference

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `GOOGLE_API_KEY` | all | — | **Required.** Google AI Studio API key |
| `MILVUS_URI` | mcp-server, index | `http://localhost:19530` | Milvus connection URI |
| `MILVUS_TOKEN` | mcp-server, index | `""` | Milvus auth token (empty for local) |
| `MILVUS_COLLECTION` | mcp-server, index | `research_papers` | Milvus collection name |
| `INDEX_TYPE` | mcp-server, index | `HNSW` | Index algorithm: `HNSW`, `IVF_PQ`, or `DISKANN` |
| `GEMINI_EMB_MODEL` | mcp-server, index | `gemini-embedding-001` | Embedding model name |
| `GEMINI_MODEL` | mcp-host | `gemini-3-flash-preview` | Chat/generation model name |
| `TRANSLATION_SERVICE_URL` | mcp-server | `http://localhost:7000` | Translation service base URL |
| `MCP_TOOL_URL` | mcp-host | `http://localhost:9000/mcp` | MCP server endpoint |
| `LANGGRAPH_API_BASE` | streamlit | `http://localhost:8000` | Agent API base URL |
| `CHUNK_SIZE` | mcp-server, index | `800` | Text chunk size (characters) |
| `CHUNK_OVERLAP` | mcp-server, index | `120` | Chunk overlap (characters) |
| `TOP_K` | mcp-server | `4` | Number of vectors to retrieve per search |
| `SEMANTIC_SCHOLAR_API_KEY` | mcp-server | `""` | Optional; increases Semantic Scholar rate limit |

---

## 11. API Endpoints

### Agent Server (`mcp-host`, port 8000)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/ok` | Health check |
| POST | `/invoke` | Run the LangGraph agent on a query |
| POST | `/upload` | Index a PDF paper into Milvus |
| GET | `/stats` | Get per-domain paper counts |

**`POST /invoke` request body:**
```json
{
  "query": "¿Cuáles son los avances recientes en la búsqueda de arquitecturas neuronales?",
  "language": "Spanish",
  "domain": "AI"
}
```

**`POST /invoke` response:**
```json
{
  "answer": "La búsqueda de arquitecturas neuronales ha avanzado mediante...",
  "papers": [
    {"title": "EfficientNet Rethinking Model Scaling.pdf", "url": "https://www.semanticscholar.org/paper/..."},
    {"title": "Distilling the Knowledge in a Neural Network.pdf", "url": "https://www.semanticscholar.org/paper/..."}
  ],
  "recommended_papers": [
    {"title": "Accelerating Large Language Model Decoding.pdf", "url": "https://www.semanticscholar.org/paper/..."},
    {"title": "ASurvey on Large Language Model based Autonomous.pdf", "url": "https://www.semanticscholar.org/paper/..."}
  ],
  "language": "Spanish"
}
```

### Translation Service (port 7000)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/ok` | Health check |
| POST | `/translate` | Translate text between supported languages |

**`POST /translate` request:**
```json
{"text": "Hello", "source_lang": "English", "target_lang": "French"}
```

---

## 12. Limitations & Assumptions

### Limitations

1. **Paper count is fixed at 30.** The system was designed and tested for the 30 pre-indexed papers. Adding papers beyond this set requires re-running the index script or using the Streamlit upload feature, but performance characteristics (especially IVF_PQ with `nlist=128`) may degrade if the collection grows significantly without re-tuning.

2. **Language detection is user-driven.** The system does not auto-detect the query language. Users must manually select their language. Selecting a mismatched language will cause the translation step to operate incorrectly.

3. **Translation quality depends on Google Translate.** The `deep-translator` library wraps Google Translate's free tier, which may have rate limits for high-volume usage and may not perfectly preserve technical terminology.

4. **Semantic Scholar rate limiting.** The free Semantic Scholar API allows a limited number of requests per second. Under concurrent load, paper URL resolution may fall back to a search URL rather than a direct paper link. Setting `SEMANTIC_SCHOLAR_API_KEY` mitigates this.

5. **Context window constraints.** The system retrieves `TOP_K=4` chunks per query. For very broad questions, the 4-chunk context may be insufficient to generate a comprehensive answer. Increasing `TOP_K` improves coverage but increases latency and LLM token usage.

6. **No conversation memory.** Each query is stateless; the agent does not retain history across multiple questions in the same session.

7. **DiskANN on small datasets.** DiskANN is designed for disk-resident billion-scale indices. On 30 papers, the disk I/O overhead makes it slower to build without providing recall benefits over HNSW.

### Assumptions

1. All uploaded PDFs are text-readable (not scanned images). The system uses `pypdf` for text extraction and will raise an error if no text is found.

2. The Milvus instance is accessible and healthy before any service starts. There is no automatic retry or circuit breaker for Milvus connection failures at startup.

3. The `gemini-embedding-001` model produces 3072-dimensional vectors. The Milvus schema is hardcoded to `dim=3072`; using a different embedding model would require recreating the collection.

4. Users select the correct research domain when uploading papers. The system does not auto-classify papers by domain; misclassified papers will appear in wrong-domain searches.

5. GKE nodes have sufficient CPU and memory to run all services simultaneously. The recommended node pool uses `e2-standard-4` instances (4 vCPUs, 16 GB RAM).

### Design Trade-offs

- **FastMCP over direct REST calls:** Using MCP as the tool interface adds a protocol layer but enables the LangGraph agent to discover and invoke tools dynamically without hardcoded function wrappers, making the system more extensible.

- **Stateless agent per request:** Re-creating the LangGraph graph per invocation (`make_graph()`) ensures clean state isolation but incurs MCP client connection overhead on every request. A persistent graph with connection pooling would be more efficient at scale.

- **`deep-translator` over `googletrans`:** `googletrans` has known compatibility issues with `fastmcp`. `deep-translator` provides a stable, maintained wrapper around Google Translate with a consistent API.

- **Cosine similarity for all index types:** Cosine similarity normalizes for vector magnitude, making it more robust to variations in document length and embedding scale. This ensures fair comparison across all three index algorithms.
