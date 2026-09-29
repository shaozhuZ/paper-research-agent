import json
import os
from collections import Counter

import requests
import streamlit as st


# Page setup
st.set_page_config(
    page_title="AI Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI Research Assistant")
st.caption(
    "Answers questions from the indexed papers, quoting the passages it used · LangGraph + MCP + Milvus"
)
st.divider()


# Backend endpoint
LANGGRAPH_API_BASE = os.getenv("LANGGRAPH_API_BASE", "http://localhost:8000").rstrip("/")


# Supported options
LANGUAGE_OPTIONS = ["English", "Spanish", "French", "Italian"]

DOMAIN_OPTIONS = ["AI", "Security", "Other"]
DOMAIN_LABELS = {
    "AI": "Artificial Intelligence (AI)",
    "Security": "Security",
    "Other": "Other",
}

QUERY_DOMAIN_OPTIONS = ["AI", "Security", "Other", "All"]
QUERY_DOMAIN_LABELS = {
    "AI": "AI",
    "Security": "Security",
    "Other": "Other",
    "All": "All Domains",
}

PLACEHOLDER_MAP = {
    "English": "e.g. What are the latest advances in neural architecture search?",
    "Spanish": "e.g. ¿Cuáles son los avances recientes en la búsqueda de arquitecturas neuronales?",
    "French": "e.g. Quelles sont les dernières avancées dans la recherche d'architecture neuronale?",
    "Italian": "e.g. Quali sono i recenti progressi nella ricerca di architetture neurali?",
}


# Session state
if "upload_history" not in st.session_state:
    st.session_state.upload_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "stats" not in st.session_state:
    st.session_state.stats = {}


# Cached stats fetch
@st.cache_data(ttl=60)
def fetch_stats(api_base: str):
    try:
        resp = requests.get(f"{api_base}/stats", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ───────────────── Sidebar ─────────────────
with st.sidebar:
    st.title("Paper Management")
    st.caption("Upload academic papers to the vector database")
    st.divider()

    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    domain = st.selectbox(
        "Research Domain",
        options=DOMAIN_OPTIONS,
        format_func=lambda x: DOMAIN_LABELS[x],
    )

    if st.button("Upload & Index Papers", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one PDF file first.")
        else:
            for uploaded_file in uploaded_files:
                if any(
                    h["name"] == uploaded_file.name
                    for h in st.session_state.upload_history
                ):
                    st.warning(f'"{uploaded_file.name}" already uploaded, skipping.')
                    continue

                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    try:
                        resp = requests.post(
                            f"{LANGGRAPH_API_BASE}/upload",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    "application/pdf",
                                )
                            },
                            data={"domain": domain},
                            timeout=120,
                        )

                        # If the status code is not right, going to the except logic
                        resp.raise_for_status()

                        st.session_state.upload_history.append(
                            {"name": uploaded_file.name, "domain": domain}
                        )

                        st.success(f"{uploaded_file.name} indexed to {domain}")

                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend.")
                    except Exception as e:
                        st.error(f"{uploaded_file.name} upload failed: {e}")
            # clear cache
            fetch_stats.clear()
            # The page updates immediately
            st.rerun()

    st.divider()
    st.subheader("Index Statistics")


    # Check the backend
    stats = fetch_stats(LANGGRAPH_API_BASE)

    if stats is None:
        stats = st.session_state.get("stats", {})
    else:
        st.session_state.stats = stats

    if not stats and st.session_state.upload_history:
        c = Counter(h["domain"] for h in st.session_state.upload_history)
        stats = {"AI": c["AI"], "Security": c["Security"], "Other": c["Other"]}

    # Total number 
    ai_n = stats.get("AI", 0)
    sec_n = stats.get("Security", 0)
    other_n = stats.get("Other", 0)
    total_n = ai_n + sec_n + other_n

    st.metric("Total Papers", f"{total_n} / 30")

    c1, c2, c3 = st.columns(3)
    c1.metric("AI", ai_n)
    c2.metric("Security", sec_n)
    c3.metric("Other", other_n)

    # if user upload something
    if st.session_state.upload_history:
        st.divider()
        st.subheader("Uploaded Papers")
        # disply the latest history
        for h in reversed(st.session_state.upload_history[-10:]):
            st.caption(f"{h['name']}  ·  {h['domain']}")


# ───────────────── Main Query Area ─────────────────
# seperate it into 2 columns
col_lang, col_dom = st.columns(2)

with col_lang:
    language = st.selectbox("Query Language", options=LANGUAGE_OPTIONS)

with col_dom:
    query_domain = st.selectbox(
        "Research Domain",
        options=QUERY_DOMAIN_OPTIONS,
        format_func=lambda x: QUERY_DOMAIN_LABELS[x],
        key="query_domain",
    )

query_text = st.text_area(
    "Enter your question",
    placeholder=PLACEHOLDER_MAP[language],
    height=120,
)

def stream_events(payload: dict):
    """Read server-sent events from /invoke/stream, one dict per event."""
    with requests.post(f"{LANGGRAPH_API_BASE}/invoke/stream", json=payload, stream=True,
                       timeout=(5, 300)) as resp:
        resp.raise_for_status()
        resp.encoding = "utf-8"
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                yield json.loads(line[len("data: "):])


if st.button("Submit Query", use_container_width=True, type="primary"):
    if not query_text.strip():
        st.warning("Please enter a question.")
    else:
        st.session_state.last_result = None
        st.divider()
        st.subheader("Answer")
        live = st.empty()
        live.caption("Searching the papers...")
        text, result = "", None
        try:
            payload = {"query": query_text.strip(), "language": language, "domain": query_domain}
            for event in stream_events(payload):
                if event["type"] == "delta":
                    text += event["text"]
                    live.info(text + " ▌")
                elif event["type"] == "done":
                    result = event["result"]
                elif event["type"] == "error":
                    st.error(event.get("detail", "The agent failed."))
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend.")
        except requests.exceptions.Timeout:
            st.error("Request timed out.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

        if result is not None:
            st.session_state.last_result = result
            st.rerun()  # redraw with the full result (sources, passages) below


def show_papers(title: str, papers: list) -> None:
    if not papers:
        return
    st.subheader(title)
    for paper in papers[:2]:
        with st.container(border=True):
            st.markdown(f"**{paper.get('title', 'Unknown title')}**")
            if paper.get("url"):
                st.markdown(f"[{paper['url']}]({paper['url']})")


# Render Result
if st.session_state.last_result:
    res = st.session_state.last_result
    usage = res.get("usage") or {}

    st.divider()
    st.subheader("Answer")
    st.info(res.get("answer") or "(No answer received)")
    meta = [f"{usage['latency_ms'] / 1000:.1f} s" if usage.get("latency_ms") else "",
            f"{usage.get('llm_turns', 0)} model call(s)" if usage else "",
            usage.get("model", ""), f"answer in {res.get('language', language)}"]
    st.caption(" · ".join(m for m in meta if m))

    # the passages the answer is based on, so the reader can check it
    contexts = res.get("contexts") or []
    cited = [n for n in usage.get("cited", []) if 1 <= n <= len(contexts)]
    if contexts:
        st.subheader("Passages used")
        order = cited or list(range(1, len(contexts) + 1))
        for n in order:
            c = contexts[n - 1]
            with st.expander(f"[{n}] {c['filename']} · chunk {c['chunk_id']}", expanded=(n == order[0])):
                st.write(c["content"])
        others = len(contexts) - len(order)
        if others > 0:
            st.caption(f"{others} more passage(s) were retrieved but not cited.")

    show_papers("Reference Papers", res.get("papers", []))
    show_papers("Recommended Papers", res.get("recommended_papers", []))
