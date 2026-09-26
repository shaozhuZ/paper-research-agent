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
    "Multilingual RAG-based Academic Research Assistant · Powered by LangGraph + Milvus + Gemini"
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

if st.button("Submit Query", use_container_width=True, type="primary"):
    if not query_text.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                #This sends the user's query and selected options to the backend API for processing and response generation.
                resp = requests.post(
                    f"{LANGGRAPH_API_BASE}/invoke",
                    json={
                        "query": query_text.strip(),
                        "language": language,
                        "domain": query_domain,
                    },
                    timeout=600,
                )
                resp.raise_for_status()

                result = resp.json()
                # store result
                st.session_state.last_result = {
                    "answer": result.get("answer", "(No answer received)"),
                    "papers": result.get("papers", []),
                    "recommended_papers": result.get("recommended_papers", []),
                    "language": result.get("language", language),
                }

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend.")
            except requests.exceptions.Timeout:
                st.error("Request timed out.")
            except Exception as e:
                st.error(f"An error occurred: {e}")


# Render Result
if st.session_state.last_result:
    res = st.session_state.last_result

    st.divider()
    st.caption(f"Response language: {res['language']}")

    st.subheader("Answer")
    st.info(res["answer"])
    # reference paper
    if res["papers"]:
        st.subheader("Reference Papers")
        for i, paper in enumerate(res["papers"][:2], 1):
            with st.container(border=True):
                st.markdown(f"**Paper {i}: {paper.get('title', 'Unknown title')}**")
                url = paper.get("url", "#")
                st.markdown(f"[{url}]({url})")
    else:
        st.info("No reference papers returned.")
    # recommended_papers
    if res["recommended_papers"]:
        st.subheader("Recommended Papers")
        for i, paper in enumerate(res["recommended_papers"][:2], 1):
            with st.container(border=True):
                st.markdown(f"**Paper {i}: {paper.get('title', 'Unknown title')}**")
                url = paper.get("url", "#")
                st.markdown(f"[{url}]({url})")
    else:
        st.info("No recommended papers returned.")