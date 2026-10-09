"""QueryBot Streamlit frontend — document upload and question answering."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.getenv("QUERYBOT_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="QueryBot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def api_get(path: str, **kwargs):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the QueryBot API. "
            "Make sure the backend is running: `uvicorn app.main:app --reload`"
        )
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error: {exc.response.text}")
        return None


def api_post(path: str, **kwargs):
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=240, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the QueryBot API. "
            "Make sure the backend is running: `uvicorn app.main:app --reload`"
        )
        return None
    except requests.exceptions.ReadTimeout:
        st.error(
            "The request timed out. The model is still loading or the document is very large. "
            "Please wait a moment and try again."
        )
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error ({r.status_code}): {exc.response.text}")
        return None


def api_delete(path: str):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the QueryBot API.")
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error: {exc.response.text}")
        return None


# ---------------------------------------------------------------------------
# Sidebar — documents
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📄 QueryBot")
    st.caption("Local-first document assistant")
    st.divider()

    st.subheader("Upload document")
    uploaded = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md"],
        label_visibility="collapsed",
    )
    if uploaded and st.button("Index document", type="primary"):
        with st.spinner("Indexing…"):
            result = api_post(
                "/documents/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            )
        if result:
            st.success(
                f"✓ Indexed **{result['filename']}** "
                f"({result['chunk_count']} chunks)"
            )
            st.rerun()

    st.divider()
    st.subheader("Indexed documents")

    docs_data = api_get("/documents")
    documents = docs_data["documents"] if docs_data else []

    if not documents:
        st.info("No documents indexed yet.")
    else:
        for doc in documents:
            col1, col2 = st.columns([3, 1])
            with col1:
                label = doc["filename"]
                if len(label) > 28:
                    label = label[:25] + "…"
                st.markdown(
                    f"**{label}**  \n"
                    f"<small>{doc['chunk_count']} chunks"
                    + (f" · {doc['page_count']} pages" if doc.get("page_count") else "")
                    + "</small>",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("🗑", key=f"del_{doc['document_id']}", help="Delete"):
                    with st.spinner("Deleting…"):
                        api_delete(f"/documents/{doc['document_id']}")
                    st.rerun()


# ---------------------------------------------------------------------------
# Main area — question answering
# ---------------------------------------------------------------------------

st.header("Ask a question")

if not documents:
    st.info("Upload at least one document in the sidebar to get started.")
else:
    # Optional document filter
    doc_options = {d["filename"]: d["document_id"] for d in documents}
    selected_names = st.multiselect(
        "Filter by document (leave empty to search all)",
        options=list(doc_options.keys()),
    )
    selected_ids = [doc_options[n] for n in selected_names] if selected_names else None

    question = st.text_area(
        "Question",
        placeholder="What does the document say about…?",
        height=80,
        label_visibility="collapsed",
    )

    col_ask, col_search = st.columns([1, 1])
    ask_clicked = col_ask.button("Ask (with answer)", type="primary", use_container_width=True)
    search_clicked = col_search.button("Search (chunks only)", use_container_width=True)

    if ask_clicked and question.strip():
        with st.spinner("Thinking…"):
            result = api_post(
                "/ask",
                json={
                    "question": question.strip(),
                    "document_ids": selected_ids,
                },
            )
        if result:
            st.divider()
            if result["insufficient_evidence"]:
                st.warning("⚠️ Insufficient evidence in the indexed documents.")
            st.markdown("### Answer")
            st.markdown(result["answer"])

            if result["sources"]:
                st.markdown("### Sources")
                for src in result["sources"]:
                    label = src["filename"]
                    if src.get("page"):
                        label += f" — page {src['page']}"
                    with st.expander(label):
                        st.markdown(f"**Chunk:** `{src['chunk_id']}`")
                        st.markdown(src["excerpt"])

            st.caption(f"Request ID: `{result['request_id']}`")

    elif search_clicked and question.strip():
        with st.spinner("Searching…"):
            result = api_post(
                "/search",
                json={
                    "query": question.strip(),
                    "document_ids": selected_ids,
                },
            )
        if result:
            st.divider()
            st.markdown(f"### Retrieved chunks ({result['total']})")
            if not result["results"]:
                st.info("No relevant chunks found above the threshold.")
            for chunk in result["results"]:
                label = chunk["filename"]
                if chunk.get("page"):
                    label += f" — page {chunk['page']}"
                label += f"  (distance: {chunk['distance']:.3f})"
                with st.expander(label):
                    st.markdown(f"**Chunk:** `{chunk['chunk_id']}`")
                    st.markdown(chunk["excerpt"])

    elif (ask_clicked or search_clicked) and not question.strip():
        st.warning("Please enter a question.")
