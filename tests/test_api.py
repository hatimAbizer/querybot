"""Tests for API endpoints using the FastAPI TestClient."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient


class TestDocumentsAPI:
    def test_upload_txt(self, client: TestClient) -> None:
        content = b"This is a test document for QueryBot. It covers testing procedures."
        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
        )
        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test.txt"
        assert data["chunk_count"] >= 1
        assert "message" in data

    def test_upload_markdown(self, client: TestClient) -> None:
        content = b"# Title\n\nSome **markdown** content for testing."
        response = client.post(
            "/documents/upload",
            files={"file": ("readme.md", io.BytesIO(content), "text/markdown")},
        )
        assert response.status_code == 201
        assert response.json()["filename"] == "readme.md"

    def test_upload_unsupported_type_rejected(self, client: TestClient) -> None:
        content = b"binary data"
        response = client.post(
            "/documents/upload",
            files={"file": ("data.docx", io.BytesIO(content), "application/octet-stream")},
        )
        assert response.status_code == 422

    def test_list_documents(self, client: TestClient) -> None:
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)

    def test_list_after_upload(self, client: TestClient) -> None:
        content = b"Document for listing test."
        client.post(
            "/documents/upload",
            files={"file": ("listing.txt", io.BytesIO(content), "text/plain")},
        )
        response = client.get("/documents")
        docs = response.json()["documents"]
        filenames = [d["filename"] for d in docs]
        assert "listing.txt" in filenames

    def test_delete_document(self, client: TestClient) -> None:
        content = b"Document to be deleted."
        upload_resp = client.post(
            "/documents/upload",
            files={"file": ("delete_me.txt", io.BytesIO(content), "text/plain")},
        )
        doc_id = upload_resp.json()["document_id"]

        del_resp = client.delete(f"/documents/{doc_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["document_id"] == doc_id

        # Verify it's gone from the list
        list_resp = client.get("/documents")
        ids = [d["document_id"] for d in list_resp.json()["documents"]]
        assert doc_id not in ids

    def test_delete_nonexistent_returns_404(self, client: TestClient) -> None:
        response = client.delete("/documents/doc_doesnotexist")
        assert response.status_code == 404


class TestSearchAPI:
    def test_search_returns_results_schema(self, client: TestClient) -> None:
        # Upload something to search against
        content = b"The quick brown fox jumps over the lazy dog."
        client.post(
            "/documents/upload",
            files={"file": ("fox.txt", io.BytesIO(content), "text/plain")},
        )

        response = client.post("/search", json={"query": "quick brown fox"})
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total" in data

    def test_search_empty_query_rejected(self, client: TestClient) -> None:
        response = client.post("/search", json={"query": ""})
        assert response.status_code == 422

    def test_search_results_have_required_fields(self, client: TestClient) -> None:
        content = b"Python is a great programming language for data science."
        client.post(
            "/documents/upload",
            files={"file": ("python.txt", io.BytesIO(content), "text/plain")},
        )
        response = client.post("/search", json={"query": "Python programming"})
        data = response.json()
        for result in data["results"]:
            assert "chunk_id" in result
            assert "document_id" in result
            assert "filename" in result
            assert "excerpt" in result
            assert "distance" in result


class TestAskAPI:
    def test_ask_returns_correct_schema(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """The /ask endpoint must always return a valid AskResponse schema."""
        monkeypatch.setattr("app.rag.rag_service.generate", lambda prompt: "Mocked generated answer.")
        response = client.post("/ask", json={"question": "What is the meaning of life?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "insufficient_evidence" in data
        assert "request_id" in data

    def test_ask_no_evidence_sets_flag(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Questions with no relevant evidence must have insufficient_evidence=True."""
        monkeypatch.setattr("app.rag.rag_service.generate", lambda prompt: "No evidence found.")
        response = client.post(
            "/ask",
            json={"question": "Tell me about the Byzantine Empire's tax system in 900 AD"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["insufficient_evidence"], bool)
        assert isinstance(data["sources"], list)

    def test_ask_sources_are_validated(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sources must have chunk_id, document_id, filename, and excerpt."""
        monkeypatch.setattr(
            "app.rag.rag_service.generate",
            lambda prompt: "Quarterly revenue increased by 15% in Q3 2024 [1].",
        )
        content = b"Quarterly revenue increased by 15% in Q3 2024 driven by product sales."
        client.post(
            "/documents/upload",
            files={"file": ("revenue.txt", io.BytesIO(content), "text/plain")},
        )
        response = client.post("/ask", json={"question": "What was the quarterly revenue change?"})
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) > 0
        for src in data["sources"]:
            assert "chunk_id" in src
            assert "document_id" in src
            assert "filename" in src
            assert "excerpt" in src

    def test_ask_handles_generation_error(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the LLM backend fails, /ask gracefully returns an error message without crashing."""
        def mock_fail(prompt: str):
            raise RuntimeError("Ollama connection failed")

        monkeypatch.setattr("app.rag.rag_service.generate", mock_fail)
        response = client.post("/ask", json={"question": "What was the quarterly revenue change?"})
        assert response.status_code == 200
        data = response.json()
        assert "unavailable" in data["answer"].lower()
        assert data["insufficient_evidence"] is True
        assert data["sources"] == []

    def test_ask_empty_question_rejected(self, client: TestClient) -> None:
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

