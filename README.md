# QueryBot

**A local-first, RAG-powered knowledge assistant for company documents.**

QueryBot lets users upload documents, ask natural-language questions, and receive answers grounded in retrieved passages with source references. The first milestone is a dependable core product—not a collection of optional AI features.

## Project goals

- Ingest PDF, TXT, and Markdown documents.
- Split extracted text into useful chunks and preserve document/page metadata.
- Generate embeddings with a lightweight open model.
- Store chunks and embeddings in a persistent vector database.
- Retrieve relevant evidence for a user's question.
- Generate an answer from that evidence using a locally served open-weight language model.
- Return validated responses with citations to actual retrieved chunks.
- Clearly abstain when the indexed documents do not provide sufficient evidence.
- Provide a simple, usable interface and a documented API.

## Core technology choices

| Concern | Choice | Purpose |
|---|---|---|
| Language | Python 3.11+ | Application logic and RAG pipeline |
| API | FastAPI | Typed REST endpoints and interactive API docs |
| Request/response validation | Pydantic | Validate API data contracts |
| UI | Streamlit | Simple document upload and chat interface |
| PDF extraction | pypdf | Extract text and page numbers |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight 384-dimensional sentence embeddings |
| Vector store | ChromaDB | Persistent semantic search over document chunks |
| Local LLM runtime | Ollama | Run the generation model locally behind an HTTP API |
| Generation model | `qwen2.5:3b-instruct` (recommended starting point) | Lightweight instruction-following model; useful for grounded answers and structured output |
| Tests and quality | pytest, Ruff | Automated tests and consistent Python style |
| Version control | Git | Track changes and collaborate |

### Model notes

Use open-weight models locally to avoid requiring a paid hosted LLM API.

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`. It produces 384-dimensional vectors and is a compact, widely used baseline for semantic search. Its model card notes that inputs longer than 256 word pieces are truncated, so chunks should be kept appropriately sized. Model card: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- **Generation:** start with `qwen2.5:3b-instruct` through Ollama. Qwen2.5-3B-Instruct is designed for instruction following and structured data/JSON generation. Model card: <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct>
- **Alternative:** try `qwen3:4b-instruct` if your machine has enough memory and you want to compare answer quality. Ollama's listed quantized package is about 2.5 GB on disk; actual RAM use is higher and depends on context length and runtime settings. Model page: <https://ollama.com/library/qwen3:4b-instruct>

Model availability, quantization, and hardware performance can vary. Keep model names configurable rather than hard-coded throughout the codebase. Do not claim a model is “fully open source” without distinguishing open weights from training data and training code; document each model's license and usage terms in the project.

## Architecture

```text
                         ┌─────────────────────┐
                         │ Streamlit UI         │
                         │ Upload / Chat /      │
                         │ Sources / Feedback   │
                         └──────────┬──────────┘
                                    │ HTTP + JSON
                         ┌──────────▼──────────┐
                         │ FastAPI              │
                         │ Validation / Routes  │
                         └───────┬─────────────┘
                                 │
              ┌──────────────────┴───────────────────┐
              │                                      │
    ┌─────────▼─────────┐                  ┌─────────▼─────────┐
    │ Ingestion pipeline │                  │ Query pipeline    │
    │ Parse → Clean      │                  │ Question          │
    │ → Chunk → Embed    │                  │ → Retrieve        │
    │ → Index            │                  │ → Build context   │
    └─────────┬─────────┘                  │ → Generate answer │
              │                            │ → Validate sources│
    ┌─────────▼─────────┐                  └─────────┬─────────┘
    │ ChromaDB           │◄───────────────────────────┘
    │ Chunks + vectors   │
    │ + metadata         │
    └────────────────────┘
                                 │
                         ┌───────▼─────────┐
                         │ Ollama          │
                         │ Local LLM       │
                         └─────────────────┘
```

### Ingestion flow

1. Validate the uploaded file type and size.
2. Extract text and page metadata from the document.
3. Normalize whitespace and split text into overlapping chunks.
4. Attach stable IDs and metadata (document ID, filename, page, chunk index).
5. Generate embeddings with Sentence Transformers.
6. Upsert the chunks, embeddings, and metadata into persistent ChromaDB storage.
7. Record the document's indexing status and useful counts.

### Question-answering flow

1. Validate the question and selected document filters.
2. Resolve follow-up questions using only the relevant conversation history when needed.
3. Embed the search query and retrieve the top-k chunks from ChromaDB.
4. Apply a configurable relevance threshold and prepare a bounded context from retrieved evidence.
5. Ask the local LLM to answer only from the supplied context and to say when evidence is insufficient.
6. Validate the response schema and verify every citation against the actual retrieved chunk IDs.
7. Return answer, citations, evidence status, and request metadata as JSON.
8. Display the answer and expandable source excerpts in Streamlit.

**Important:** vector similarity is not a calibrated confidence score. Do not label a similarity score as “answer confidence.” RAG can still hallucinate; use evidence checks, citation validation, and evaluation tests.

## Product scope

### Milestone 1 — Core product (build this first)

- [x] FastAPI application starts and exposes `/health`.
- [x] Upload and index PDF, TXT, and Markdown files.
- [x] Persistent ChromaDB collection.
- [x] Document list endpoint and basic document status.
- [x] Semantic retrieval endpoint.
- [x] `/ask` endpoint that uses retrieved passages and the local LLM.
- [x] Validated JSON request/response schemas.
- [x] Source references grounded in retrieved chunks.
- [x] Clear response when no relevant evidence is found.
- [x] Streamlit interface for uploading documents and asking questions.
- [x] Basic tests for ingestion, retrieval, API validation, and citations.
- [x] README setup steps that a new developer can follow.

### Milestone 2 — Reliability and polish

Only after Milestone 1 works end-to-end:

- Duplicate upload detection and safe document deletion.
- Follow-up question handling with bounded conversation history.
- Logging, request IDs, latency measurements, and user feedback.
- Evaluation dataset with expected source documents/passages.
- Retrieval metrics such as Recall@k and a citation correctness review.
- Dockerfile or Compose setup, if useful for reproducibility.
- Optional hybrid search or reranking, only if evaluation shows a need.

### Out of scope for the first version

Do not begin with multi-agent workflows, complex orchestration frameworks, user/tenant management, OCR, elaborate analytics, a React frontend, or deployment infrastructure. These can be considered after the core product is reliable.

## Suggested repository structure

```text
querybot/
├── app/
│   ├── main.py                 # FastAPI app and route registration
│   ├── config.py               # Environment-based settings
│   ├── api/
│   │   ├── health.py
│   │   ├── documents.py
│   │   └── query.py
│   ├── schemas/                # Pydantic request/response models
│   ├── ingestion/              # Parsing, cleaning, chunking, indexing
│   ├── retrieval/              # Embeddings, vector store, retrieval
│   ├── rag/                    # Context builder and answer generation
│   └── services/               # Application/business logic
├── frontend/
│   └── streamlit_app.py
├── tests/
├── data/
│   └── .gitkeep
├── evaluation/
│   └── questions.json
├── .env.example
├── .gitignore
├── AGENTS.md
├── README.md
└── requirements.txt
```

Keep modules small and cohesive. Do not create empty abstractions or split code into files without a clear responsibility.

## API contract (initial)

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Report service health |
| `POST` | `/documents/upload` | Upload and index one document |
| `GET` | `/documents` | List indexed document records |
| `DELETE` | `/documents/{document_id}` | Delete a document and its vectors |
| `POST` | `/search` | Return retrieved chunks without generation |
| `POST` | `/ask` | Generate an evidence-grounded answer |
| `POST` | `/feedback` | Optional later milestone |

FastAPI's generated documentation should be available at `/docs` during development.

Example `/ask` response shape:

```json
{
  "answer": "The policy states ...",
  "sources": [
    {
      "document_id": "doc_123",
      "filename": "employee_handbook.pdf",
      "page": 12,
      "chunk_id": "chunk_456",
      "excerpt": "Relevant source passage..."
    }
  ],
  "insufficient_evidence": false,
  "request_id": "req_789"
}
```

This is a contract example, not hard-coded output. Citation metadata must be sourced from retrieved records, not invented by the LLM. If evidence is insufficient, return a clear answer with `insufficient_evidence: true` and no unsupported citations.

## Local development

Prerequisites:

- Python 3.11 or newer.
- Ollama installed and running.
- Enough RAM for the selected local generation model; 16 GB system RAM is a more comfortable target than 8 GB, particularly when other applications are open.
- Git.

Example setup (adjust commands to the operating system):

```bash
git clone <your-repository-url>
cd querybot

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

ollama pull qwen2.5:3b-instruct
```

After implementation, run the backend and frontend in separate terminals:

```bash
uvicorn app.main:app --reload
```

```bash
streamlit run frontend/streamlit_app.py
```

Run tests and lint checks:

```bash
pytest
ruff check .
```

These commands describe the intended workflow; the exact dependency list and entry points must match the implementation as it evolves.

## Configuration

Use environment variables for settings such as:

```dotenv
APP_ENV=development
CHROMA_PATH=./data/chroma
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b-instruct
TOP_K=5
CHUNK_SIZE=500
CHUNK_OVERLAP=80
```

Treat chunk size, overlap, top-k, and retrieval thresholds as configurable starting points, not universally optimal values. Validate them against the evaluation set. Never commit secrets, private documents, virtual environments, model files, or generated database files to Git.

## Quality and acceptance criteria

The core product is ready for a demo when:

- A user can upload a supported document and see indexing status.
- Indexed content remains searchable after restarting the application.
- A question retrieves relevant chunks and returns a grounded answer.
- Every displayed source points to a real retrieved chunk and correct document metadata.
- Unsupported questions produce a clear abstention rather than fabricated policy details.
- Invalid file types and malformed requests produce controlled errors.
- Document deletion removes associated vectors and records.
- Core ingestion, retrieval, API, and citation tests pass.
- A new developer can follow the setup instructions from a clean checkout.

## UI and product design principles

- Use a clean, restrained, professional interface.
- Prioritize readability, useful information hierarchy, and fast workflows.
- Prefer neutral surfaces, simple borders, consistent spacing, and a small typography scale.
- Avoid glassmorphism, excessive gradients, glowing effects, oversized rounded cards, gratuitous animation, decorative charts, fake metrics, and other visual patterns that look AI-generated.
- Do not add UI elements without a clear user need.
- Show loading, empty, success, and error states explicitly.
- Keep the source evidence easy to inspect.
- Make the app responsive and accessible; preserve keyboard navigation and readable contrast.

## Security and responsible handling

- Validate file type and size; do not trust the filename alone.
- Never execute uploaded files or treat document contents as system instructions.
- Treat retrieved document text as untrusted input. Defend against prompt injection by clearly separating system instructions from retrieved content and never allowing retrieved text to override system rules or trigger tools.
- Do not expose local filesystem paths or secrets in API responses.
- Do not log full private document contents or sensitive user queries by default.
- Make it clear that answers are generated and may require human verification for consequential decisions.
- If adding multiple users later, implement authorization and document-level access filtering before describing the app as multi-tenant or enterprise-secure.

## Project principles

1. Working core before optional features.
2. Evidence and correctness before visual polish.
3. Small modules, typed interfaces, and explicit error handling.
4. Configurable models and retrieval settings.
5. Tests for meaningful behavior, not just implementation details.
6. No fabricated benchmarks, metrics, citations, or production-readiness claims.
