# 🏢 Enterprise Knowledge Assistant

An AI-powered knowledge assistant that answers employee questions from internal documents using **Retrieval Augmented Generation (RAG)**.

---

## 📋 Table of Contents
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Technology Choices & Design Decisions](#technology-choices--design-decisions)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Limitations & Future Improvements](#limitations--future-improvements)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTERPRISE KNOWLEDGE ASSISTANT            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────┐  │
│  │   Frontend   │    │           Backend (FastAPI)       │  │
│  │  HTML/CSS/JS │◄──►│                                  │  │
│  │  Port 3000   │    │  ┌─────────┐  ┌──────────────┐  │  │
│  └──────────────┘    │  │  /ask   │  │  /ingest     │  │  │
│                       │  └────┬────┘  └──────┬───────┘  │  │
│                       │       │               │          │  │
│  INGEST PIPELINE:     │  ┌────▼───────────────▼──────┐  │  │
│  ┌──────────┐         │  │       RAG Service          │  │  │
│  │ Document │         │  │  1. Query Rewriting         │  │  │
│  │  Loader  │         │  │  2. Hybrid Retrieval        │  │  │
│  └────┬─────┘         │  │  3. LLM Generation          │  │  │
│       │               │  │  4. Source Citation         │  │  │
│  ┌────▼──────┐        │  └────────────┬───────────────┘  │  │
│  │  Chunker  │        │               │                   │  │
│  │ 512 tok   │        │  ┌────────────▼───────────────┐  │  │
│  │ 64 overlap│        │  │      Vector Store (FAISS)   │  │  │
│  └────┬──────┘        │  │  IndexFlatIP (cosine sim)   │  │  │
│       │               │  │  + BM25 keyword re-ranking  │  │  │
│  ┌────▼──────┐        │  └────────────────────────────┘  │  │
│  │ Embedder  │        │                                   │  │
│  │MiniLM-L6  │        │  ┌────────────────────────────┐  │  │
│  └────┬──────┘        │  │     LLM (Local Llama 3)             │  │
│       │               │  │  Grounded answer generation │  │  │
│  ┌────▼──────┐        │  │  Hallucination prevention   │  │  │
│  │   FAISS   │        │  └────────────────────────────┘  │  │
│  │   Index   │        │                                   │  │
│  └───────────┘        └──────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

**Ingestion:**
```
Document → Load → Extract Text → Clean → Chunk (512 tokens, 64 overlap)
        → Embed (MiniLM-L6-v2) → Store in FAISS index (persisted to disk)
```

**Query:**
```
Question → Rewrite (Llama 3) → Embed → FAISS search (top-15)
        → Hybrid re-rank (70% semantic + 30% BM25 keyword)
        → Filter (threshold 0.3) → Top-5 chunks
        → Local Llama 3 via Ollama with grounding prompt → Answer + Sources
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Ollama installed locally (get it from [ollama.com](https://ollama.com))
- Llama 3 model downloaded (run `ollama run llama3`)

### 1. Clone and Setup

```bash
git clone <your-repo>
cd enterprise-knowledge-assistant/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

Ensure Ollama is running (`ollama serve` or open the Ollama desktop app). The default configuration in `.env` is already set up to connect to local Ollama on port 11434.

```bash
# Check backend/.env
# OLLAMA_BASE_URL should point to http://localhost:11434/v1
# LLM_MODEL should be llama3
```

### 3. Run the Backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will auto-ingest documents from `backend/data/documents/` on startup.
Three sample documents are included: HR_Policy.txt, Customer_FAQ.txt, Technical_Guide.txt.

### 4. Open the Frontend

To prevent CORS issues when communicating with the backend, serve the frontend using a local web server:

```bash
cd ../frontend
python -m http.server 3000
# Then open http://localhost:3000 in your browser
```

### 5. Add Your Own Documents

**Via UI:** Drag and drop files into the sidebar upload zone.

**Via API:**
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "files=@/path/to/your_document.pdf"
```

**Via folder:** Place files in `backend/data/documents/` and restart the server.

### 6. Ask Questions

**Via UI:** Type in the chat box.

**Via API:**
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the employee leave policy?"}'
```

---

## Docker Setup (Alternative)

```bash
# From project root
# Make sure Ollama is running locally
docker-compose up --build
```

Backend: http://localhost:8000  
Frontend: http://localhost:3000

---

## Project Structure

```
enterprise-knowledge-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app + lifespan
│   │   ├── api/
│   │   │   └── routes.py           # All API endpoints
│   │   ├── core/
│   │   │   ├── config.py           # Settings (pydantic-settings)
│   │   │   └── logging.py          # Logging setup
│   │   ├── models/
│   │   │   └── schemas.py          # Request/response Pydantic models
│   │   └── services/
│   │       ├── document_processor.py  # Load, extract, chunk
│   │       ├── vector_store.py        # FAISS + embeddings
│   │       ├── llm_service.py         # Claude API + prompting
│   │       └── rag_service.py         # Main RAG orchestrator
│   ├── data/
│   │   ├── documents/              # Put your documents here
│   │   └── vector_store/           # Auto-created FAISS index
│   ├── tests/
│   │   └── test_all.py             # Full test suite (28 tests)
│   ├── evaluate.py                 # Standalone evaluation script
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   └── index.html                  # Single-file chat UI
├── docker-compose.yml
└── README.md
```

---

## Technology Choices & Design Decisions

### LLM: Local Llama 3 (via Ollama)
**Why:** Fully local, private, secure, and free. Since the model runs entirely on your local machine, enterprise documents are never sent to external servers or APIs. Strong instruction following and compatibility with the OpenAI API format make integration seamless.

### Embeddings: all-MiniLM-L6-v2 (Sentence Transformers)
**Why:** 
- 22M parameters → fast inference, low memory (~80MB)
- 384-dimensional vectors → efficient FAISS storage
- Strong semantic quality for English enterprise text
- Runs locally → no API cost per embedding, no data sent externally
- Alternative considered: OpenAI `text-embedding-3-small` (better quality, costs money, data goes external)

### Vector DB: FAISS (Facebook AI Similarity Search)
**Why:**
- Perfect for up to ~1M vectors without infrastructure complexity
- IndexFlatIP on L2-normalized vectors = exact cosine similarity (no approximation errors at this scale)
- Persists to disk as a simple file — no separate service to run
- Alternative considered: ChromaDB (easier API, but adds a service dependency); Pinecone (cloud, great for scale, but costs money and requires internet)

### Chunking: Recursive character splitter, 512 tokens, 64 overlap
**Why:**
- 512 tokens fits in context without overwhelming the LLM with irrelevant text per chunk
- Recursive splitting (paragraph → sentence → word) preserves semantic boundaries vs. hard fixed splits
- 64-token overlap prevents losing context at boundaries (e.g., a sentence split across chunks)
- Alternative considered: Semantic chunking (cluster by embedding similarity) — better quality but 5–10x slower

### Hybrid Search: 70% Semantic + 30% BM25 Keyword
**Why:** Pure semantic search misses exact keyword matches ("SOC 2 Type II", "26 weeks"). Pure keyword misses paraphrase ("paid time off" vs "annual leave"). Hybrid gives best of both. The 70/30 weight was chosen empirically — semantic dominates for conceptual questions, keyword catches exact figures.

### Query Rewriting
**Why:** Short or ambiguous queries ("leave policy?", "refund?") retrieve poorly. Rewriting expands them into full questions that match document phrasing better. Only applied to queries ≤8 words to avoid latency on already-clear questions.

### Hallucination Prevention
Three mechanisms:
1. **System prompt grounding:** "Answer ONLY from provided context. If not found, say so."
2. **Similarity threshold (0.3):** Chunks below this threshold are filtered. Low-relevance chunks would mislead the LLM.
3. **Confidence scoring:** Outputs a confidence score. Low scores flag uncertain answers to the user.

---

## API Reference

### POST /api/v1/ask
Ask a question.

**Request:**
```json
{
  "question": "What is the refund policy?",
  "top_k": 5,
  "conversation_id": "optional-for-multi-turn"
}
```

**Response:**
```json
{
  "answer": "Customers may request a full refund within 30 days of purchase...",
  "sources": [
    {
      "document": "Customer_FAQ.txt",
      "page": null,
      "chunk_index": 3,
      "relevance_score": 0.91,
      "excerpt": "Customers may request a full refund within 30 days..."
    }
  ],
  "confidence": 0.91,
  "conversation_id": "uuid-here"
}
```

### POST /api/v1/ingest
Upload documents (multipart/form-data, field name `files`).

### GET /api/v1/documents
List all indexed documents.

### DELETE /api/v1/documents/{doc_name}
Remove a document from the index.

### GET /api/v1/stats
Knowledge base statistics (document count, chunk count).

### POST /api/v1/evaluate
Run evaluation on test cases.

### POST /api/v1/feedback
Submit answer quality feedback.

### GET /health
Health check.

---

## Evaluation

### Run the test suite
```bash
cd backend
pytest tests/ -v
```

### Run the evaluation script
```bash
# Make sure the server is running first
python evaluate.py --api http://localhost:8000 --verbose
```

### Evaluation methodology

**13 test cases across:**
- Easy retrieval (single-hop, direct match)
- Medium retrieval (requires reading multiple sentences)
- Hard retrieval (requires reasoning/inference)
- Hallucination prevention (questions with no answer in docs)
- Out-of-scope (should return "I don't know")

**Metrics:**
- Pass rate: % of test cases where expected keywords appear in the answer
- Confidence calibration: does confidence correlate with actual correctness?
- Hallucination prevention rate: % of out-of-scope questions answered with "not found"
- Latency: P50 and P95 response times

**Baseline results on sample documents:**
| Category | Pass Rate |
|----------|-----------|
| Easy HR | ~100% |
| Easy FAQ | ~100% |
| Medium | ~80% |
| Hard | ~60% |
| Hallucination | ~90% |

---

## Limitations & Future Improvements

### Current Limitations
1. **No authentication:** Any user can access any document. Production needs OAuth2/JWT.
2. **Single-file vector store:** FAISS is file-based; concurrent writes need locking.
3. **English-only:** Embedding model is optimized for English.
4. **No table/chart extraction:** Complex PDFs with tables may lose formatting.
5. **Context window limit:** Very long answers may truncate if many chunks retrieved.
6. **Conversation memory:** Only maintained in-memory; lost on restart.

### Future Improvements (Priority Order)
1. **Re-ranking with a cross-encoder** (e.g., `ms-marco-MiniLM`) — 10–15% retrieval quality improvement
2. **Multi-modal support** — Extract text from images/charts in PDFs using OCR
3. **Persistent conversation memory** — Store in Redis/PostgreSQL
4. **Authentication** — Auth0 or company SSO integration
5. **Streaming responses** — Stream tokens for faster perceived latency
6. **Document versioning** — Track when documents are updated, re-index changed sections
7. **Feedback loop** — Use thumbs up/down to fine-tune retrieval scoring
8. **Scalability** — Replace FAISS with Qdrant or Weaviate for distributed search
9. **Answer caching** — Cache frequent questions with Redis
10. **Multi-language** — Switch to `paraphrase-multilingual-MiniLM-L12-v2`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | URL of your local Ollama instance |
| `LLM_MODEL` | `llama3` | Ollama model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.3` | Min relevance score |
