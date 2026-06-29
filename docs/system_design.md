# System Design Document
## Enterprise Knowledge Assistant — RAG System

**Version:** 1.0 | **Date:** June 2026

---

## 1. High-Level Architecture

The system follows a classic **two-phase RAG architecture**:

**Phase 1 — Offline Ingestion (one-time / on upload):**
Documents are loaded, cleaned, split into overlapping chunks, embedded as dense vectors, and stored in a FAISS index that is persisted to disk.

**Phase 2 — Online Query (real-time):**
User questions are optionally rewritten, embedded, matched against the vector index using hybrid (semantic + keyword) search, filtered by relevance threshold, and passed to Claude claude-sonnet-4-6 with a strict grounding prompt. The LLM generates an answer citing only the retrieved context.

---

## 2. Data Flow

```
INGESTION:
File Upload ──► Document Loader ──► Text Cleaner ──► Recursive Chunker
              (PDF/TXT/DOCX/MD)   (normalize WS,    (512 chars, 64 overlap,
                                   strip ctrl chars)  paragraph-first strategy)
                    │
                    ▼
              Sentence Transformer ──► FAISS IndexFlatIP ──► Disk (faiss.index
              (all-MiniLM-L6-v2)       (L2-normalized,         + chunks.pkl)
              384-dim embeddings         cosine similarity)

QUERY:
Question ──► Query Rewriter ──► Embedder ──► FAISS.search(top-15)
             (Claude, optional,  (same model)
              for short queries)
                    │
                    ▼
              BM25 Keyword Scorer ──► Hybrid Re-ranker ──► Threshold Filter
              (TF on query terms)      (0.7 × semantic +     (score ≥ 0.3)
                                        0.3 × keyword)
                    │
                    ▼
              Top-5 Chunks ──► Context Builder ──► Claude claude-sonnet-4-6
                               (format with         (grounded system prompt)
                                doc/page metadata)
                    │
                    ▼
              Answer + Sources + Confidence Score ──► API Response ──► UI
```

---

## 3. Component Explanation

| Component | Technology | Responsibility |
|-----------|-----------|----------------|
| **API Server** | FastAPI + uvicorn | REST endpoints, request validation, error handling |
| **Document Processor** | pypdf, python-docx | File loading, text extraction, recursive chunking |
| **Embedding Engine** | sentence-transformers | Dense vector generation (384-dim, normalized) |
| **Vector Store** | FAISS (IndexFlatIP) | Fast cosine similarity search over all chunks |
| **Hybrid Ranker** | Custom BM25 + FAISS | Re-ranks results combining semantic + lexical signals |
| **LLM Service** | Anthropic Claude claude-sonnet-4-6 | Grounded answer generation, query rewriting |
| **RAG Orchestrator** | Python | Coordinates all services end-to-end |
| **Frontend** | Vanilla HTML/CSS/JS | Chat UI, document management, source display |

---

## 4. Key Design Decisions

**Chunking strategy:** Recursive character splitting is preferred over fixed-size splitting because it naturally respects document structure (paragraphs → sentences → words), preserving semantic coherence within each chunk. 512-character chunks are long enough to contain meaningful information but short enough to stay focused. 64-character overlap prevents losing context at boundaries.

**FAISS over ChromaDB/Pinecone:** For this use case (100–10,000 chunks, single-server deployment), FAISS provides excellent performance with zero infrastructure overhead. It persists to a single file and loads in milliseconds. At 1M+ chunks or multi-node requirements, Qdrant or Pinecone would be preferable.

**Hybrid search:** Enterprise documents contain many exact-match terms (policy codes, product names, numbers). Pure dense retrieval underperforms on these. The 70/30 semantic/keyword split was chosen to prioritize conceptual understanding while recovering from literal keyword misses.

**Grounding prompt design:** The system prompt explicitly states the LLM must answer only from context and must acknowledge when information is unavailable. This single design decision is the primary hallucination prevention mechanism.

---

## 5. Scalability Considerations

**Current limits (single server):**
- FAISS handles ~1M vectors on a single machine with ~4GB RAM
- Suitable for: ~5,000 documents × ~200 chunks = 1M chunks
- Latency: ~200ms retrieval + ~1s LLM = ~1.2s end-to-end

**To scale to 10M+ chunks:**
- Replace FAISS with **Qdrant** (distributed, supports filtering, built-in payload storage)
- Add **Redis** for conversation history and answer caching
- Introduce a **task queue (Celery + Redis)** for async document ingestion
- Use **PostgreSQL** for chunk metadata instead of pickle files
- Deploy behind a **load balancer** with stateless API instances

**To scale LLM throughput:**
- Use Anthropic **Batch API** for bulk evaluation queries
- Cache frequent question-answer pairs in Redis (TTL: 1 hour)
- Add streaming response support for faster perceived latency

**Estimated capacity at current design:**
| Metric | Value |
|--------|-------|
| Max documents | ~5,000 (200 chunks each) |
| Max chunks | ~1M |
| Concurrent users | ~50 (single server) |
| P50 latency | ~1.2s |
| P95 latency | ~3s |
