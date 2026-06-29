"""
Vector Store Service
Uses FAISS for fast similarity search + sentence-transformers for embeddings.

Design choices:
- FAISS (IndexFlatIP with normalized vectors = cosine similarity): 
  Fast, local, no external API needed, works great for 100k chunks
- all-MiniLM-L6-v2: lightweight (22M params), strong semantic quality,
  384-dim vectors, perfect for enterprise doc Q&A
- Persisted to disk: survives restarts without re-embedding
- Hybrid search: cosine similarity + BM25 keyword scoring
"""
import json
import logging
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, store_dir: Path, embedding_model: str = "all-MiniLM-L6-v2"):
        self.store_dir = store_dir
        self.embedding_model_name = embedding_model
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.embedder = None
        self._initialized = False

        self.index_path = store_dir / "faiss.index"
        self.chunks_path = store_dir / "chunks.pkl"
        self.meta_path = store_dir / "meta.json"

    def initialize(self):
        """Load or create the vector store."""
        self._load_embedder()

        if self.index_path.exists() and self.chunks_path.exists():
            self._load_from_disk()
            logger.info(f"Loaded vector store: {len(self.chunks)} chunks")
        else:
            self._create_empty_index()
            logger.info("Created new empty vector store")

        self._initialized = True

    def _load_embedder(self):
        """Load sentence transformer model."""
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()

    def _create_empty_index(self):
        """Create empty FAISS index."""
        import faiss
        # Inner product on L2-normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)

    def _load_from_disk(self):
        """Load persisted index and chunks."""
        import faiss
        self.index = faiss.read_index(str(self.index_path))
        with open(self.chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

    def _save_to_disk(self):
        """Persist index and chunks to disk."""
        import faiss
        faiss.write_index(self.index, str(self.index_path))
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        meta = {"num_chunks": len(self.chunks), "model": self.embedding_model_name}
        self.meta_path.write_text(json.dumps(meta, indent=2))

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of texts."""
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # For cosine similarity via inner product
        )
        return embeddings.astype(np.float32)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Add chunks to the index. Returns number added."""
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self.embed_texts(texts)

        start_idx = len(self.chunks)
        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            chunk["_vector_id"] = start_idx + i
            self.chunks.append(chunk)

        self._save_to_disk()
        logger.info(f"Added {len(chunks)} chunks. Total: {len(self.chunks)}")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with optional hybrid BM25 re-ranking.

        Hybrid search combines:
        - Dense (semantic) scores from FAISS
        - Sparse (keyword) scores from BM25-style TF matching
        """
        if not self.chunks:
            return []

        query_emb = self.embed_texts([query])
        k = min(top_k * 3, len(self.chunks))  # Over-retrieve for re-ranking

        scores, indices = self.index.search(query_emb, k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk["semantic_score"] = float(score)

            if doc_filter and chunk.get("doc_name") != doc_filter:
                continue

            results.append(chunk)

        if use_hybrid:
            results = self._hybrid_rerank(query, results)

        # Return top_k after reranking
        return results[:top_k]

    def _hybrid_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """BM25-style keyword scoring combined with semantic score."""
        query_terms = set(re.findall(r'\w+', query.lower()))

        for chunk in results:
            text_lower = chunk["text"].lower()
            text_terms = re.findall(r'\w+', text_lower)
            total = len(text_terms) + 1

            # TF-based keyword score
            keyword_score = sum(
                text_terms.count(term) / total
                for term in query_terms
            )

            # Weighted combination: 70% semantic, 30% keyword
            chunk["final_score"] = 0.7 * chunk["semantic_score"] + 0.3 * min(keyword_score * 10, 1.0)

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        docs = {}
        for chunk in self.chunks:
            name = chunk.get("doc_name", "unknown")
            docs[name] = docs.get(name, 0) + 1
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(docs),
            "documents": docs,
        }

    def delete_document(self, doc_name: str) -> int:
        """Remove all chunks for a document (requires index rebuild)."""
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.get("doc_name") != doc_name]
        after = len(self.chunks)

        if before != after:
            self._rebuild_index()
            self._save_to_disk()

        return before - after

    def _rebuild_index(self):
        """Rebuild FAISS index from current chunks."""
        import faiss
        self._create_empty_index()
        if self.chunks:
            texts = [c["text"] for c in self.chunks]
            embeddings = self.embed_texts(texts)
            self.index.add(embeddings)
