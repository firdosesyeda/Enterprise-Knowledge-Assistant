"""
RAG Service - Main orchestrator
Coordinates: document ingestion → vector store → retrieval → LLM generation
"""
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.core.config import settings
from app.models.schemas import (
    QuestionResponse, SourceReference, DocumentInfo,
    IngestResponse, EvaluationResult, EvaluationReport,
)
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class RAGService:
    """
    Main RAG pipeline orchestrator.

    Pipeline:
    1. INGEST: Load → Extract → Chunk → Embed → Store
    2. QUERY:  Rewrite → Retrieve (hybrid) → Re-rank → Generate → Cite
    """

    def __init__(self):
        self.processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.vector_store = VectorStore(
            store_dir=settings.vector_store_dir,
            embedding_model=settings.embedding_model,
        )
        self.llm = LLMService()
        self._ready = False

    async def initialize(self):
        """Initialize vector store (load model + index)."""
        logger.info("Initializing RAG service...")
        self.vector_store.initialize()

        # Auto-ingest any documents in data dir that aren't indexed yet
        await self._auto_ingest()
        self._ready = True
        logger.info("RAG service ready")

    async def _auto_ingest(self):
        """Ingest documents from data directory on startup."""
        data_dir = settings.data_dir
        existing_docs = set(
            c.get("doc_name") for c in self.vector_store.chunks
        )

        for file_path in data_dir.iterdir():
            if (
                file_path.suffix.lower() in DocumentProcessor.SUPPORTED_EXTENSIONS
                and file_path.name not in existing_docs
            ):
                try:
                    await self.ingest_file(file_path)
                    logger.info(f"Auto-ingested: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to auto-ingest {file_path.name}: {e}")

    async def ingest_file(self, file_path: Path) -> DocumentInfo:
        """Process and index a single document file."""
        text, metadata = self.processor.load_document(file_path)

        chunks = self.processor.chunk_text(
            text,
            doc_name=file_path.name,
            doc_type=metadata["doc_type"],
        )

        # Convert to dicts for storage
        chunk_dicts = [
            {
                "text": c.text,
                "doc_name": c.doc_name,
                "doc_type": c.doc_type,
                "page_num": c.page_num,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        self.vector_store.add_chunks(chunk_dicts)

        return DocumentInfo(
            filename=file_path.name,
            num_chunks=len(chunks),
            file_size_kb=metadata["file_size_kb"],
            doc_type=metadata["doc_type"],
        )

    async def ingest_directory(self, directory: Path) -> IngestResponse:
        """Ingest all supported documents from a directory."""
        docs_info = []
        total_chunks = 0
        errors = []

        for file_path in directory.iterdir():
            if file_path.suffix.lower() not in DocumentProcessor.SUPPORTED_EXTENSIONS:
                continue
            try:
                doc_info = await self.ingest_file(file_path)
                docs_info.append(doc_info)
                total_chunks += doc_info.num_chunks
            except Exception as e:
                errors.append(f"{file_path.name}: {str(e)}")
                logger.error(f"Failed to ingest {file_path.name}: {e}")

        msg = f"Ingested {len(docs_info)} documents, {total_chunks} chunks"
        if errors:
            msg += f". Errors: {'; '.join(errors)}"

        return IngestResponse(
            success=len(docs_info) > 0,
            message=msg,
            documents_processed=docs_info,
            total_chunks=total_chunks,
        )

    async def answer_question(
        self,
        question: str,
        top_k: int = 5,
        conversation_id: Optional[str] = None,
        rewrite_query: bool = True,
    ) -> QuestionResponse:
        """
        Full RAG pipeline for a user question.

        Steps:
        1. Query rewriting (optional) - improves retrieval
        2. Hybrid semantic + keyword retrieval
        3. Filter by similarity threshold
        4. LLM answer generation with citation
        5. Build structured response
        """
        if not self._ready:
            raise RuntimeError("RAG service not initialized")

        if not self.vector_store.chunks:
            return QuestionResponse(
                answer="No documents have been indexed yet. Please upload documents first.",
                sources=[],
                confidence=0.0,
                conversation_id=conversation_id,
            )

        # Step 1: Query rewriting
        search_query = question
        if rewrite_query and len(question.split()) <= 8:
            try:
                search_query = self.llm.rewrite_query(question)
            except Exception:
                pass

        # Step 2: Retrieve relevant chunks
        chunks = self.vector_store.search(
            query=search_query,
            top_k=top_k,
            use_hybrid=True,
        )

        # Step 3: Filter low-relevance results
        filtered_chunks = [
            c for c in chunks
            if c.get("final_score", c.get("semantic_score", 0)) >= settings.similarity_threshold
        ]

        if not filtered_chunks and chunks:
            # If all below threshold, still try with best chunk
            filtered_chunks = chunks[:1]

        # Step 4: Generate answer
        conv_id = conversation_id or str(uuid.uuid4())
        answer, confidence = self.llm.generate_answer(question, filtered_chunks, conv_id)

        # Step 5: Build source references
        sources = self._build_sources(filtered_chunks)

        return QuestionResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            conversation_id=conv_id,
        )

    def _build_sources(self, chunks: List[Dict[str, Any]]) -> List[SourceReference]:
        """Deduplicate and format source references."""
        seen = set()
        sources = []

        for chunk in chunks:
            key = (chunk.get("doc_name"), chunk.get("page_num"), chunk.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)

            excerpt = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            sources.append(
                SourceReference(
                    document=chunk.get("doc_name", "Unknown"),
                    page=chunk.get("page_num"),
                    chunk_index=chunk.get("chunk_index", 0),
                    relevance_score=round(
                        chunk.get("final_score", chunk.get("semantic_score", 0)), 3
                    ),
                    excerpt=excerpt,
                )
            )

        return sources

    async def run_evaluation(self, test_cases: List[Dict]) -> EvaluationReport:
        """
        Run evaluation on a set of test Q&A pairs.
        Measures: accuracy, source retrieval, confidence calibration.
        """
        results = []
        total_confidence = 0.0

        for tc in test_cases:
            question = tc["question"]
            expected = tc.get("expected_keywords", [])

            try:
                response = await self.answer_question(question, rewrite_query=True)
                answer_lower = response.answer.lower()

                # Check if expected keywords appear in answer
                matched = sum(
                    1 for kw in expected
                    if kw.lower() in answer_lower
                ) if expected else 1

                passed = (matched >= len(expected) * 0.5) if expected else (response.confidence > 0.3)
                total_confidence += response.confidence

                results.append(EvaluationResult(
                    question=question,
                    answer=response.answer,
                    expected=tc.get("expected_answer"),
                    sources_found=len(response.sources),
                    confidence=response.confidence,
                    passed=passed,
                ))
            except Exception as e:
                results.append(EvaluationResult(
                    question=question,
                    answer=f"ERROR: {str(e)}",
                    expected=tc.get("expected_answer"),
                    sources_found=0,
                    confidence=0.0,
                    passed=False,
                ))

        passed_count = sum(1 for r in results if r.passed)
        avg_conf = total_confidence / max(len(results), 1)

        return EvaluationReport(
            total_tests=len(results),
            passed=passed_count,
            failed=len(results) - passed_count,
            avg_confidence=round(avg_conf, 3),
            results=results,
        )

    def get_stats(self) -> Dict[str, Any]:
        return self.vector_store.get_stats()


# Singleton instance
rag_service = RAGService()
