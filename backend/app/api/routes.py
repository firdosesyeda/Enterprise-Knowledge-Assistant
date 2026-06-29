"""
API Routes - All endpoints for the knowledge assistant.
"""
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.config import settings
from app.models.schemas import (
    EvaluationReport,
    FeedbackRequest,
    IngestResponse,
    QuestionRequest,
    QuestionResponse,
)
from app.services.document_processor import DocumentProcessor
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

router = APIRouter()

# ─── Q&A ──────────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=QuestionResponse, summary="Ask a question")
async def ask_question(request: QuestionRequest):
    """
    Main Q&A endpoint.

    - Performs hybrid semantic + keyword retrieval
    - Generates grounded answer using Claude
    - Returns sources with excerpts for citation
    """
    try:
        response = await rag_service.answer_question(
            question=request.question,
            top_k=request.top_k or 5,
            conversation_id=request.conversation_id,
        )
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in /ask")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ─── Document Ingestion ───────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, summary="Upload documents")
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Upload and index one or more documents.
    Supported formats: PDF, TXT, MD, DOCX, CSV
    """
    supported = DocumentProcessor.SUPPORTED_EXTENSIONS
    docs_info = []
    total_chunks = 0
    errors = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for upload in files:
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in supported:
                errors.append(f"{upload.filename}: unsupported format (use {supported})")
                continue

            # Save to temp dir
            tmp_path = Path(tmp_dir) / upload.filename
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)

            # Also save to persistent data dir
            dest_path = settings.data_dir / upload.filename
            shutil.copy(tmp_path, dest_path)

            try:
                doc_info = await rag_service.ingest_file(dest_path)
                docs_info.append(doc_info)
                total_chunks += doc_info.num_chunks
            except Exception as e:
                errors.append(f"{upload.filename}: {str(e)}")
                logger.error(f"Ingest failed for {upload.filename}: {e}")

    msg = f"Processed {len(docs_info)} document(s), {total_chunks} chunks created"
    if errors:
        msg += f". Errors: {'; '.join(errors)}"

    return IngestResponse(
        success=len(docs_info) > 0,
        message=msg,
        documents_processed=docs_info,
        total_chunks=total_chunks,
    )


@router.post("/ingest/directory", response_model=IngestResponse, summary="Ingest from data directory")
async def ingest_directory():
    """Ingest all documents from the configured data directory."""
    try:
        return await rag_service.ingest_directory(settings.data_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Knowledge Base Management ───────────────────────────────────────────────

@router.get("/documents", summary="List indexed documents")
async def list_documents():
    """List all indexed documents and chunk counts."""
    stats = rag_service.get_stats()
    return {
        "total_documents": stats["total_documents"],
        "total_chunks": stats["total_chunks"],
        "documents": [
            {"name": name, "chunks": count}
            for name, count in stats["documents"].items()
        ],
    }


@router.delete("/documents/{doc_name}", summary="Delete a document")
async def delete_document(doc_name: str):
    """Remove a document from the index."""
    removed = rag_service.vector_store.delete_document(doc_name)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found")

    # Also remove from data dir if present
    doc_path = settings.data_dir / doc_name
    if doc_path.exists():
        doc_path.unlink()

    return {"message": f"Removed '{doc_name}' ({removed} chunks deleted)"}


# ─── Evaluation ──────────────────────────────────────────────────────────────

@router.post("/evaluate", response_model=EvaluationReport, summary="Run evaluation")
async def evaluate(test_cases: List[dict]):
    """
    Run evaluation on a list of test cases.

    Each test case: {"question": "...", "expected_keywords": ["kw1", "kw2"]}
    """
    if not test_cases:
        raise HTTPException(status_code=400, detail="Provide at least one test case")
    return await rag_service.run_evaluation(test_cases)


# ─── Feedback ────────────────────────────────────────────────────────────────

@router.post("/feedback", summary="Submit answer feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Collect user feedback on answer quality (stored for analysis)."""
    feedback_path = settings.base_dir / "data" / "feedback.jsonl"
    import json
    with open(feedback_path, "a") as f:
        f.write(json.dumps(feedback.model_dump()) + "\n")
    return {"message": "Feedback recorded. Thank you!"}


# ─── Stats ────────────────────────────────────────────────────────────────────

@router.get("/stats", summary="Knowledge base statistics")
async def get_stats():
    """Get knowledge base statistics."""
    return rag_service.get_stats()
