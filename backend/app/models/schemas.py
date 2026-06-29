"""
Pydantic models for request/response schemas.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="User question")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    conversation_id: Optional[str] = Field(None, description="For multi-turn conversations")


class SourceReference(BaseModel):
    document: str
    page: Optional[int] = None
    chunk_index: int
    relevance_score: float
    excerpt: str


class QuestionResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    confidence: float
    conversation_id: Optional[str] = None


class DocumentInfo(BaseModel):
    filename: str
    num_chunks: int
    file_size_kb: float
    doc_type: str


class IngestResponse(BaseModel):
    success: bool
    message: str
    documents_processed: List[DocumentInfo]
    total_chunks: int


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class EvaluationResult(BaseModel):
    question: str
    answer: str
    expected: Optional[str]
    sources_found: int
    confidence: float
    passed: bool


class EvaluationReport(BaseModel):
    total_tests: int
    passed: int
    failed: int
    avg_confidence: float
    results: List[EvaluationResult]
