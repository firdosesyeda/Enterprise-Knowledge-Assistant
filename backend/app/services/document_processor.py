"""
Document Processing Service
Handles: loading, text extraction, chunking, metadata management, embedding generation
"""
import logging
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a processed chunk of a document."""
    text: str
    doc_name: str
    doc_type: str
    page_num: Optional[int]
    chunk_index: int
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.metadata.update({
            "doc_name": self.doc_name,
            "doc_type": self.doc_type,
            "page_num": self.page_num,
            "chunk_index": self.chunk_index,
        })


class DocumentProcessor:
    """
    Processes documents into chunks suitable for RAG.

    Design choices:
    - Recursive character splitter: preserves paragraph/sentence boundaries > arbitrary fixed splits
    - Overlap: prevents context loss at chunk boundaries
    - Metadata-rich chunks: enables filtered retrieval and source citation
    - Support for PDF, TXT, MD, DOCX
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv"}

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Load document and return (text, metadata)."""
        ext = file_path.suffix.lower()
        metadata = {
            "filename": file_path.name,
            "file_size_kb": round(file_path.stat().st_size / 1024, 2),
            "doc_type": self._infer_doc_type(file_path.name),
        }

        if ext == ".pdf":
            text = self._load_pdf(file_path)
        elif ext in {".txt", ".md"}:
            text = self._load_text(file_path)
        elif ext == ".docx":
            text = self._load_docx(file_path)
        elif ext == ".csv":
            text = self._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.info(f"Loaded {file_path.name}: {len(text)} chars")
        return text, metadata

    def _load_pdf(self, file_path: Path) -> str:
        """Extract text from PDF with page markers."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"[PAGE {i+1}]\n{text}")
            return "\n\n".join(pages)
        except ImportError:
            # Fallback: try pdfminer
            try:
                from pdfminer.high_level import extract_text
                return extract_text(str(file_path))
            except ImportError:
                raise ImportError("Install pypdf or pdfminer.six for PDF support")

    def _load_text(self, file_path: Path) -> str:
        """Load plain text / markdown."""
        return file_path.read_text(encoding="utf-8", errors="replace")

    def _load_docx(self, file_path: Path) -> str:
        """Load Word document."""
        try:
            import docx
            doc = docx.Document(str(file_path))
            return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except ImportError:
            raise ImportError("Install python-docx for DOCX support")

    def _load_csv(self, file_path: Path) -> str:
        """Load CSV as formatted text."""
        import csv
        lines = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lines.append(", ".join(f"{k}: {v}" for k, v in row.items()))
        return "\n".join(lines)

    def _infer_doc_type(self, filename: str) -> str:
        """Infer document type from filename. Order matters: more specific first."""
        name_lower = filename.lower()
        # Compliance checked before "guide" to avoid false matches on "compliance_guidelines"
        if any(k in name_lower for k in ["compliance", "legal", "regulation"]):
            return "Compliance"
        elif any(k in name_lower for k in ["hr", "policy", "leave", "employee"]):
            return "HR Policy"
        elif any(k in name_lower for k in ["faq", "customer", "support"]):
            return "Customer FAQ"
        elif any(k in name_lower for k in ["tech", "guide", "manual", "install"]):
            return "Technical Guide"
        elif any(k in name_lower for k in ["product", "feature", "spec"]):
            return "Product Documentation"
        elif any(k in name_lower for k in ["process", "workflow", "procedure", "onboard"]):
            return "Process Document"
        return "General Document"

    def chunk_text(self, text: str, doc_name: str, doc_type: str) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks using recursive splitting.

        Strategy:
        1. Split on double newlines (paragraph boundaries)
        2. If chunk still too large, split on single newline
        3. If still too large, split on sentence boundaries
        4. Apply overlap by including tail of previous chunk
        """
        # Extract page numbers from text
        page_chunks = self._split_by_pages(text)

        all_chunks = []
        chunk_index = 0

        for page_num, page_text in page_chunks:
            # Clean text
            clean_text = self._clean_text(page_text)
            if not clean_text.strip():
                continue

            # Recursive split
            raw_chunks = self._recursive_split(clean_text)

            for raw_chunk in raw_chunks:
                if len(raw_chunk.strip()) < 30:  # Skip tiny chunks
                    continue
                chunk = DocumentChunk(
                    text=raw_chunk.strip(),
                    doc_name=doc_name,
                    doc_type=doc_type,
                    page_num=page_num,
                    chunk_index=chunk_index,
                    char_start=0,
                    char_end=len(raw_chunk),
                )
                all_chunks.append(chunk)
                chunk_index += 1

        logger.info(f"Chunked '{doc_name}' into {len(all_chunks)} chunks")
        return all_chunks

    def _split_by_pages(self, text: str) -> List[Tuple[Optional[int], str]]:
        """Split text by page markers, return (page_num, text) tuples."""
        page_pattern = re.compile(r'\[PAGE (\d+)\]')
        parts = page_pattern.split(text)

        if len(parts) == 1:
            return [(None, text)]

        result = []
        i = 0
        if parts[0].strip():
            result.append((None, parts[0]))
            i = 1
        else:
            i = 1

        while i < len(parts) - 1:
            page_num = int(parts[i])
            page_text = parts[i + 1]
            result.append((page_num, page_text))
            i += 2

        return result if result else [(None, text)]

    def _recursive_split(self, text: str) -> List[str]:
        """Recursively split text with overlap."""
        separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        return self._split_recursive(text, separators)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Split text using first matching separator that produces valid chunks."""
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Hard split
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
            return chunks

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            parts = list(text)
        else:
            parts = text.split(sep)

        chunks = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    if len(current) > self.chunk_size:
                        chunks.extend(self._split_recursive(current, remaining_seps))
                    else:
                        chunks.append(current)
                    # Apply overlap: take tail of current chunk
                    overlap_text = current[-self.chunk_overlap:] if len(current) > self.chunk_overlap else current
                    current = overlap_text + (sep if overlap_text else "") + part
                else:
                    current = part

        if current.strip():
            if len(current) > self.chunk_size:
                chunks.extend(self._split_recursive(current, remaining_seps))
            else:
                chunks.append(current)

        return [c for c in chunks if c.strip()]

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove control chars
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text.strip()
