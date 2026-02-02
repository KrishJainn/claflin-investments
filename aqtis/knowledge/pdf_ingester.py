"""
AQTIS PDF Ingester.

Ingests local PDF files (textbooks, research papers) into the knowledge base.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from .base_ingester import BaseIngester

logger = logging.getLogger(__name__)


class PDFIngester(BaseIngester):
    """
    Ingests local PDF files using PyMuPDF (fitz).

    Extracts text from each page, chunks it, and stores it
    with page number metadata for reference.
    """

    def __init__(self, vector_store):
        super().__init__(vector_store, source_name="pdf")

    def ingest(self, **kwargs) -> Dict[str, Any]:
        """
        Ingest PDF files.

        kwargs:
            path: Path to a single PDF or directory of PDFs (required).
            category: Category label for metadata (default "textbook").
        """
        path = kwargs.get("path")
        if not path:
            return {
                "documents_ingested": 0,
                "chunks_created": 0,
                "errors": ["path is required"],
            }

        category = kwargs.get("category", "textbook")
        path = Path(path)

        if path.is_file() and path.suffix.lower() == ".pdf":
            pdf_files = [path]
        elif path.is_dir():
            pdf_files = sorted(path.glob("**/*.pdf"))
        else:
            return {
                "documents_ingested": 0,
                "chunks_created": 0,
                "errors": [f"Invalid path: {path}"],
            }

        total_chunks = 0
        errors = []

        for pdf_file in pdf_files:
            try:
                chunks = self._ingest_pdf(pdf_file, category)
                total_chunks += chunks
            except Exception as e:
                errors.append(f"{pdf_file.name}: {e}")
                self.logger.error(f"Failed to ingest {pdf_file}: {e}")

        return {
            "documents_ingested": len(pdf_files) - len(errors),
            "chunks_created": total_chunks,
            "errors": errors,
        }

    def _ingest_pdf(self, pdf_path: Path, category: str) -> int:
        """Ingest a single PDF file. Returns number of chunks created."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF required for PDF ingestion: pip install PyMuPDF"
            )

        doc = fitz.open(str(pdf_path))
        total_chunks = 0

        # Extract text page by page
        full_text = ""
        page_boundaries = []  # (start_char, page_num)

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():
                page_boundaries.append((len(full_text), page_num + 1))
                full_text += page_text + "\n\n"

        doc.close()

        if not full_text.strip():
            return 0

        # Chunk the full text
        chunks = self._chunk_text(full_text, max_tokens=500)

        for i, chunk in enumerate(chunks):
            # Determine which page this chunk starts on
            chunk_start = full_text.find(chunk[:100])
            page_num = 1
            for start_char, pnum in page_boundaries:
                if start_char <= chunk_start:
                    page_num = pnum
                else:
                    break

            self._store_chunk(
                text=chunk,
                metadata={
                    "category": category,
                    "topic": pdf_path.stem,
                    "file_name": pdf_path.name,
                    "page_number": page_num,
                    "chunk_index": i,
                },
            )
            total_chunks += 1

        return total_chunks
