"""
AQTIS Markdown Ingester.

Ingests curated markdown knowledge files from the knowledge_base/ directory.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from .base_ingester import BaseIngester

logger = logging.getLogger(__name__)


class MarkdownIngester(BaseIngester):
    """
    Ingests curated markdown files into the knowledge base.

    Chunks markdown files by section headers, preserving document structure
    and storing category/topic metadata for filtered retrieval.
    """

    def __init__(self, vector_store, knowledge_base_dir: str = "knowledge_base"):
        super().__init__(vector_store, source_name="markdown")
        self.knowledge_base_dir = Path(knowledge_base_dir)

    def ingest(self, **kwargs) -> Dict[str, Any]:
        """
        Ingest all markdown files from knowledge_base/ directory.

        kwargs:
            category: Optional category filter (e.g., "derivatives").
        """
        category_filter = kwargs.get("category")
        if not self.knowledge_base_dir.exists():
            return {
                "documents_ingested": 0,
                "chunks_created": 0,
                "errors": [f"Directory not found: {self.knowledge_base_dir}"],
            }

        md_files = self._discover_files(category_filter)
        self.logger.info(f"Found {len(md_files)} markdown files to ingest")

        total_chunks = 0
        errors = []

        for md_file in md_files:
            try:
                chunks = self._ingest_file(md_file)
                total_chunks += chunks
            except Exception as e:
                errors.append(f"{md_file}: {e}")
                self.logger.error(f"Failed to ingest {md_file}: {e}")

        return {
            "documents_ingested": len(md_files) - len(errors),
            "chunks_created": total_chunks,
            "errors": errors,
        }

    def _discover_files(self, category: str = None) -> List[Path]:
        """Find all markdown files, optionally filtered by category subdirectory."""
        if category:
            search_dir = self.knowledge_base_dir / category
            if not search_dir.exists():
                return []
            return sorted(search_dir.glob("*.md"))
        return sorted(self.knowledge_base_dir.rglob("*.md"))

    def _ingest_file(self, file_path: Path) -> int:
        """Ingest a single markdown file. Returns number of chunks created."""
        text = file_path.read_text(encoding="utf-8")
        if not text.strip():
            return 0

        # Extract category from directory structure
        relative = file_path.relative_to(self.knowledge_base_dir)
        category = relative.parent.name if relative.parent.name else "general"
        topic = file_path.stem

        # Chunk by sections
        sections = self._chunk_by_sections(text, max_tokens=500)

        for section in sections:
            self._store_chunk(
                text=section["text"],
                metadata={
                    "category": category,
                    "topic": topic,
                    "section": section["section"],
                    "file_path": str(relative),
                },
            )

        self.logger.debug(f"Ingested {file_path.name}: {len(sections)} chunks")
        return len(sections)
