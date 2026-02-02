"""
AQTIS Base Ingester.

Abstract base class for all knowledge ingesters.
"""

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseIngester(ABC):
    """
    Abstract base class for knowledge ingesters.

    All ingesters share the same interface for ingesting content into
    the ChromaDB knowledge_base collection via VectorStore.
    """

    def __init__(self, vector_store, source_name: str):
        """
        Args:
            vector_store: VectorStore instance with knowledge_base collection.
            source_name: Identifier for this source (e.g., "markdown", "ssrn").
        """
        self.vector_store = vector_store
        self.source_name = source_name
        self.logger = logging.getLogger(f"aqtis.knowledge.{source_name}")

    @abstractmethod
    def ingest(self, **kwargs) -> Dict[str, Any]:
        """
        Ingest content from this source into the knowledge base.

        Returns:
            Dict with keys: documents_ingested, chunks_created, errors.
        """
        ...

    def _chunk_text(
        self,
        text: str,
        max_tokens: int = 500,
        overlap: int = 50,
    ) -> List[str]:
        """
        Split text into overlapping chunks of approximately max_tokens words.

        Args:
            text: Full text to chunk.
            max_tokens: Approximate max words per chunk.
            overlap: Number of overlapping words between chunks.
        """
        words = text.split()
        if len(words) <= max_tokens:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = start + max_tokens
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - overlap

        return chunks

    def _chunk_by_sections(self, text: str, max_tokens: int = 500) -> List[Dict]:
        """
        Chunk markdown text by section headers, falling back to token-based
        chunking for sections that exceed max_tokens.

        Returns:
            List of dicts with keys: text, section.
        """
        lines = text.split("\n")
        sections = []
        current_section = ""
        current_header = "Introduction"

        for line in lines:
            if re.match(r"^#{1,3}\s+", line):
                if current_section.strip():
                    sections.append({
                        "text": current_section.strip(),
                        "section": current_header,
                    })
                current_header = re.sub(r"^#+\s*", "", line).strip()
                current_section = line + "\n"
            else:
                current_section += line + "\n"

        if current_section.strip():
            sections.append({
                "text": current_section.strip(),
                "section": current_header,
            })

        # Sub-chunk sections that are too long
        result = []
        for section in sections:
            words = section["text"].split()
            if len(words) <= max_tokens:
                result.append(section)
            else:
                sub_chunks = self._chunk_text(section["text"], max_tokens)
                for i, chunk in enumerate(sub_chunks):
                    result.append({
                        "text": chunk,
                        "section": f"{section['section']} (part {i + 1})",
                    })

        return result

    def _store_chunk(
        self,
        text: str,
        metadata: Dict,
    ) -> str:
        """
        Store a single chunk in the knowledge_base collection.

        Args:
            text: Chunk text.
            metadata: Metadata dict (must contain str/int/float/bool values).

        Returns:
            Document ID.
        """
        doc_id = self._generate_id(text)
        metadata["source"] = self.source_name
        metadata["date_ingested"] = datetime.now().isoformat()

        return self.vector_store.add_knowledge(
            document={"id": doc_id, "text": text, "metadata": metadata}
        )

    def _generate_id(self, text: str) -> str:
        """Generate a deterministic ID from text content."""
        return hashlib.md5(text.encode()).hexdigest()
