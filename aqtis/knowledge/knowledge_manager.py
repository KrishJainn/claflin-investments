"""
AQTIS Knowledge Manager.

Unified facade for managing all knowledge sources and searching
the knowledge base.
"""

import logging
from typing import Any, Dict, List, Optional

from aqtis.memory.vector_store import VectorStore

from .markdown_ingester import MarkdownIngester
from .ssrn_ingester import SSRNIngester
from .sec_ingester import SECIngester
from .wikipedia_ingester import WikipediaIngester
from .pdf_ingester import PDFIngester
from .trading_kb_ingester import TradingKBIngester

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    Unified interface for managing the AQTIS knowledge base.

    Coordinates all ingesters and provides a single search interface
    over the knowledge_base ChromaDB collection.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_base_dir: str = "knowledge_base",
    ):
        self.vector_store = vector_store
        self.knowledge_base_dir = knowledge_base_dir

        # Initialize ingesters
        self.markdown = MarkdownIngester(vector_store, knowledge_base_dir)
        self.ssrn = SSRNIngester(vector_store)
        self.sec = SECIngester(vector_store)
        self.wikipedia = WikipediaIngester(vector_store)
        self.pdf = PDFIngester(vector_store)
        self.trading_kb = TradingKBIngester(vector_store)

    def ingest_all(self, **kwargs) -> Dict[str, Any]:
        """
        Ingest from all available sources.

        Returns:
            Summary of ingestion across all sources.
        """
        results = {}

        # Trading Knowledge Base (from 5-player coach model)
        logger.info("Ingesting 5-player trading knowledge base...")
        results["trading_kb"] = self.trading_kb.ingest()

        # Markdown (always available)
        logger.info("Ingesting markdown knowledge base...")
        results["markdown"] = self.markdown.ingest()

        # Wikipedia
        logger.info("Ingesting Wikipedia articles...")
        results["wikipedia"] = self.wikipedia.ingest()

        # SSRN (if query provided)
        ssrn_query = kwargs.get("ssrn_query", "algorithmic trading quantitative finance")
        logger.info(f"Ingesting SSRN papers for: {ssrn_query}")
        results["ssrn"] = self.ssrn.ingest(query=ssrn_query)

        # Summary
        total_docs = sum(r.get("documents_ingested", 0) for r in results.values())
        total_chunks = sum(r.get("chunks_created", 0) for r in results.values())
        all_errors = []
        for source, r in results.items():
            for err in r.get("errors", []):
                all_errors.append(f"[{source}] {err}")

        return {
            "total_documents_ingested": total_docs,
            "total_chunks_created": total_chunks,
            "total_errors": len(all_errors),
            "by_source": results,
            "errors": all_errors,
        }

    def ingest_markdown(self, category: str = None) -> Dict[str, Any]:
        """Ingest curated markdown knowledge files."""
        return self.markdown.ingest(category=category)

    def ingest_ssrn(self, query: str, max_papers: int = 20) -> Dict[str, Any]:
        """Ingest SSRN papers matching a query."""
        return self.ssrn.ingest(query=query, max_papers=max_papers)

    def ingest_sec(self, ticker: str, filing_type: str = "10-K", num_filings: int = 3) -> Dict[str, Any]:
        """Ingest SEC EDGAR filings for a ticker."""
        return self.sec.ingest(ticker=ticker, filing_type=filing_type, num_filings=num_filings)

    def ingest_wikipedia(self, topics: List[str] = None) -> Dict[str, Any]:
        """Ingest Wikipedia articles on finance topics."""
        kwargs = {}
        if topics:
            kwargs["topics"] = topics
        return self.wikipedia.ingest(**kwargs)

    def ingest_pdf(self, path: str, category: str = "textbook") -> Dict[str, Any]:
        """Ingest a PDF file or directory of PDFs."""
        return self.pdf.ingest(path=path, category=category)

    def ingest_trading_kb(self) -> Dict[str, Any]:
        """Ingest the 5-player coach model's curated trading knowledge base."""
        return self.trading_kb.ingest()

    def search(self, query: str, top_k: int = 10, category: str = None) -> List[Dict]:
        """
        Search the knowledge base.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            category: Optional category filter.

        Returns:
            List of matching documents with metadata.
        """
        results = self.vector_store.search_knowledge(query, top_k=top_k)

        if category:
            results = [
                r for r in results
                if r.get("metadata", {}).get("category") == category
            ]

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        collection = self.vector_store._get_collection("knowledge_base")
        total = collection.count()

        # Get metadata breakdown if documents exist
        stats = {
            "total_documents": total,
            "collection": "knowledge_base",
        }

        if total > 0:
            # Sample to get source distribution
            try:
                sample = collection.get(limit=min(total, 1000), include=["metadatas"])
                sources = {}
                categories = {}
                for meta in sample.get("metadatas", []):
                    if meta:
                        src = meta.get("source", "unknown")
                        sources[src] = sources.get(src, 0) + 1
                        cat = meta.get("category", "unknown")
                        categories[cat] = categories.get(cat, 0) + 1
                stats["by_source"] = sources
                stats["by_category"] = categories
            except Exception:
                pass

        return stats
