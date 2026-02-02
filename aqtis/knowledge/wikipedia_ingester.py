"""
AQTIS Wikipedia Ingester.

Ingests Wikipedia articles on finance and quantitative trading topics.
"""

import logging
import time
from typing import Any, Dict, List

from .base_ingester import BaseIngester

logger = logging.getLogger(__name__)

# Curated list of relevant Wikipedia articles
DEFAULT_TOPICS = [
    # Core quantitative finance
    "Algorithmic trading",
    "Quantitative analysis (finance)",
    "Black-Scholes model",
    "Value at risk",
    "Kelly criterion",
    "Sharpe ratio",
    "Mean reversion (finance)",
    "Momentum (finance)",
    "Statistical arbitrage",
    "Pairs trade",
    "Market microstructure",
    "Order (exchange)",
    "Slippage (finance)",
    "Greeks (finance)",
    "Volatility smile",
    "Monte Carlo methods in finance",
    "GARCH",
    "Efficient-market hypothesis",
    "Capital asset pricing model",
    "Fama-French three-factor model",
    "Risk management",
    "Drawdown (economics)",
    "Portfolio optimization",
    "Modern portfolio theory",
    "Tail risk",
    # Indian markets
    "National Stock Exchange of India",
    "Bombay Stock Exchange",
    "NIFTY 50",
    "Securities and Exchange Board of India",
    "India VIX",
    "Nifty Bank",
    "National Securities Depository Limited",
    "Securities Transaction Tax",
    "Foreign portfolio investment",
    "Reserve Bank of India",
    # Broader market concepts
    "Factor investing",
    "Smart beta",
    "High-frequency trading",
    "Dark pool",
    "Behavioral finance",
    "Loss aversion",
    "Herd behavior",
]


class WikipediaIngester(BaseIngester):
    """
    Ingests Wikipedia articles on quantitative finance topics.

    Uses the wikipedia-api package to fetch article content and
    chunks it for the knowledge base.
    """

    def __init__(self, vector_store, rate_limit_seconds: float = 1.0):
        super().__init__(vector_store, source_name="wikipedia")
        self.rate_limit = rate_limit_seconds

    def ingest(self, **kwargs) -> Dict[str, Any]:
        """
        Ingest Wikipedia articles.

        kwargs:
            topics: List of article titles (default: DEFAULT_TOPICS).
        """
        topics = kwargs.get("topics", DEFAULT_TOPICS)

        try:
            import wikipediaapi
        except ImportError:
            return {
                "documents_ingested": 0,
                "chunks_created": 0,
                "errors": ["wikipedia-api required: pip install wikipedia-api"],
            }

        wiki = wikipediaapi.Wikipedia(
            user_agent="AQTIS-Research-Bot/1.0 (https://github.com/aqtis)",
            language="en",
        )

        total_chunks = 0
        errors = []
        ingested = 0

        for topic in topics:
            try:
                page = wiki.page(topic)
                if not page.exists():
                    errors.append(f"Page not found: {topic}")
                    continue

                chunks = self._ingest_article(page, topic)
                total_chunks += chunks
                ingested += 1
                time.sleep(self.rate_limit)

            except Exception as e:
                errors.append(f"{topic}: {e}")
                self.logger.error(f"Failed to ingest '{topic}': {e}")

        return {
            "documents_ingested": ingested,
            "chunks_created": total_chunks,
            "errors": errors,
        }

    def _ingest_article(self, page, topic: str) -> int:
        """Ingest a single Wikipedia article. Returns number of chunks created."""
        text = page.text
        if not text.strip():
            return 0

        # Build markdown-style text with section headers
        full_text = f"# {page.title}\n\n{page.summary}\n\n"
        for section in page.sections:
            section_text = self._extract_sections(section)
            if section_text:
                full_text += section_text

        sections = self._chunk_by_sections(full_text, max_tokens=500)

        for section in sections:
            self._store_chunk(
                text=section["text"],
                metadata={
                    "category": "encyclopedia",
                    "topic": topic,
                    "section": section["section"],
                    "url": page.fullurl,
                },
            )

        return len(sections)

    def _extract_sections(self, section, depth: int = 0) -> str:
        """Recursively extract section text."""
        if depth > 3:
            return ""

        prefix = "#" * (depth + 2)
        text = f"{prefix} {section.title}\n\n{section.text}\n\n"

        for sub in section.sections:
            text += self._extract_sections(sub, depth + 1)

        return text
