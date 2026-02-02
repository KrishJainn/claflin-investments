"""
AQTIS SSRN Ingester.

Ingests working papers from SSRN (Social Science Research Network).
"""

import logging
import time
from typing import Any, Dict, List

from .base_ingester import BaseIngester

logger = logging.getLogger(__name__)

SSRN_SEARCH_URL = "https://api.ssrn.com/content/v1/papers"
SSRN_FALLBACK_URL = "https://papers.ssrn.com/sol3/JELJOUR_Results.cfm"


class SSRNIngester(BaseIngester):
    """
    Ingests SSRN working papers via web scraping.

    Uses BeautifulSoup to parse SSRN search results and extract
    paper titles, abstracts, and metadata.
    """

    def __init__(self, vector_store, rate_limit_seconds: float = 2.0):
        super().__init__(vector_store, source_name="ssrn")
        self.rate_limit = rate_limit_seconds

    def ingest(self, **kwargs) -> Dict[str, Any]:
        """
        Ingest SSRN papers matching a search query.

        kwargs:
            query: Search query string (required).
            max_papers: Maximum papers to fetch (default 20).
        """
        query = kwargs.get("query", "algorithmic trading")
        max_papers = kwargs.get("max_papers", 20)

        papers = self._search_papers(query, max_papers)
        self.logger.info(f"Found {len(papers)} SSRN papers for '{query}'")

        total_chunks = 0
        errors = []

        for paper in papers:
            try:
                chunks = self._ingest_paper(paper)
                total_chunks += chunks
                time.sleep(self.rate_limit)
            except Exception as e:
                errors.append(f"{paper.get('title', 'Unknown')}: {e}")
                self.logger.error(f"Failed to ingest SSRN paper: {e}")

        return {
            "documents_ingested": len(papers) - len(errors),
            "chunks_created": total_chunks,
            "errors": errors,
        }

    def _search_papers(self, query: str, max_papers: int) -> List[Dict]:
        """Search SSRN for papers matching query."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            self.logger.error(
                "requests and beautifulsoup4 required: "
                "pip install requests beautifulsoup4"
            )
            return []

        papers = []
        try:
            params = {
                "keywords": query,
                "npage": 1,
            }
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; AQTIS-Research-Bot/1.0; "
                    "+https://github.com/aqtis)"
                ),
            }

            url = (
                f"https://papers.ssrn.com/sol3/results.cfm"
                f"?txtKey_Words={query.replace(' ', '+')}"
            )

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Parse search results
            results = soup.find_all("div", class_="result-item")
            if not results:
                results = soup.find_all("a", class_="title")

            for item in results[:max_papers]:
                title_tag = item.find("a", class_="title") if item.name == "div" else item
                if not title_tag:
                    continue

                title = title_tag.get_text(strip=True)
                href = title_tag.get("href", "")
                abstract_tag = item.find("div", class_="abstract-text") if item.name == "div" else None
                abstract = abstract_tag.get_text(strip=True) if abstract_tag else ""

                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "url": href,
                    "query": query,
                })

        except Exception as e:
            self.logger.error(f"SSRN search failed: {e}")

        return papers

    def _ingest_paper(self, paper: Dict) -> int:
        """Ingest a single SSRN paper. Returns number of chunks created."""
        text = f"{paper.get('title', '')}. {paper.get('abstract', '')}"
        if not text.strip(". "):
            return 0

        chunks = self._chunk_text(text, max_tokens=500)

        for i, chunk in enumerate(chunks):
            self._store_chunk(
                text=chunk,
                metadata={
                    "category": "research_paper",
                    "title": paper.get("title", "")[:200],
                    "url": paper.get("url", ""),
                    "query": paper.get("query", ""),
                    "chunk_index": i,
                },
            )

        return len(chunks)
