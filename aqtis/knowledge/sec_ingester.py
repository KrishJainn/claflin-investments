"""
AQTIS SEC EDGAR Ingester.

Ingests SEC filings (10-K risk factor sections) for financial companies.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from .base_ingester import BaseIngester

logger = logging.getLogger(__name__)


class SECIngester(BaseIngester):
    """
    Ingests SEC EDGAR filings, focusing on 10-K risk factor sections.

    Uses sec-edgar-downloader to fetch filings and extracts relevant
    risk-related sections for the knowledge base.
    """

    def __init__(self, vector_store, download_dir: str = "sec_filings", rate_limit_seconds: float = 0.5):
        super().__init__(vector_store, source_name="sec_edgar")
        self.download_dir = Path(download_dir)
        self.rate_limit = rate_limit_seconds

    def ingest(self, **kwargs) -> Dict[str, Any]:
        """
        Ingest SEC filings for a given ticker.

        kwargs:
            ticker: Stock ticker symbol (required).
            filing_type: Filing type, default "10-K".
            num_filings: Number of recent filings to fetch (default 3).
        """
        ticker = kwargs.get("ticker")
        if not ticker:
            return {
                "documents_ingested": 0,
                "chunks_created": 0,
                "errors": ["ticker is required"],
            }

        filing_type = kwargs.get("filing_type", "10-K")
        num_filings = kwargs.get("num_filings", 3)

        filings = self._download_filings(ticker, filing_type, num_filings)
        self.logger.info(f"Downloaded {len(filings)} {filing_type} filings for {ticker}")

        total_chunks = 0
        errors = []

        for filing in filings:
            try:
                chunks = self._ingest_filing(filing, ticker, filing_type)
                total_chunks += chunks
            except Exception as e:
                errors.append(f"{ticker} {filing_type}: {e}")
                self.logger.error(f"Failed to ingest filing: {e}")

        return {
            "documents_ingested": len(filings) - len(errors),
            "chunks_created": total_chunks,
            "errors": errors,
        }

    def _download_filings(self, ticker: str, filing_type: str, num_filings: int) -> List[str]:
        """Download SEC filings using sec-edgar-downloader."""
        try:
            from sec_edgar_downloader import Downloader
        except ImportError:
            self.logger.error(
                "sec-edgar-downloader required: pip install sec-edgar-downloader"
            )
            return []

        try:
            self.download_dir.mkdir(parents=True, exist_ok=True)
            dl = Downloader("AQTIS", "aqtis@research.local", str(self.download_dir))
            dl.get(filing_type, ticker, limit=num_filings)
            time.sleep(self.rate_limit)

            # Find downloaded filing text files
            filing_dir = self.download_dir / "sec-edgar-filings" / ticker / filing_type
            if not filing_dir.exists():
                return []

            filings = []
            for filing_path in sorted(filing_dir.iterdir()):
                txt_files = list(filing_path.glob("*.txt")) + list(filing_path.glob("*.htm*"))
                if txt_files:
                    filings.append(str(txt_files[0]))

            return filings[:num_filings]

        except Exception as e:
            self.logger.error(f"SEC download failed for {ticker}: {e}")
            return []

    def _ingest_filing(self, filing_path: str, ticker: str, filing_type: str) -> int:
        """Ingest a single SEC filing. Returns number of chunks created."""
        path = Path(filing_path)
        text = path.read_text(encoding="utf-8", errors="ignore")

        # Extract risk factors section
        risk_text = self._extract_risk_factors(text)
        if not risk_text:
            self.logger.debug(f"No risk factors found in {filing_path}")
            return 0

        chunks = self._chunk_text(risk_text, max_tokens=500)

        for i, chunk in enumerate(chunks):
            self._store_chunk(
                text=chunk,
                metadata={
                    "category": "sec_filing",
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "section": "risk_factors",
                    "file_path": str(path.name),
                    "chunk_index": i,
                },
            )

        return len(chunks)

    def _extract_risk_factors(self, text: str) -> str:
        """Extract the Risk Factors section from a 10-K filing."""
        # Clean HTML tags if present
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean)

        # Find "Risk Factors" or "Item 1A" section
        patterns = [
            r"(?i)(item\s+1a[\.\s]*risk\s+factors)(.*?)(item\s+1b|item\s+2[\.\s])",
            r"(?i)(risk\s+factors)(.*?)(quantitative\s+and\s+qualitative|unresolved\s+staff)",
        ]

        for pattern in patterns:
            match = re.search(pattern, clean, re.DOTALL)
            if match:
                return match.group(2).strip()[:50000]  # Cap at 50k chars

        return ""
