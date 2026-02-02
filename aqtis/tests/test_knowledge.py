"""
Tests for AQTIS Knowledge Ingestion System.

Covers BaseIngester, MarkdownIngester, KnowledgeManager,
VectorStore knowledge operations, and agent integration.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aqtis.memory.vector_store import VectorStore
from aqtis.knowledge.base_ingester import BaseIngester
from aqtis.knowledge.markdown_ingester import MarkdownIngester
from aqtis.knowledge.ssrn_ingester import SSRNIngester
from aqtis.knowledge.sec_ingester import SECIngester
from aqtis.knowledge.wikipedia_ingester import WikipediaIngester
from aqtis.knowledge.pdf_ingester import PDFIngester
from aqtis.knowledge.knowledge_manager import KnowledgeManager


# ─────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def vector_store(tmp_path):
    """Create a temporary VectorStore."""
    return VectorStore(persist_dir=str(tmp_path / "vectors"))


@pytest.fixture
def knowledge_base_dir(tmp_path):
    """Create a temporary knowledge base with sample markdown files."""
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()

    # Create derivatives category
    deriv_dir = kb_dir / "derivatives"
    deriv_dir.mkdir()

    (deriv_dir / "options_basics.md").write_text(
        "# Options Basics\n\n"
        "## What are Options?\n\n"
        "An option is a derivative contract that gives the holder the right, "
        "but not the obligation, to buy or sell an underlying asset at a "
        "specified price within a specified time period.\n\n"
        "## Call Options\n\n"
        "A call option gives the holder the right to buy the underlying asset. "
        "The payoff is max(S - K, 0) where S is the spot price and K is the strike.\n\n"
        "## Put Options\n\n"
        "A put option gives the holder the right to sell the underlying asset. "
        "The payoff is max(K - S, 0).\n\n"
        "## Put-Call Parity\n\n"
        "C - P = S - K * e^(-rT). This fundamental relationship must hold "
        "to prevent arbitrage opportunities.\n"
    )

    (deriv_dir / "greeks.md").write_text(
        "# The Greeks\n\n"
        "## Delta\n\n"
        "Delta measures the rate of change of the option price with respect to "
        "the change in the underlying asset price. For calls, delta ranges from 0 to 1.\n\n"
        "## Gamma\n\n"
        "Gamma is the rate of change of delta. It measures convexity of the option price.\n\n"
        "## Theta\n\n"
        "Theta measures time decay. Options lose value as expiration approaches.\n"
    )

    # Create risk_management category
    risk_dir = kb_dir / "risk_management"
    risk_dir.mkdir()

    (risk_dir / "kelly_criterion.md").write_text(
        "# Kelly Criterion\n\n"
        "## Formula\n\n"
        "The Kelly fraction is f* = (bp - q) / b, where b is the odds, "
        "p is the probability of winning, and q = 1 - p.\n\n"
        "## Fractional Kelly\n\n"
        "In practice, traders use a fraction of the full Kelly (typically 1/4 to 1/2) "
        "to reduce variance and drawdowns.\n\n"
        "## Common Pitfalls\n\n"
        "- Overestimating edge leads to ruin\n"
        "- Kelly assumes known probabilities\n"
        "- Does not account for serial correlation\n"
    )

    # Create trading_strategies category
    strat_dir = kb_dir / "trading_strategies"
    strat_dir.mkdir()

    (strat_dir / "momentum.md").write_text(
        "# Momentum Trading\n\n"
        "## Core Concept\n\n"
        "Momentum strategies buy assets that have performed well recently "
        "and sell those that have performed poorly.\n\n"
        "## Jegadeesh and Titman (1993)\n\n"
        "The seminal paper showing 3-12 month momentum generates significant alpha.\n\n"
        "## Implementation\n\n"
        "Typical lookback periods range from 1 to 12 months, with a common "
        "choice of 12-1 (12 month lookback, skip the most recent month).\n"
    )

    return kb_dir


@pytest.fixture
def knowledge_manager(vector_store, knowledge_base_dir):
    """Create a KnowledgeManager with temp directories."""
    return KnowledgeManager(
        vector_store=vector_store,
        knowledge_base_dir=str(knowledge_base_dir),
    )


# ─────────────────────────────────────────────────────────────────
# BASE INGESTER TESTS
# ─────────────────────────────────────────────────────────────────

class TestBaseIngester:
    """Tests for the BaseIngester abstract class."""

    def test_chunk_text_short(self, vector_store):
        """Short text should not be chunked."""
        class DummyIngester(BaseIngester):
            def ingest(self, **kwargs):
                return {}

        ingester = DummyIngester(vector_store, "test")
        text = "This is a short text that should not be chunked."
        chunks = ingester._chunk_text(text, max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_long(self, vector_store):
        """Long text should be split into overlapping chunks."""
        class DummyIngester(BaseIngester):
            def ingest(self, **kwargs):
                return {}

        ingester = DummyIngester(vector_store, "test")
        # Create text with 200 words
        words = [f"word{i}" for i in range(200)]
        text = " ".join(words)
        chunks = ingester._chunk_text(text, max_tokens=50, overlap=10)
        assert len(chunks) > 1
        # Verify overlap: last words of chunk N should appear in chunk N+1
        for i in range(len(chunks) - 1):
            chunk_words = chunks[i].split()
            next_chunk_words = chunks[i + 1].split()
            # There should be overlap
            overlap = set(chunk_words[-10:]) & set(next_chunk_words[:10])
            assert len(overlap) > 0

    def test_chunk_by_sections(self, vector_store):
        """Markdown should be chunked by section headers."""
        class DummyIngester(BaseIngester):
            def ingest(self, **kwargs):
                return {}

        ingester = DummyIngester(vector_store, "test")
        text = (
            "# Introduction\n\nSome intro text.\n\n"
            "## Section One\n\nContent of section one.\n\n"
            "## Section Two\n\nContent of section two.\n"
        )
        sections = ingester._chunk_by_sections(text)
        assert len(sections) == 3
        assert sections[0]["section"] == "Introduction"
        assert sections[1]["section"] == "Section One"
        assert sections[2]["section"] == "Section Two"

    def test_generate_id_deterministic(self, vector_store):
        """Same text should always produce the same ID."""
        class DummyIngester(BaseIngester):
            def ingest(self, **kwargs):
                return {}

        ingester = DummyIngester(vector_store, "test")
        id1 = ingester._generate_id("test content")
        id2 = ingester._generate_id("test content")
        id3 = ingester._generate_id("different content")
        assert id1 == id2
        assert id1 != id3


# ─────────────────────────────────────────────────────────────────
# VECTOR STORE KNOWLEDGE TESTS
# ─────────────────────────────────────────────────────────────────

class TestVectorStoreKnowledge:
    """Tests for knowledge_base collection in VectorStore."""

    def test_add_and_search_knowledge(self, vector_store):
        """Should store and retrieve knowledge documents."""
        vector_store.add_knowledge({
            "id": "test-1",
            "text": "The Kelly criterion provides optimal bet sizing for repeated bets",
            "metadata": {
                "source": "markdown",
                "category": "risk_management",
                "topic": "kelly_criterion",
            },
        })

        results = vector_store.search_knowledge("optimal position sizing betting")
        assert len(results) >= 1
        assert "kelly" in results[0]["text"].lower()

    def test_knowledge_metadata_preserved(self, vector_store):
        """Metadata should be preserved after storage."""
        vector_store.add_knowledge({
            "id": "meta-test",
            "text": "Black-Scholes model for option pricing",
            "metadata": {
                "source": "markdown",
                "category": "derivatives",
                "topic": "black_scholes",
                "section": "Formula",
            },
        })

        results = vector_store.search_knowledge("Black-Scholes option pricing")
        assert len(results) >= 1
        meta = results[0]["metadata"]
        assert meta["category"] == "derivatives"
        assert meta["source"] == "markdown"

    def test_knowledge_empty_collection(self, vector_store):
        """Should return empty list when collection has no documents."""
        results = vector_store.search_knowledge("anything")
        assert results == []

    def test_knowledge_in_stats(self, vector_store):
        """knowledge_base should appear in stats."""
        stats = vector_store.get_stats()
        assert "knowledge_base" in stats

    def test_add_multiple_and_search(self, vector_store):
        """Should rank results by relevance."""
        vector_store.add_knowledge({
            "id": "kb-1",
            "text": "Momentum strategies exploit the tendency of winners to keep winning",
            "metadata": {"source": "markdown", "category": "trading_strategies"},
        })
        vector_store.add_knowledge({
            "id": "kb-2",
            "text": "Value at Risk measures the potential loss in a portfolio",
            "metadata": {"source": "markdown", "category": "risk_management"},
        })
        vector_store.add_knowledge({
            "id": "kb-3",
            "text": "Mean reversion strategies bet on prices returning to their average",
            "metadata": {"source": "markdown", "category": "trading_strategies"},
        })

        results = vector_store.search_knowledge("momentum trading strategies", top_k=3)
        assert len(results) == 3
        # Momentum result should be most relevant
        assert "momentum" in results[0]["text"].lower()


# ─────────────────────────────────────────────────────────────────
# MARKDOWN INGESTER TESTS
# ─────────────────────────────────────────────────────────────────

class TestMarkdownIngester:
    """Tests for MarkdownIngester."""

    def test_ingest_all_files(self, vector_store, knowledge_base_dir):
        """Should ingest all markdown files from all categories."""
        ingester = MarkdownIngester(vector_store, str(knowledge_base_dir))
        result = ingester.ingest()

        assert result["documents_ingested"] == 4  # 4 markdown files
        assert result["chunks_created"] > 0
        assert result["errors"] == []

    def test_ingest_category_filter(self, vector_store, knowledge_base_dir):
        """Should filter by category when specified."""
        ingester = MarkdownIngester(vector_store, str(knowledge_base_dir))
        result = ingester.ingest(category="derivatives")

        assert result["documents_ingested"] == 2  # options_basics.md, greeks.md
        assert result["errors"] == []

    def test_ingest_nonexistent_category(self, vector_store, knowledge_base_dir):
        """Should return 0 for nonexistent category."""
        ingester = MarkdownIngester(vector_store, str(knowledge_base_dir))
        result = ingester.ingest(category="nonexistent")

        assert result["documents_ingested"] == 0
        assert result["chunks_created"] == 0

    def test_ingest_nonexistent_dir(self, vector_store):
        """Should handle missing directory gracefully."""
        ingester = MarkdownIngester(vector_store, "/tmp/nonexistent_dir_xyz")
        result = ingester.ingest()

        assert result["documents_ingested"] == 0
        assert len(result["errors"]) == 1

    def test_ingested_content_searchable(self, vector_store, knowledge_base_dir):
        """Ingested content should be searchable."""
        ingester = MarkdownIngester(vector_store, str(knowledge_base_dir))
        ingester.ingest()

        results = vector_store.search_knowledge("put-call parity")
        assert len(results) >= 1
        assert any("parity" in r["text"].lower() for r in results)


# ─────────────────────────────────────────────────────────────────
# KNOWLEDGE MANAGER TESTS
# ─────────────────────────────────────────────────────────────────

class TestKnowledgeManager:
    """Tests for KnowledgeManager facade."""

    def test_ingest_markdown(self, knowledge_manager):
        """Should ingest markdown files."""
        result = knowledge_manager.ingest_markdown()
        assert result["documents_ingested"] > 0
        assert result["chunks_created"] > 0

    def test_search_after_ingest(self, knowledge_manager):
        """Should find results after ingestion."""
        knowledge_manager.ingest_markdown()
        results = knowledge_manager.search("Kelly criterion position sizing")
        assert len(results) >= 1

    def test_search_empty_knowledge_base(self, knowledge_manager):
        """Should return empty list when nothing is ingested."""
        results = knowledge_manager.search("anything")
        assert results == []

    def test_search_with_category_filter(self, knowledge_manager):
        """Should filter by category."""
        knowledge_manager.ingest_markdown()
        results = knowledge_manager.search("options", category="derivatives")
        assert all(
            r.get("metadata", {}).get("category") == "derivatives"
            for r in results
        )

    def test_get_stats(self, knowledge_manager):
        """Should return valid statistics."""
        stats = knowledge_manager.get_stats()
        assert "total_documents" in stats
        assert stats["total_documents"] == 0

        knowledge_manager.ingest_markdown()
        stats = knowledge_manager.get_stats()
        assert stats["total_documents"] > 0

    def test_ingest_markdown_specific_category(self, knowledge_manager):
        """Should ingest only a specific category."""
        result = knowledge_manager.ingest_markdown(category="risk_management")
        assert result["documents_ingested"] == 1  # kelly_criterion.md only


# ─────────────────────────────────────────────────────────────────
# SSRN INGESTER TESTS
# ─────────────────────────────────────────────────────────────────

class TestSSRNIngester:
    """Tests for SSRNIngester (without network calls)."""

    def test_ingest_paper_chunking(self, vector_store):
        """Should chunk paper content correctly."""
        ingester = SSRNIngester(vector_store)
        paper = {
            "title": "Test Paper on Algorithmic Trading",
            "abstract": "This paper studies the effects of " + " ".join(["word"] * 600),
            "url": "https://ssrn.com/123",
            "query": "test",
        }
        chunks = ingester._ingest_paper(paper)
        assert chunks > 0


# ─────────────────────────────────────────────────────────────────
# SEC INGESTER TESTS
# ─────────────────────────────────────────────────────────────────

class TestSECIngester:
    """Tests for SECIngester (without network calls)."""

    def test_extract_risk_factors(self, vector_store):
        """Should extract risk factors from filing text."""
        ingester = SECIngester(vector_store)

        filing_text = (
            "Some preamble text. "
            "Item 1A. Risk Factors "
            "Our business faces many risks including market volatility, "
            "regulatory changes, and competitive pressures. "
            "Item 1B. Unresolved Staff Comments "
            "None."
        )
        risk_text = ingester._extract_risk_factors(filing_text)
        assert "market volatility" in risk_text
        assert "preamble" not in risk_text

    def test_ingest_requires_ticker(self, vector_store):
        """Should require ticker parameter."""
        ingester = SECIngester(vector_store)
        result = ingester.ingest()
        assert result["documents_ingested"] == 0
        assert "ticker is required" in result["errors"][0]


# ─────────────────────────────────────────────────────────────────
# PDF INGESTER TESTS
# ─────────────────────────────────────────────────────────────────

class TestPDFIngester:
    """Tests for PDFIngester (without actual PDF files)."""

    def test_ingest_requires_path(self, vector_store):
        """Should require path parameter."""
        ingester = PDFIngester(vector_store)
        result = ingester.ingest()
        assert result["documents_ingested"] == 0
        assert "path is required" in result["errors"][0]

    def test_ingest_invalid_path(self, vector_store):
        """Should handle invalid path gracefully."""
        ingester = PDFIngester(vector_store)
        result = ingester.ingest(path="/tmp/nonexistent.txt")
        assert result["documents_ingested"] == 0
        assert len(result["errors"]) == 1


# ─────────────────────────────────────────────────────────────────
# INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests for knowledge base with memory layer."""

    def test_memory_layer_knowledge_search(self, tmp_path):
        """MemoryLayer should expose knowledge search."""
        from aqtis.memory.memory_layer import MemoryLayer

        db_path = str(tmp_path / "test.db")
        vector_path = str(tmp_path / "vectors")
        memory = MemoryLayer(db_path=db_path, vector_path=vector_path)

        # Store knowledge directly through memory layer
        memory.store_knowledge({
            "id": "test-kb",
            "text": "The Sharpe ratio measures risk-adjusted return",
            "metadata": {"source": "test", "category": "risk_management"},
        })

        results = memory.search_knowledge("risk adjusted return metrics")
        assert len(results) >= 1
        assert "sharpe" in results[0]["text"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
