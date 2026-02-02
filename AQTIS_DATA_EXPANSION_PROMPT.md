# AQTIS Data & Knowledge Expansion - Claude Code Prompt

Copy and paste this entire prompt into Claude Code to build the expanded data layer.

---

## Context

AQTIS is a multi-agent trading system for NIFTY 50 stocks. It currently has:
- 6 agents (Strategy Generator, Backtester, Risk Manager, Researcher, Post-Mortem, Prediction Tracker)
- Memory layer with SQLite + ChromaDB
- Research Agent that only ingests arXiv papers

**Goal**: Expand AQTIS with a comprehensive knowledge base + MCP server so external AI assistants (like Claude) can query it, AND integrate multiple free data sources.

---

## What to Build

### Part 1: Knowledge Base Module (`aqtis/knowledge/`)

Create a new module for ingesting and managing multiple knowledge sources:

```
aqtis/knowledge/
    __init__.py
    base_ingester.py          # Abstract base class for all ingesters
    ssrn_ingester.py          # SSRN working papers
    sec_ingester.py           # SEC EDGAR filings (10-K, 10-Q risk sections)
    fred_ingester.py          # FRED economic data documentation
    wikipedia_ingester.py     # Wikipedia finance/quant articles
    investopedia_ingester.py  # Investopedia definitions (if legal)
    pdf_ingester.py           # Local PDF ingestion for owned books
    markdown_ingester.py      # Curated markdown knowledge files
    openbb_ingester.py        # OpenBB platform documentation
    knowledge_manager.py      # Unified facade to manage all sources
```

#### Base Ingester Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import chromadb

@dataclass
class KnowledgeDocument:
    id: str
    title: str
    content: str
    source: str  # 'ssrn', 'sec', 'wikipedia', 'pdf', 'markdown', etc.
    category: str  # 'derivatives', 'risk', 'strategies', 'ml', etc.
    url: Optional[str] = None
    metadata: Optional[dict] = None

class BaseIngester(ABC):
    def __init__(self, chroma_client: chromadb.PersistentClient, collection_name: str = "knowledge_base"):
        self.client = chroma_client
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    @abstractmethod
    def ingest(self, **kwargs) -> List[KnowledgeDocument]:
        """Ingest documents from the source."""
        pass

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def store_documents(self, documents: List[KnowledgeDocument]):
        """Store documents in ChromaDB with embeddings."""
        for doc in documents:
            chunks = self.chunk_text(doc.content)
            for i, chunk in enumerate(chunks):
                self.collection.add(
                    ids=[f"{doc.id}_chunk_{i}"],
                    documents=[chunk],
                    metadatas=[{
                        "title": doc.title,
                        "source": doc.source,
                        "category": doc.category,
                        "url": doc.url or "",
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }]
                )
```

### Part 2: Data Sources Integration

#### 2.1 SSRN Ingester (`ssrn_ingester.py`)

```python
import requests
from bs4 import BeautifulSoup
import time

class SSRNIngester(BaseIngester):
    BASE_URL = "https://papers.ssrn.com/sol3/JELJOUR_Results.cfm"

    def ingest(self, query: str = "algorithmic trading", max_papers: int = 50) -> List[KnowledgeDocument]:
        """Fetch papers from SSRN based on query."""
        documents = []
        # Note: SSRN has rate limits, implement respectful scraping
        # Consider using their API if available
        # Store abstracts and key findings
        return documents
```

#### 2.2 SEC EDGAR Ingester (`sec_ingester.py`)

```python
from sec_edgar_downloader import Downloader
import os

class SECIngester(BaseIngester):
    def __init__(self, chroma_client, email: str = "your@email.com"):
        super().__init__(chroma_client)
        self.downloader = Downloader("AQTIS", email)

    def ingest(self, tickers: List[str], filing_types: List[str] = ["10-K", "10-Q"]) -> List[KnowledgeDocument]:
        """Download and extract risk factors from SEC filings."""
        documents = []
        for ticker in tickers:
            for filing_type in filing_types:
                # Download filings
                self.downloader.get(filing_type, ticker, limit=5)
                # Extract risk factors section (Item 1A)
                # Parse and chunk
        return documents

    def extract_risk_factors(self, filing_path: str) -> str:
        """Extract Item 1A (Risk Factors) from 10-K/10-Q."""
        # Parse HTML/SGML filing
        # Find Risk Factors section
        # Return cleaned text
        pass
```

#### 2.3 FRED Ingester (`fred_ingester.py`)

```python
import requests

class FREDIngester(BaseIngester):
    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, chroma_client, api_key: str = None):
        super().__init__(chroma_client)
        self.api_key = api_key or os.getenv("FRED_API_KEY")

    def ingest(self, series_ids: List[str] = None) -> List[KnowledgeDocument]:
        """Ingest FRED economic indicator documentation."""
        # Default important series for trading
        default_series = [
            "VIXCLS",      # VIX
            "DFF",         # Fed Funds Rate
            "T10Y2Y",      # Yield Curve
            "UNRATE",      # Unemployment
            "CPIAUCSL",    # CPI
            "INDPRO",      # Industrial Production
            "UMCSENT",     # Consumer Sentiment
        ]
        series_ids = series_ids or default_series
        documents = []
        for series_id in series_ids:
            # Fetch series info and documentation
            # Create knowledge document
            pass
        return documents
```

#### 2.4 Wikipedia Finance Ingester (`wikipedia_ingester.py`)

```python
import wikipediaapi

class WikipediaIngester(BaseIngester):
    FINANCE_TOPICS = [
        "Black–Scholes_model",
        "Greeks_(finance)",
        "Value_at_risk",
        "Kelly_criterion",
        "Momentum_(technical_analysis)",
        "Mean_reversion_(finance)",
        "Statistical_arbitrage",
        "Market_making",
        "Algorithmic_trading",
        "High-frequency_trading",
        "Volatility_(finance)",
        "Sharpe_ratio",
        "Maximum_drawdown",
        "GARCH",
        "Kalman_filter",
        "Hidden_Markov_model",
    ]

    def __init__(self, chroma_client):
        super().__init__(chroma_client)
        self.wiki = wikipediaapi.Wikipedia('AQTIS/1.0', 'en')

    def ingest(self, topics: List[str] = None) -> List[KnowledgeDocument]:
        """Ingest Wikipedia articles on finance topics."""
        topics = topics or self.FINANCE_TOPICS
        documents = []
        for topic in topics:
            page = self.wiki.page(topic)
            if page.exists():
                doc = KnowledgeDocument(
                    id=f"wiki_{topic}",
                    title=page.title,
                    content=page.text,
                    source="wikipedia",
                    category=self._categorize_topic(topic),
                    url=page.fullurl
                )
                documents.append(doc)
        return documents
```

#### 2.5 PDF Ingester (`pdf_ingester.py`)

```python
import fitz  # PyMuPDF

class PDFIngester(BaseIngester):
    def ingest(self, pdf_paths: List[str], category: str = "books") -> List[KnowledgeDocument]:
        """Extract text from PDF files."""
        documents = []
        for path in pdf_paths:
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()

            documents.append(KnowledgeDocument(
                id=f"pdf_{os.path.basename(path)}",
                title=os.path.basename(path).replace(".pdf", ""),
                content=text,
                source="pdf",
                category=category,
                metadata={"path": path, "pages": len(doc)}
            ))
        return documents
```

#### 2.6 Markdown Knowledge Ingester (`markdown_ingester.py`)

```python
import os
import glob

class MarkdownIngester(BaseIngester):
    def ingest(self, knowledge_dir: str = "knowledge_base") -> List[KnowledgeDocument]:
        """Ingest all markdown files from knowledge directory."""
        documents = []
        for md_path in glob.glob(f"{knowledge_dir}/**/*.md", recursive=True):
            with open(md_path, 'r') as f:
                content = f.read()

            # Extract category from directory structure
            rel_path = os.path.relpath(md_path, knowledge_dir)
            category = os.path.dirname(rel_path).replace("/", "_") or "general"

            documents.append(KnowledgeDocument(
                id=f"md_{rel_path.replace('/', '_')}",
                title=os.path.basename(md_path).replace(".md", ""),
                content=content,
                source="markdown",
                category=category,
                metadata={"path": md_path}
            ))
        return documents
```

### Part 3: Curated Knowledge Base (`knowledge_base/`)

Create comprehensive markdown files:

```
knowledge_base/
    derivatives/
        options_basics.md
        black_scholes.md
        greeks_delta_gamma.md
        greeks_vega_theta_rho.md
        implied_volatility.md
        volatility_surface.md
        exotic_options.md
        options_strategies.md

    risk_management/
        value_at_risk.md
        expected_shortfall.md
        kelly_criterion.md
        position_sizing.md
        drawdown_control.md
        correlation_risk.md
        tail_risk.md
        stress_testing.md

    trading_strategies/
        momentum.md
        mean_reversion.md
        statistical_arbitrage.md
        pairs_trading.md
        trend_following.md
        breakout_strategies.md
        market_making.md
        factor_investing.md

    market_microstructure/
        order_types.md
        bid_ask_spread.md
        market_impact.md
        slippage.md
        liquidity.md
        price_discovery.md

    machine_learning/
        feature_engineering.md
        regime_detection.md
        overfitting_prevention.md
        walk_forward_optimization.md
        ensemble_methods.md
        time_series_cv.md
        alternative_data.md

    technical_analysis/
        moving_averages.md
        rsi_macd.md
        bollinger_bands.md
        volume_analysis.md
        support_resistance.md
        candlestick_patterns.md

    quantitative_methods/
        statistical_inference.md
        hypothesis_testing.md
        regression_analysis.md
        time_series_models.md
        monte_carlo.md
        optimization.md
```

#### Example Content for `kelly_criterion.md`:

```markdown
# Kelly Criterion for Position Sizing

## Core Formula

The Kelly Criterion determines optimal bet size to maximize long-term growth:

f* = (p * b - q) / b

Where:
- f* = Optimal fraction of capital to risk
- p = Probability of winning
- q = Probability of losing (1 - p)
- b = Odds (win amount / loss amount)

For trading with equal win/loss amounts:
f* = 2p - 1

## Practical Application in AQTIS

### Quarter Kelly (Recommended)
Full Kelly is too aggressive for real trading. AQTIS uses 0.25x Kelly:

position_size = 0.25 * kelly_optimal * portfolio_value

### Calculating from Historical Trades

```python
def calculate_kelly(trades: List[Trade]) -> float:
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    win_rate = len(wins) / len(trades)
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 1

    b = avg_win / avg_loss
    kelly = (win_rate * b - (1 - win_rate)) / b
    return max(0, kelly)  # Never go negative
```

## Key Considerations

1. **Estimation Error**: Win rates and payoffs are estimated, not known. Use conservative estimates.

2. **Non-Normal Returns**: Kelly assumes returns are independent. Markets have fat tails and serial correlation.

3. **Drawdown Risk**: Full Kelly can produce 50%+ drawdowns. Most traders use 1/4 to 1/2 Kelly.

4. **Multiple Bets**: When trading multiple positions, divide Kelly by number of concurrent positions.

## When to Override Kelly

- **Regime Changes**: Reduce size in high-volatility regimes
- **Correlation Spikes**: Reduce when assets become correlated
- **Uncertainty**: When confidence is low, use smaller fraction

## References

- "Fortune's Formula" by William Poundstone
- "The Kelly Capital Growth Investment Criterion" by MacLean, Thorp, Ziemba
```

### Part 4: Knowledge Manager (`knowledge_manager.py`)

```python
from typing import List, Dict, Optional
import chromadb

class KnowledgeManager:
    """Unified interface for all knowledge sources."""

    def __init__(self, persist_directory: str = "aqtis_knowledge"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize ingesters
        self.ingesters = {
            "ssrn": SSRNIngester(self.client),
            "sec": SECIngester(self.client),
            "fred": FREDIngester(self.client),
            "wikipedia": WikipediaIngester(self.client),
            "pdf": PDFIngester(self.client),
            "markdown": MarkdownIngester(self.client),
        }

    def ingest_all(self, include_external: bool = True) -> Dict[str, int]:
        """Ingest from all sources."""
        stats = {}

        # Always ingest markdown (fast, local)
        md_docs = self.ingesters["markdown"].ingest()
        stats["markdown"] = len(md_docs)

        # Always ingest Wikipedia (free, no API key)
        wiki_docs = self.ingesters["wikipedia"].ingest()
        stats["wikipedia"] = len(wiki_docs)

        if include_external:
            # FRED (requires free API key)
            try:
                fred_docs = self.ingesters["fred"].ingest()
                stats["fred"] = len(fred_docs)
            except Exception as e:
                stats["fred"] = f"Error: {e}"

            # SEC (free, no API key)
            try:
                sec_docs = self.ingesters["sec"].ingest(
                    tickers=["AAPL", "GOOGL", "MSFT", "JPM", "GS"]
                )
                stats["sec"] = len(sec_docs)
            except Exception as e:
                stats["sec"] = f"Error: {e}"

        return stats

    def search(self, query: str, n_results: int = 10,
               category: Optional[str] = None,
               source: Optional[str] = None) -> List[Dict]:
        """Search the knowledge base."""
        where_filter = {}
        if category:
            where_filter["category"] = category
        if source:
            where_filter["source"] = source

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None
        )

        return [
            {
                "content": doc,
                "metadata": meta,
                "distance": dist
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def get_stats(self) -> Dict:
        """Get knowledge base statistics."""
        all_docs = self.collection.get()

        stats = {
            "total_documents": len(all_docs["ids"]),
            "by_source": {},
            "by_category": {}
        }

        for meta in all_docs["metadatas"]:
            source = meta.get("source", "unknown")
            category = meta.get("category", "unknown")
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

        return stats
```

### Part 5: MCP Server (`aqtis/mcp_server/`)

Create an MCP server that exposes AQTIS knowledge and data:

```
aqtis/mcp_server/
    __init__.py
    server.py           # Main MCP server
    tools.py            # Tool definitions
    resources.py        # Resource providers
```

#### MCP Server Implementation (`server.py`)

```python
#!/usr/bin/env python3
"""
AQTIS MCP Server - Exposes trading knowledge and data to AI assistants.

Run with: python -m aqtis.mcp_server.server
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource

from aqtis.knowledge.knowledge_manager import KnowledgeManager
from aqtis.memory.memory_layer import MemoryLayer
from aqtis.models.regime_detector import RegimeDetector
from aqtis.data.market_data import get_market_data

# Initialize AQTIS components
knowledge = KnowledgeManager()
memory = MemoryLayer()
regime_detector = RegimeDetector()

server = Server("aqtis-mcp")

# ============ TOOLS ============

@server.tool()
async def search_knowledge(query: str, category: str = None, limit: int = 5) -> str:
    """
    Search AQTIS knowledge base for quant finance concepts, strategies, and theory.

    Args:
        query: Natural language search query
        category: Optional filter (derivatives, risk, strategies, ml, etc.)
        limit: Maximum results to return

    Returns:
        Relevant knowledge chunks with sources
    """
    results = knowledge.search(query, n_results=limit, category=category)

    formatted = []
    for r in results:
        formatted.append(f"""
**{r['metadata'].get('title', 'Unknown')}** (Source: {r['metadata']['source']})
Category: {r['metadata'].get('category', 'general')}

{r['content'][:500]}...

---
""")

    return "\n".join(formatted) if formatted else "No relevant knowledge found."


@server.tool()
async def get_similar_trades(setup_description: str, limit: int = 10) -> str:
    """
    Find similar historical trades from AQTIS memory.

    Args:
        setup_description: Description of current trade setup
            Example: "BUY RELIANCE.NS in trending_up regime with RSI oversold"
        limit: Maximum trades to return

    Returns:
        Similar historical trades with outcomes
    """
    similar = memory.get_similar_trades(setup_description, top_k=limit)

    if not similar:
        return "No similar trades found in memory."

    formatted = []
    for trade in similar:
        formatted.append(f"""
**{trade['action']} {trade['asset']}** ({trade['timestamp']})
- Strategy: {trade['strategy_id']}
- Regime: {trade['market_regime']}
- Entry: {trade['entry_price']} → Exit: {trade['exit_price']}
- P&L: {trade['pnl_percent']:.2%}
- Similarity: {trade['similarity']:.2f}
""")

    return "\n".join(formatted)


@server.tool()
async def get_market_regime(symbol: str = "^NSEI") -> str:
    """
    Get current market regime detected by AQTIS.

    Args:
        symbol: Market index symbol (default: NIFTY 50)

    Returns:
        Current regime and confidence
    """
    try:
        data = get_market_data(symbol, period="60d")
        regime, confidence = regime_detector.detect(data)

        return f"""
**Current Market Regime**: {regime}
**Confidence**: {confidence:.1%}

Regime characteristics:
- low_volatility: Narrow ranges, low vol - favor mean reversion
- trending_up: Sustained upward momentum - favor momentum/trend following
- mean_reverting: Range-bound, oscillating - favor counter-trend
- trending_down: Sustained downward momentum - reduce exposure
- high_volatility: Large swings, elevated vol - reduce position sizes
"""
    except Exception as e:
        return f"Error detecting regime: {e}"


@server.tool()
async def get_market_data(symbol: str, period: str = "1mo") -> str:
    """
    Fetch market data for a symbol.

    Args:
        symbol: Stock/index symbol (e.g., RELIANCE.NS, ^NSEI)
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)

    Returns:
        OHLCV data and basic statistics
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)

    if data.empty:
        return f"No data found for {symbol}"

    latest = data.iloc[-1]
    returns = data['Close'].pct_change().dropna()

    return f"""
**{symbol}** - Last {period}

Latest: {latest['Close']:.2f}
High: {data['High'].max():.2f}
Low: {data['Low'].min():.2f}

**Statistics**:
- Daily Return (avg): {returns.mean():.2%}
- Volatility (annualized): {returns.std() * (252**0.5):.2%}
- Sharpe (annualized): {(returns.mean() / returns.std()) * (252**0.5):.2f}
- Max Drawdown: {((data['Close'] / data['Close'].cummax()) - 1).min():.2%}

**Latest OHLCV**:
Open: {latest['Open']:.2f}
High: {latest['High']:.2f}
Low: {latest['Low']:.2f}
Close: {latest['Close']:.2f}
Volume: {latest['Volume']:,.0f}
"""


@server.tool()
async def get_economic_indicators() -> str:
    """
    Get key economic indicators from FRED.

    Returns:
        Latest values for major economic indicators
    """
    import pandas_datareader as pdr
    from datetime import datetime, timedelta

    indicators = {
        "VIXCLS": "VIX (Volatility Index)",
        "DFF": "Fed Funds Rate",
        "T10Y2Y": "10Y-2Y Yield Spread",
        "UNRATE": "Unemployment Rate",
    }

    end = datetime.now()
    start = end - timedelta(days=30)

    results = []
    for code, name in indicators.items():
        try:
            data = pdr.get_data_fred(code, start, end)
            latest = data.iloc[-1].values[0]
            results.append(f"- {name}: {latest:.2f}")
        except:
            results.append(f"- {name}: N/A")

    return "**Economic Indicators (Latest)**\n" + "\n".join(results)


@server.tool()
async def calculate_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    portfolio_value: float,
    kelly_fraction: float = 0.25
) -> str:
    """
    Calculate optimal position size using Kelly Criterion.

    Args:
        win_rate: Historical win rate (0.0 to 1.0)
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (positive number)
        portfolio_value: Total portfolio value
        kelly_fraction: Fraction of Kelly to use (default 0.25 = quarter Kelly)

    Returns:
        Recommended position size and reasoning
    """
    if avg_loss == 0:
        return "Error: avg_loss cannot be zero"

    b = avg_win / avg_loss
    kelly = (win_rate * b - (1 - win_rate)) / b

    if kelly <= 0:
        return f"""
**Position Size: $0 (DO NOT TRADE)**

Kelly is negative ({kelly:.2%}), indicating this is not a positive expectancy trade.
Win Rate: {win_rate:.1%}
Win/Loss Ratio: {b:.2f}
Expected Value per trade: {win_rate * avg_win - (1-win_rate) * avg_loss:.2%}

Consider improving the strategy before trading.
"""

    adjusted_kelly = kelly * kelly_fraction
    position = adjusted_kelly * portfolio_value

    return f"""
**Recommended Position Size: ${position:,.2f}**

Calculations:
- Full Kelly: {kelly:.2%}
- Quarter Kelly (safer): {adjusted_kelly:.2%}
- Position: {adjusted_kelly:.2%} × ${portfolio_value:,.2f} = ${position:,.2f}

Inputs:
- Win Rate: {win_rate:.1%}
- Win/Loss Ratio: {b:.2f}
- Expected Value per trade: {win_rate * avg_win - (1-win_rate) * avg_loss:.2%}

Risk Notes:
- This is quarter Kelly to reduce drawdown risk
- Consider reducing further in high-volatility regimes
- Never exceed 10% of portfolio in a single position
"""


@server.tool()
async def get_trade_statistics(strategy_id: str = None, days: int = 90) -> str:
    """
    Get trading statistics from AQTIS memory.

    Args:
        strategy_id: Optional filter by strategy
        days: Lookback period

    Returns:
        Performance statistics
    """
    stats = memory.get_trade_stats(strategy_id=strategy_id, days=days)

    return f"""
**Trading Statistics** (Last {days} days)
{f'Strategy: {strategy_id}' if strategy_id else 'All Strategies'}

- Total Trades: {stats['total_trades']}
- Win Rate: {stats['win_rate']:.1%}
- Average Return: {stats['avg_return']:.2%}
- Sharpe Ratio: {stats['sharpe']:.2f}
- Max Drawdown: {stats['max_drawdown']:.2%}
- Profit Factor: {stats['profit_factor']:.2f}

Best Trade: {stats['best_trade']:.2%}
Worst Trade: {stats['worst_trade']:.2%}
Average Hold: {stats['avg_hold_hours']:.1f} hours
"""


@server.tool()
async def search_research(query: str, limit: int = 5) -> str:
    """
    Search AQTIS research database (arXiv papers + external sources).

    Args:
        query: Research topic to search
        limit: Maximum papers to return

    Returns:
        Relevant research papers with summaries
    """
    results = memory.search_research(query, top_k=limit)

    if not results:
        return "No relevant research found."

    formatted = []
    for paper in results:
        formatted.append(f"""
**{paper['title']}**
Authors: {paper.get('authors', 'Unknown')}
Relevance: {paper.get('relevance_score', 0):.0%}

{paper.get('summary', paper.get('abstract', 'No summary available.')[:300])}...

URL: {paper.get('url', 'N/A')}
""")

    return "\n---\n".join(formatted)


# ============ RESOURCES ============

@server.resource("aqtis://strategies")
async def list_strategies() -> str:
    """List all active trading strategies in AQTIS."""
    strategies = memory.get_strategies(active_only=True)

    formatted = ["# AQTIS Active Strategies\n"]
    for s in strategies:
        formatted.append(f"""
## {s['strategy_name']} ({s['strategy_id']})
Type: {s['strategy_type']}
Description: {s['description']}
Win Rate: {s['win_rate']:.1%}
Sharpe: {s['sharpe_ratio']:.2f}
""")

    return "\n".join(formatted)


@server.resource("aqtis://config")
async def get_config() -> str:
    """Get current AQTIS configuration."""
    from aqtis.config.settings import load_config
    config = load_config()
    return json.dumps(config, indent=2, default=str)


# ============ RUN SERVER ============

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

### Part 6: CLI Extensions

Add new CLI commands for knowledge management:

```python
# In aqtis/cli/main.py - add these commands

@cli.group()
def knowledge():
    """Knowledge base management commands."""
    pass

@knowledge.command("ingest-all")
@click.option("--include-external/--local-only", default=True)
def ingest_all(include_external: bool):
    """Ingest knowledge from all sources."""
    from aqtis.knowledge.knowledge_manager import KnowledgeManager
    km = KnowledgeManager()
    stats = km.ingest_all(include_external=include_external)
    console.print(f"[green]Ingestion complete:[/green]")
    for source, count in stats.items():
        console.print(f"  {source}: {count}")

@knowledge.command("ingest-markdown")
@click.option("--dir", default="knowledge_base", help="Knowledge directory")
def ingest_markdown(dir: str):
    """Ingest curated markdown knowledge."""
    from aqtis.knowledge.markdown_ingester import MarkdownIngester
    ingester = MarkdownIngester(get_chroma_client())
    docs = ingester.ingest(dir)
    console.print(f"[green]Ingested {len(docs)} markdown documents[/green]")

@knowledge.command("search")
@click.argument("query")
@click.option("--category", "-c", default=None)
@click.option("--limit", "-n", default=5)
def search_knowledge(query: str, category: str, limit: int):
    """Search the knowledge base."""
    from aqtis.knowledge.knowledge_manager import KnowledgeManager
    km = KnowledgeManager()
    results = km.search(query, n_results=limit, category=category)

    for r in results:
        console.print(f"\n[bold]{r['metadata']['title']}[/bold]")
        console.print(f"Source: {r['metadata']['source']} | Category: {r['metadata']['category']}")
        console.print(r['content'][:300] + "...")
        console.print("-" * 40)

@knowledge.command("stats")
def knowledge_stats():
    """Show knowledge base statistics."""
    from aqtis.knowledge.knowledge_manager import KnowledgeManager
    km = KnowledgeManager()
    stats = km.get_stats()
    console.print(json.dumps(stats, indent=2))

@cli.group()
def mcp():
    """MCP server commands."""
    pass

@mcp.command("serve")
def serve_mcp():
    """Start the AQTIS MCP server."""
    import subprocess
    subprocess.run(["python", "-m", "aqtis.mcp_server.server"])
```

### Part 7: Integration with Existing Agents

Update agents to use the knowledge base:

```python
# In strategy_generator.py - add knowledge queries

class StrategyGenerator:
    def __init__(self, ...):
        self.knowledge = KnowledgeManager()

    def analyze_signal(self, signal: MarketSignal) -> TradeDecision:
        # Existing logic...

        # NEW: Query knowledge for relevant theory
        knowledge_context = self.knowledge.search(
            f"{signal.action} {signal.asset} {self.current_regime}",
            category="trading_strategies",
            limit=3
        )

        # Add to LLM prompt
        prompt = f"""
        ...existing prompt...

        RELEVANT KNOWLEDGE:
        {self._format_knowledge(knowledge_context)}
        """


# In risk_manager.py - add Kelly knowledge

class RiskManager:
    def calculate_position_size(self, prediction, portfolio_value):
        # Existing Kelly calculation...

        # NEW: Query knowledge for regime-specific adjustments
        regime_advice = self.knowledge.search(
            f"position sizing {self.current_regime} regime",
            category="risk_management",
            limit=2
        )

        # Adjust based on knowledge recommendations
        ...


# In post_mortem.py - add theoretical context

class PostMortemAgent:
    def analyze_trade(self, trade_id: str):
        # Existing analysis...

        # NEW: Find relevant theory for this trade type
        theory = self.knowledge.search(
            f"{trade['strategy_type']} {trade['market_regime']} performance",
            limit=3
        )

        # Include in LLM prompt for deeper insights
        ...
```

### Part 8: Dependencies

Add to `requirements.txt`:

```
# Knowledge Ingestion
sec-edgar-downloader>=5.0.0
PyMuPDF>=1.23.0
beautifulsoup4>=4.12.0
wikipedia-api>=0.6.0
pandas-datareader>=0.10.0

# MCP Server
mcp>=0.1.0
```

### Part 9: Environment Variables

```bash
# Optional - for enhanced data sources
FRED_API_KEY=your_fred_api_key
SEC_EMAIL=your@email.com  # Required for SEC EDGAR
```

---

## Implementation Order

1. **Day 1**: Create `knowledge/` module structure + markdown ingester
2. **Day 1**: Create `knowledge_base/` directory with 10+ curated markdown files
3. **Day 2**: Implement Wikipedia and PDF ingesters
4. **Day 2**: Implement KnowledgeManager facade
5. **Day 3**: Add SEC EDGAR and FRED ingesters
6. **Day 3**: Create CLI commands for knowledge management
7. **Day 4**: Build MCP server with all tools
8. **Day 4**: Integrate knowledge queries into existing agents
9. **Day 5**: Write tests (15+ new tests)
10. **Day 5**: Documentation and final integration

---

## Testing

```bash
# Test knowledge ingestion
python -m pytest aqtis/tests/test_knowledge.py -v

# Test MCP server
python -m aqtis.mcp_server.server &
# Then test with MCP client

# Ingest all knowledge
python -m aqtis.cli.main knowledge ingest-all

# Search knowledge
python -m aqtis.cli.main knowledge search "Kelly criterion position sizing"
```

---

## Final Result

After implementation, AQTIS will have:

1. **Local Knowledge Base** (ChromaDB)
   - 50+ curated markdown documents on quant finance
   - Wikipedia finance articles
   - SEC filings (risk factors)
   - FRED economic indicator documentation
   - Your own PDFs (if added)

2. **MCP Server** exposing 8 tools:
   - `search_knowledge` - Query the knowledge base
   - `get_similar_trades` - Find similar historical trades
   - `get_market_regime` - Current regime detection
   - `get_market_data` - Live market data (yfinance)
   - `get_economic_indicators` - FRED data
   - `calculate_position_size` - Kelly Criterion calculator
   - `get_trade_statistics` - Performance metrics
   - `search_research` - Research paper search

3. **Smarter Agents** that reference knowledge before making decisions

This makes AQTIS both more knowledgeable internally AND accessible to external AI assistants via MCP.
