"""
AQTIS Command Line Interface.

Primary interface for interacting with the AQTIS system.
"""

import json
import logging
import sys
from pathlib import Path

import click

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from aqtis.config.settings import AQTISConfig, load_config
from aqtis.memory.memory_layer import MemoryLayer
from aqtis.llm.gemini_provider import GeminiProvider
from aqtis.llm.base import MockLLMProvider
from aqtis.orchestrator.orchestrator import MultiAgentOrchestrator
from aqtis.knowledge.knowledge_manager import KnowledgeManager


def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _get_system(config_path: str = None):
    """Initialize the full AQTIS system."""
    config = load_config(config_path)
    _setup_logging(config.system.log_level)

    memory = MemoryLayer(
        db_path=str(config.system.db_path),
        vector_path=str(config.system.vector_db_path),
    )

    # Initialize LLM
    llm = None
    if config.llm.api_key:
        try:
            llm = GeminiProvider(
                model=config.llm.model,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                api_key=config.llm.api_key,
            )
        except Exception as e:
            logging.warning(f"Failed to initialize LLM: {e}")

    orchestrator = MultiAgentOrchestrator(
        memory=memory,
        llm=llm,
        config={
            "portfolio_value": config.execution.initial_capital,
            "risk_limits": {
                "max_position_size": config.risk.max_position_size,
                "max_portfolio_leverage": config.risk.max_portfolio_leverage,
                "max_daily_loss": config.risk.max_daily_loss,
                "max_drawdown": config.risk.max_drawdown,
                "max_correlated_exposure": config.risk.max_correlated_exposure,
                "min_prediction_confidence": config.risk.min_prediction_confidence,
            },
        },
    )

    return config, memory, orchestrator


def _print_json(data, indent=2):
    """Pretty print JSON data."""
    try:
        from rich.console import Console
        from rich.json import JSON
        console = Console()
        console.print(JSON(json.dumps(data, indent=indent, default=str)))
    except ImportError:
        click.echo(json.dumps(data, indent=indent, default=str))


def _print_table(title, rows, headers):
    """Print a table."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title=title)
        for h in headers:
            table.add_column(h)
        for row in rows:
            table.add_row(*[str(v) for v in row])
        console.print(table)
    except ImportError:
        click.echo(f"\n{title}")
        click.echo("-" * 60)
        click.echo("\t".join(headers))
        for row in rows:
            click.echo("\t".join(str(v) for v in row))


# ─────────────────────────────────────────────────────────────────
# CLI GROUP
# ─────────────────────────────────────────────────────────────────

@click.group()
@click.option("--config", default=None, help="Path to config YAML")
@click.pass_context
def cli(ctx, config):
    """AQTIS - Adaptive Quantitative Trading Intelligence System"""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


# ─────────────────────────────────────────────────────────────────
# TRADE COMMANDS
# ─────────────────────────────────────────────────────────────────

@cli.group()
def trade():
    """Trade analysis and execution commands."""
    pass


@trade.command()
@click.argument("symbol")
@click.pass_context
def analyze(ctx, symbol):
    """Analyze a potential trade for SYMBOL."""
    config, memory, orchestrator = _get_system(ctx.obj.get("config_path"))

    click.echo(f"Analyzing {symbol}...")

    signal = {
        "asset": symbol,
        "action": "BUY",
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }

    # Try to get current price
    try:
        from aqtis.data.market_data import MarketDataProvider
        provider = MarketDataProvider()
        quote = provider.get_quote(symbol)
        if quote and quote.get("price"):
            signal["price"] = quote["price"]
            click.echo(f"Current price: {quote['price']}")
    except Exception:
        pass

    result = orchestrator.pre_trade_workflow(signal)
    _print_json(result)


@trade.command()
@click.argument("symbol")
@click.option("--size", default=100, help="Position size")
@click.option("--action", default="BUY", type=click.Choice(["BUY", "SELL", "SHORT", "COVER"]))
@click.pass_context
def execute(ctx, symbol, size, action):
    """Execute a trade for SYMBOL (simulation mode)."""
    config, memory, orchestrator = _get_system(ctx.obj.get("config_path"))

    signal = {
        "asset": symbol,
        "action": action,
        "position_size": size,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }

    result = orchestrator.pre_trade_workflow(signal)

    if result.get("decision") == "execute":
        # Log the trade
        trade_id = memory.store_trade({
            "asset": symbol,
            "action": action,
            "strategy_id": result.get("strategy", {}).get("strategy_id", "manual"),
            "position_size": size,
            "prediction_id": result.get("prediction_id"),
            "entry_price": signal.get("price", 0),
            "market_regime": "unknown",
        })
        click.echo(f"Trade executed: {trade_id}")
        _print_json(result.get("details", {}))
    else:
        click.echo(f"Trade not executed: {result.get('decision')}")
        click.echo(f"Reason: {result.get('reason')}")


# ─────────────────────────────────────────────────────────────────
# PORTFOLIO COMMANDS
# ─────────────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def portfolio(ctx):
    """View portfolio status."""
    config, memory, orchestrator = _get_system(ctx.obj.get("config_path"))

    stats = memory.get_stats()
    click.echo("Portfolio Status")
    click.echo("=" * 40)
    click.echo(f"Total Trades:      {stats.get('trades', 0)}")
    click.echo(f"Total Predictions: {stats.get('predictions', 0)}")
    click.echo(f"Active Strategies: {stats.get('strategies', 0)}")
    click.echo(f"Market States:     {stats.get('market_states', 0)}")
    click.echo(f"Risk Events:       {stats.get('risk_events', 0)}")

    vectors = stats.get("vector_collections", {})
    click.echo(f"\nResearch Papers:   {vectors.get('trading_research', 0)}")
    click.echo(f"Trade Patterns:    {vectors.get('trade_patterns', 0)}")


# ─────────────────────────────────────────────────────────────────
# STRATEGY COMMANDS
# ─────────────────────────────────────────────────────────────────

@cli.group()
def strategy():
    """Strategy management commands."""
    pass


@strategy.command(name="list")
@click.pass_context
def strategy_list(ctx):
    """List all strategies."""
    config, memory, orchestrator = _get_system(ctx.obj.get("config_path"))

    strategies = memory.get_active_strategies()
    if not strategies:
        click.echo("No strategies found.")
        return

    rows = []
    for s in strategies:
        rows.append([
            s.get("strategy_id", ""),
            s.get("strategy_type", ""),
            str(s.get("total_trades", 0)),
            f"{(s.get('win_rate') or 0):.2%}",
            f"{(s.get('sharpe_ratio') or 0):.2f}",
        ])

    _print_table(
        "Active Strategies",
        rows,
        ["ID", "Type", "Trades", "Win Rate", "Sharpe"],
    )


@strategy.command()
@click.argument("strategy_id")
@click.pass_context
def backtest(ctx, strategy_id):
    """Run rolling backtest for a strategy."""
    config, memory, orchestrator = _get_system(ctx.obj.get("config_path"))

    strategy_data = memory.get_strategy(strategy_id)
    if not strategy_data:
        click.echo(f"Strategy {strategy_id} not found")
        return

    result = orchestrator.agents["backtester"].run({
        "action": "rolling",
        "strategy": strategy_data,
    })
    _print_json(result)


# ─────────────────────────────────────────────────────────────────
# MEMORY COMMANDS
# ─────────────────────────────────────────────────────────────────

@cli.group()
def memory():
    """Memory database commands."""
    pass


@memory.command()
@click.argument("query")
@click.pass_context
def search(ctx, query):
    """Search memory for similar trades or research."""
    config, mem, orchestrator = _get_system(ctx.obj.get("config_path"))

    click.echo(f"Searching: {query}")

    # Search both trades and research
    trades = mem.vectors.find_similar_trades(query, top_k=5)
    research = mem.search_research(query, top_k=5)

    if trades:
        click.echo(f"\nSimilar Trades ({len(trades)}):")
        for t in trades:
            click.echo(f"  - {t.get('text', '')[:100]}")
            click.echo(f"    Distance: {t.get('distance', 'N/A')}")

    if research:
        click.echo(f"\nRelevant Research ({len(research)}):")
        for r in research:
            title = r.get("metadata", {}).get("title", r.get("text", "")[:100])
            click.echo(f"  - {title}")


@memory.command()
@click.pass_context
def stats(ctx):
    """Show memory database statistics."""
    config, mem, orchestrator = _get_system(ctx.obj.get("config_path"))
    all_stats = mem.get_stats()
    _print_json(all_stats)


# ─────────────────────────────────────────────────────────────────
# RESEARCH COMMANDS
# ─────────────────────────────────────────────────────────────────

@cli.group()
def research():
    """Research commands."""
    pass


@research.command()
@click.pass_context
def scan(ctx):
    """Run daily research scan."""
    config, memory, orchestrator = _get_system(ctx.obj.get("config_path"))

    click.echo("Running research scan...")
    result = orchestrator.agents["researcher"].run({"action": "scan"})
    click.echo(f"Papers scanned: {result.get('papers_scanned', 0)}")
    click.echo(f"Relevant papers: {result.get('relevant_papers', 0)}")

    for paper in result.get("papers", []):
        click.echo(f"  + {paper.get('title', 'Unknown')} (relevance: {paper.get('relevance', 0):.2f})")


@research.command()
@click.argument("query")
@click.pass_context
def query(ctx, query):
    """Search research for a specific topic."""
    config, memory, orchestrator = _get_system(ctx.obj.get("config_path"))

    result = orchestrator.agents["researcher"].run({
        "action": "search",
        "query": query,
    })
    _print_json(result)


# ─────────────────────────────────────────────────────────────────
# KNOWLEDGE COMMANDS
# ─────────────────────────────────────────────────────────────────

@cli.group()
def knowledge():
    """Knowledge base management commands."""
    pass


def _get_knowledge_manager(config_path=None):
    """Initialize a KnowledgeManager."""
    config = load_config(config_path)
    _setup_logging(config.system.log_level)
    from aqtis.memory.vector_store import VectorStore
    vectors = VectorStore(str(config.system.vector_db_path))
    return KnowledgeManager(vectors, knowledge_base_dir="knowledge_base")


@knowledge.command(name="ingest-all")
@click.pass_context
def knowledge_ingest_all(ctx):
    """Ingest all knowledge sources."""
    km = _get_knowledge_manager(ctx.obj.get("config_path"))

    click.echo("Ingesting all knowledge sources...")
    result = km.ingest_all()

    click.echo(f"\nDocuments ingested: {result['total_documents_ingested']}")
    click.echo(f"Chunks created:     {result['total_chunks_created']}")
    click.echo(f"Errors:             {result['total_errors']}")

    for source, data in result.get("by_source", {}).items():
        click.echo(f"\n  [{source}]")
        click.echo(f"    Documents: {data.get('documents_ingested', 0)}")
        click.echo(f"    Chunks:    {data.get('chunks_created', 0)}")
        if data.get("errors"):
            for err in data["errors"]:
                click.echo(f"    Error: {err}")


@knowledge.command(name="ingest-markdown")
@click.option("--category", default=None, help="Category subdirectory to ingest")
@click.pass_context
def knowledge_ingest_markdown(ctx, category):
    """Ingest curated markdown knowledge files."""
    km = _get_knowledge_manager(ctx.obj.get("config_path"))

    click.echo("Ingesting markdown knowledge base...")
    result = km.ingest_markdown(category=category)

    click.echo(f"Documents ingested: {result['documents_ingested']}")
    click.echo(f"Chunks created:     {result['chunks_created']}")
    if result.get("errors"):
        for err in result["errors"]:
            click.echo(f"  Error: {err}")


@knowledge.command(name="ingest-ssrn")
@click.option("--query", required=True, help="SSRN search query")
@click.option("--max-papers", default=20, help="Maximum papers to ingest")
@click.pass_context
def knowledge_ingest_ssrn(ctx, query, max_papers):
    """Ingest SSRN working papers matching a query."""
    km = _get_knowledge_manager(ctx.obj.get("config_path"))

    click.echo(f"Searching SSRN for: {query}")
    result = km.ingest_ssrn(query=query, max_papers=max_papers)

    click.echo(f"Papers ingested: {result['documents_ingested']}")
    click.echo(f"Chunks created:  {result['chunks_created']}")
    if result.get("errors"):
        for err in result["errors"]:
            click.echo(f"  Error: {err}")


@knowledge.command(name="ingest-sec")
@click.option("--ticker", required=True, help="Stock ticker symbol")
@click.option("--filing-type", default="10-K", help="SEC filing type")
@click.option("--num-filings", default=3, help="Number of filings to fetch")
@click.pass_context
def knowledge_ingest_sec(ctx, ticker, filing_type, num_filings):
    """Ingest SEC EDGAR filings for a ticker."""
    km = _get_knowledge_manager(ctx.obj.get("config_path"))

    click.echo(f"Fetching {filing_type} filings for {ticker}...")
    result = km.ingest_sec(ticker=ticker, filing_type=filing_type, num_filings=num_filings)

    click.echo(f"Filings ingested: {result['documents_ingested']}")
    click.echo(f"Chunks created:   {result['chunks_created']}")
    if result.get("errors"):
        for err in result["errors"]:
            click.echo(f"  Error: {err}")


@knowledge.command(name="search")
@click.argument("query")
@click.option("--top-k", default=10, help="Number of results")
@click.option("--category", default=None, help="Filter by category")
@click.pass_context
def knowledge_search(ctx, query, top_k, category):
    """Search the knowledge base."""
    km = _get_knowledge_manager(ctx.obj.get("config_path"))

    click.echo(f"Searching knowledge base: {query}")
    results = km.search(query, top_k=top_k, category=category)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nFound {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        category_str = meta.get("category", "")
        topic_str = meta.get("topic", "")
        section_str = meta.get("section", "")
        source_str = meta.get("source", "")
        distance = r.get("distance", "N/A")

        click.echo(f"  {i}. [{source_str}] {category_str}/{topic_str}")
        if section_str:
            click.echo(f"     Section: {section_str}")
        click.echo(f"     Distance: {distance}")
        click.echo(f"     {r.get('text', '')[:150]}...")
        click.echo()


@knowledge.command(name="stats")
@click.pass_context
def knowledge_stats(ctx):
    """Show knowledge base statistics."""
    km = _get_knowledge_manager(ctx.obj.get("config_path"))
    stats = km.get_stats()
    _print_json(stats)


# ─────────────────────────────────────────────────────────────────
# SYSTEM COMMANDS
# ─────────────────────────────────────────────────────────────────

@cli.group()
def system():
    """System management commands."""
    pass


@system.command()
@click.pass_context
def check(ctx):
    """Run system health check."""
    config, memory, orchestrator = _get_system(ctx.obj.get("config_path"))

    click.echo("AQTIS System Health Check")
    click.echo("=" * 40)

    # Config validation
    issues = config.validate()
    if issues:
        click.echo(f"\nConfig Issues ({len(issues)}):")
        for issue in issues:
            click.echo(f"  ! {issue}")
    else:
        click.echo("\nConfig: OK")

    # Memory check
    stats = memory.get_stats()
    click.echo(f"\nDatabase:")
    for key, value in stats.items():
        click.echo(f"  {key}: {value}")

    # Agent status
    health = orchestrator.get_system_health()
    click.echo(f"\nAgents:")
    for name, status in health.get("agents", {}).items():
        click.echo(f"  {name}: {status.get('status', 'unknown')}")

    click.echo(f"\nCircuit Breaker: {'ACTIVE' if health.get('circuit_breaker') else 'OFF'}")


@system.command()
@click.pass_context
def calibrate(ctx):
    """Recalibrate prediction confidence."""
    config, memory, orchestrator = _get_system(ctx.obj.get("config_path"))

    click.echo("Calibrating prediction confidence...")
    result = orchestrator.agents["prediction_tracker"].run({"action": "calibrate"})
    _print_json(result)


@system.command()
@click.pass_context
def daily(ctx):
    """Run daily maintenance routine."""
    config, memory, orchestrator = _get_system(ctx.obj.get("config_path"))

    click.echo("Running daily routine...")
    result = orchestrator.daily_routine()
    _print_json(result)


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main():
    cli(obj={})


if __name__ == "__main__":
    main()
