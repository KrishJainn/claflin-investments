"""
AQTIS Runner - End-to-end entry point.

Usage:
    python -m aqtis.run                  # Run daily routine
    python -m aqtis.run --analyze RELIANCE.NS
    python -m aqtis.run --dashboard
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aqtis.config.settings import load_config
from aqtis.memory.memory_layer import MemoryLayer
from aqtis.llm.base import MockLLMProvider
from aqtis.orchestrator.orchestrator import MultiAgentOrchestrator
from aqtis.backtest.simulation_runner import SimulationRunner


def _get_llm(config):
    """Initialize the best available LLM provider."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            from aqtis.llm.gemini_provider import GeminiProvider
            provider = GeminiProvider(api_key=api_key)
            if provider.is_available():
                return provider
        except Exception:
            pass

    logging.getLogger("aqtis").warning("No LLM API key found, using MockLLMProvider")
    return MockLLMProvider(
        default_response='{"result": "ok", "should_trade": false, "confidence": 0.5}',
    )


def build_system(config_path: str = "aqtis_config.yaml"):
    """Build the full AQTIS system and return (memory, orchestrator)."""
    config = load_config(config_path)

    # Default paths for memory storage
    data_dir = Path("aqtis_data")
    data_dir.mkdir(exist_ok=True)

    memory = MemoryLayer(
        db_path=str(data_dir / "aqtis.db"),
        vector_path=str(data_dir / "vectors"),
    )

    llm = _get_llm(config)

    orchestrator = MultiAgentOrchestrator(
        memory=memory,
        llm=llm,
        config={
            "portfolio_value": config.execution.initial_capital,
            "max_position_size": config.risk.max_position_size,
            "max_daily_loss": config.risk.max_daily_loss,
            "max_drawdown": config.risk.max_drawdown,
        },
    )

    return memory, orchestrator, config


def run_analysis(symbol: str, memory, orchestrator):
    """Run pre-trade analysis on a symbol."""
    print(f"\n--- Pre-Trade Analysis: {symbol} ---\n")

    result = orchestrator.pre_trade_workflow({
        "asset": symbol,
        "action": "BUY",
    })

    print(f"Decision: {result.get('decision', 'unknown')}")
    if result.get("strategy"):
        print(f"Strategy: {result['strategy'].get('strategy_name', 'N/A')}")
    if result.get("risk_check"):
        rc = result["risk_check"]
        print(f"Risk Approved: {rc.get('approved', 'N/A')}")
        if rc.get("position_size"):
            print(f"Position Size: {rc['position_size']:.2f}")
    if result.get("backtest"):
        bt = result["backtest"]
        print(f"Backtest Confidence: {bt.get('confidence', 0):.0%}")
        print(f"Similar Trades Win Rate: {bt.get('win_rate', 0):.0%}")

    return result


def run_daily(memory, orchestrator):
    """Run the full daily routine."""
    print("\n--- AQTIS Daily Routine ---\n")
    result = orchestrator.daily_routine()

    if result.get("research"):
        r = result["research"]
        print(f"Research: {r.get('new_papers', 0)} new papers scanned")

    if result.get("degradation"):
        d = result["degradation"]
        alerts = d.get("degradation_alerts", [])
        if alerts:
            print(f"Degradation Alerts: {len(alerts)}")
            for alert in alerts:
                print(f"  - {alert}")
        else:
            print("Degradation: No alerts")

    if result.get("weekly_review"):
        w = result["weekly_review"]
        if isinstance(w, dict) and "overall" in w:
            print(f"Weekly Review: PnL={w['overall'].get('total_pnl', 0):.2f}")

    print("\nDaily routine complete.")
    return result


def run_backtest(memory, orchestrator, config, days=60, symbols=None, llm_budget=80):
    """Run a 60-day backtest simulation."""
    print(f"\n--- AQTIS Backtest Simulation ---\n")

    runner = SimulationRunner(
        memory=memory,
        orchestrator=orchestrator,
        config=config,
        llm_budget=llm_budget,
    )

    sym_list = symbols or config.market_data.symbols
    print(f"Symbols: {len(sym_list)}")
    print(f"Days: {days}")
    print(f"LLM Budget: {llm_budget} calls")
    print(f"Initial Capital: {config.execution.initial_capital:,.0f}")
    print()

    result = runner.run(symbols=sym_list, days=days)

    if result.get("error"):
        print(f"Error: {result['error']}")
        return result

    cap = result.get("capital", {})
    metrics = result.get("metrics", {})
    llm_usage = result.get("llm_usage", {})

    print(f"\n{'='*50}")
    print(f"RUN #{result.get('run_number', '?')} COMPLETE")
    print(f"{'='*50}")
    print(f"Period: {result.get('date_range', {}).get('start')} to {result.get('date_range', {}).get('end')}")
    print(f"Symbols: {len(result.get('symbols', []))}")
    print(f"Days Simulated: {result.get('days_simulated', 0)}")
    print()
    print(f"P&L: {cap.get('pnl', 0):+,.0f} ({cap.get('return_pct', 0):+.2f}%)")
    print(f"Final Capital: {cap.get('final', 0):,.0f}")
    print()
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"Avg Win: {metrics.get('avg_win', 0):,.0f} | Avg Loss: {metrics.get('avg_loss', 0):,.0f}")
    print()
    print(f"LLM Calls: {llm_usage.get('calls_used', 0)}/{llm_usage.get('budget', 0)}")

    regime = metrics.get("regime_breakdown", {})
    if regime:
        print(f"\nRegime Breakdown:")
        for r, stats in regime.items():
            wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
            print(f"  {r}: {stats['trades']} trades, WR={wr:.0f}%, P&L={stats['pnl']:+,.0f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="AQTIS - Adaptive Quantitative Trading Intelligence System")
    parser.add_argument("--config", default="aqtis_config.yaml", help="Config file path")
    parser.add_argument("--analyze", metavar="SYMBOL", help="Run pre-trade analysis on a symbol")
    parser.add_argument("--daily", action="store_true", help="Run daily routine")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--health", action="store_true", help="Check system health")
    parser.add_argument("--backtest", action="store_true", help="Run 60-day backtest simulation")
    parser.add_argument("--days", type=int, default=60, help="Number of backtest days (default: 60)")
    parser.add_argument("--llm-budget", type=int, default=80, help="Max LLM calls for backtest (default: 80)")
    parser.add_argument("--symbols", nargs="+", help="Override symbols for backtest")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.dashboard:
        import subprocess
        dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
        return

    memory, orchestrator, config = build_system(args.config)

    if args.health:
        health = orchestrator.get_system_health()
        print("\n--- System Health ---\n")
        print(f"Agents: {len(health.get('agents', {}))} initialized")
        print(f"Memory: {health.get('memory', {}).get('trades', 0)} trades stored")
        cb = health.get("circuit_breaker", False)
        cb_active = cb.get("active") if isinstance(cb, dict) else cb
        print(f"Circuit Breaker: {'ACTIVE' if cb_active else 'OFF'}")
        issues = config.validate()
        if issues:
            print(f"\nConfig Issues ({len(issues)}):")
            for issue in issues:
                print(f"  ! {issue}")
        else:
            print("\nConfig: All checks passed")
        return

    if args.backtest:
        run_backtest(
            memory, orchestrator, config,
            days=args.days,
            symbols=args.symbols,
            llm_budget=args.llm_budget,
        )
    elif args.analyze:
        run_analysis(args.analyze, memory, orchestrator)
    elif args.daily:
        run_daily(memory, orchestrator)
    else:
        # Default: show status and run health check
        print("AQTIS v0.1.0")
        print("=" * 40)
        stats = memory.get_stats()
        print(f"Trades: {stats.get('trades', 0)}")
        print(f"Strategies: {stats.get('strategies', 0)}")
        print(f"Predictions: {stats.get('predictions', 0)}")
        print(f"\nUse --analyze SYMBOL, --daily, or --dashboard")


if __name__ == "__main__":
    main()
