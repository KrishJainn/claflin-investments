"""
Phase 2 — Coach v1 Test Runner.

Tests the PostMarketAnalyzer end-to-end:
1. Creates realistic synthetic trade data (mimicking Phase 1 ledger output)
   now WITH indicator_snapshot per trade
2. Runs the analyzer (rule-based + LLM if available)
3. Validates the output JSON schema
4. Prints the full diagnosis including per-indicator scores & weight recommendations

Run:
    python test_phase2_coach.py
"""

import json
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from trading_evolution.ai_coach.post_market_analyzer import (
    PostMarketAnalyzer,
    CoachDiagnosis,
    MistakeType,
)


# Strategy weights from BEST_STRATEGY in paper_trader.py
STRATEGY_WEIGHTS = {
    "TSI_13_25": 0.8883, "NVI": 0.8611, "PVI": 0.7964, "STOCH_5_3": 0.6260,
    "ATR_14": 0.5265, "ZSCORE_20": 0.5002, "AROON_14": 0.4698, "BBANDS_20_2.5": 0.4642,
    "MASS_INDEX": 0.4535, "TEMA_20": 0.4258, "CMF_21": 0.3670, "ATR_20": 0.3443,
    "SUPERTREND_7_3": -0.9663, "AROON_25": -0.9508, "AO_5_34": -0.9286,
    "VWMA_10": -0.9283, "ADX_20": -0.8928, "EFI_13": -0.8716, "MFI_14": -0.8352,
    "VWMA_20": -0.8194, "KST": -0.7956, "WMA_20": -0.7559, "WMA_10": -0.7482,
    "DEMA_20": -0.6582, "STOCH_14_3": -0.6514, "CCI_20": -0.6433, "PIVOTS": -0.5894,
}


def create_synthetic_trades() -> list:
    """
    Create realistic synthetic trades that exercise every mistake category.
    Each trade includes indicator_snapshot matching the Phase 1 LedgerEntry format.
    """
    today = date.today()
    base = datetime(today.year, today.month, today.day, 9, 15, 0)

    trades = [
        # T0001: Winner — strong SI, good exit
        {
            "entry_id": "a1b2c3d4",
            "trade_id": "T0001",
            "symbol": "RELIANCE.NS",
            "strategy_version": "v1.0",
            "dna_id": "8748f3f8",
            "side": "BUY",
            "quantity": 40,
            "price": 2510.50,
            "si_value": 0.82,
            "entry_reason": "SI crossed above entry threshold",
            "timestamp": (base + timedelta(minutes=30)).isoformat(),
            "slippage": 2.50,
            "commission": 7.50,
            "total_cost": 10.00,
            "exit_price": 2538.00,
            "exit_time": (base + timedelta(hours=2)).isoformat(),
            "exit_reason": "SI dropped below exit threshold",
            "pnl": 1090.00,
            "indicator_snapshot": STRATEGY_WEIGHTS.copy(),
        },
        # T0002: Loser — whipsaw (in and out in 5 mins)
        {
            "entry_id": "e5f6g7h8",
            "trade_id": "T0002",
            "symbol": "TCS.NS",
            "strategy_version": "v1.0",
            "dna_id": "8748f3f8",
            "side": "BUY",
            "quantity": 15,
            "price": 3920.00,
            "si_value": 0.71,
            "entry_reason": "SI crossed above entry threshold",
            "timestamp": (base + timedelta(minutes=45)).isoformat(),
            "slippage": 3.90,
            "commission": 11.76,
            "total_cost": 15.66,
            "exit_price": 3895.00,
            "exit_time": (base + timedelta(minutes=50)).isoformat(),
            "exit_reason": "Stop loss triggered",
            "pnl": -390.66,
            "indicator_snapshot": STRATEGY_WEIGHTS.copy(),
        },
        # T0003: Winner — morning trend
        {
            "entry_id": "i9j0k1l2",
            "trade_id": "T0003",
            "symbol": "HDFCBANK.NS",
            "strategy_version": "v1.0",
            "dna_id": "8748f3f8",
            "side": "BUY",
            "quantity": 30,
            "price": 1685.00,
            "si_value": 0.88,
            "entry_reason": "SI crossed above entry threshold",
            "timestamp": (base + timedelta(hours=1)).isoformat(),
            "slippage": 1.68,
            "commission": 5.05,
            "total_cost": 6.73,
            "exit_price": 1712.50,
            "exit_time": (base + timedelta(hours=3)).isoformat(),
            "exit_reason": "SI dropped below exit threshold",
            "pnl": 818.27,
            "indicator_snapshot": STRATEGY_WEIGHTS.copy(),
        },
        # T0004: Loser — late entry, weak SI
        {
            "entry_id": "m3n4o5p6",
            "trade_id": "T0004",
            "symbol": "INFY.NS",
            "strategy_version": "v1.0",
            "dna_id": "8748f3f8",
            "side": "SELL",
            "quantity": 25,
            "price": 1520.00,
            "si_value": 0.42,
            "entry_reason": "SI crossed below entry threshold",
            "timestamp": (base + timedelta(hours=4)).isoformat(),
            "slippage": 1.52,
            "commission": 3.80,
            "total_cost": 5.32,
            "exit_price": 1535.00,
            "exit_time": (base + timedelta(hours=4, minutes=45)).isoformat(),
            "exit_reason": "Stop loss triggered",
            "pnl": -380.32,
            "indicator_snapshot": STRATEGY_WEIGHTS.copy(),
        },
        # T0005: Loser — EOD forced exit
        {
            "entry_id": "q7r8s9t0",
            "trade_id": "T0005",
            "symbol": "ICICIBANK.NS",
            "strategy_version": "v1.0",
            "dna_id": "8748f3f8",
            "side": "BUY",
            "quantity": 50,
            "price": 1150.00,
            "si_value": 0.76,
            "entry_reason": "SI crossed above entry threshold",
            "timestamp": (base + timedelta(hours=5)).isoformat(),
            "slippage": 1.15,
            "commission": 5.75,
            "total_cost": 6.90,
            "exit_price": 1145.00,
            "exit_time": (base + timedelta(hours=6)).isoformat(),
            "exit_reason": "End of day flatten",
            "pnl": -256.90,
            "indicator_snapshot": STRATEGY_WEIGHTS.copy(),
        },
        # T0006: Winner — short trade
        {
            "entry_id": "u1v2w3x4",
            "trade_id": "T0006",
            "symbol": "BHARTIARTL.NS",
            "strategy_version": "v1.0",
            "dna_id": "8748f3f8",
            "side": "SELL",
            "quantity": 20,
            "price": 1680.00,
            "si_value": -0.85,
            "entry_reason": "SI crossed below entry threshold",
            "timestamp": (base + timedelta(hours=2)).isoformat(),
            "slippage": 1.68,
            "commission": 3.36,
            "total_cost": 5.04,
            "exit_price": 1655.00,
            "exit_time": (base + timedelta(hours=3, minutes=30)).isoformat(),
            "exit_reason": "SI dropped below exit threshold",
            "pnl": 494.96,
            "indicator_snapshot": STRATEGY_WEIGHTS.copy(),
        },
        # T0007: Loser — wrong direction
        {
            "entry_id": "y5z6a7b8",
            "trade_id": "T0007",
            "symbol": "SBIN.NS",
            "strategy_version": "v1.0",
            "dna_id": "8748f3f8",
            "side": "BUY",
            "quantity": 60,
            "price": 820.00,
            "si_value": 0.73,
            "entry_reason": "SI crossed above entry threshold",
            "timestamp": (base + timedelta(hours=1, minutes=30)).isoformat(),
            "slippage": 0.82,
            "commission": 4.92,
            "total_cost": 5.74,
            "exit_price": 808.00,
            "exit_time": (base + timedelta(hours=3)).isoformat(),
            "exit_reason": "SI dropped below exit threshold",
            "pnl": -725.74,
            "indicator_snapshot": STRATEGY_WEIGHTS.copy(),
        },
        # T0008: Loser — another TCS loss (repeated symbol)
        {
            "entry_id": "c9d0e1f2",
            "trade_id": "T0008",
            "symbol": "TCS.NS",
            "strategy_version": "v1.0",
            "dna_id": "8748f3f8",
            "side": "BUY",
            "quantity": 10,
            "price": 3900.00,
            "si_value": 0.68,
            "entry_reason": "SI crossed above entry threshold",
            "timestamp": (base + timedelta(hours=2, minutes=15)).isoformat(),
            "slippage": 3.90,
            "commission": 3.90,
            "total_cost": 7.80,
            "exit_price": 3870.00,
            "exit_time": (base + timedelta(hours=3, minutes=45)).isoformat(),
            "exit_reason": "Stop loss triggered",
            "pnl": -307.80,
            "indicator_snapshot": STRATEGY_WEIGHTS.copy(),
        },
    ]

    return trades


def create_synthetic_news() -> list:
    """Create sample news items matching NewsItem.to_dict() format."""
    return [
        {
            "headline": "RBI keeps repo rate unchanged at 6.5%",
            "event": "RBI monetary policy decision",
            "impact_level": "high",
            "impact": "high",
            "affected_symbols": [],
            "symbols": [],
            "source": "Economic Times",
            "timestamp": datetime.now().isoformat(),
        },
        {
            "headline": "TCS Q3 results miss estimates, stock under pressure",
            "event": "TCS quarterly earnings miss",
            "impact_level": "high",
            "impact": "high",
            "affected_symbols": ["TCS.NS"],
            "symbols": ["TCS.NS"],
            "source": "Moneycontrol",
            "timestamp": datetime.now().isoformat(),
        },
        {
            "headline": "FII selling continues for 5th straight session",
            "event": "FII outflows continue",
            "impact_level": "medium",
            "impact": "medium",
            "affected_symbols": [],
            "symbols": [],
            "source": "LiveMint",
            "timestamp": datetime.now().isoformat(),
        },
    ]


def run_test():
    """Run Phase 2 Coach v1 end-to-end test."""
    print("=" * 80)
    print("PHASE 2 — COACH v1 TEST RUNNER (with indicator analysis)")
    print("=" * 80)

    # 1. Create synthetic data
    trades = create_synthetic_trades()
    news = create_synthetic_news()

    print(f"\nInput: {len(trades)} trades, {len(news)} news items")
    print(f"Winners: {sum(1 for t in trades if t.get('pnl', 0) > 0)}")
    print(f"Losers: {sum(1 for t in trades if t.get('pnl', 0) < 0)}")
    print(f"Total P&L: ₹{sum(t.get('pnl', 0) for t in trades):,.2f}")
    print(f"Indicators per trade: {len(trades[0].get('indicator_snapshot', {}))}")

    # 2. Run analyzer
    print("\n" + "-" * 80)
    print("Running PostMarketAnalyzer...")
    print("-" * 80)

    analyzer = PostMarketAnalyzer()
    diagnosis = analyzer.analyze(
        trades=trades,
        news_items=news,
        analysis_date=date.today(),
        strategy_version="v1.0",
    )

    # 3. Validate output
    print("\n" + "-" * 80)
    print("VALIDATION")
    print("-" * 80)

    output_dict = diagnosis.to_dict()
    is_valid, errors = CoachDiagnosis.validate_dict(output_dict)

    if is_valid:
        print("  Schema validation: PASSED")
    else:
        print("  Schema validation: FAILED")
        for err in errors:
            print(f"    - {err}")

    # Check JSON serialization round-trip
    try:
        json_str = diagnosis.to_json()
        reparsed = json.loads(json_str)
        print("  JSON round-trip: PASSED")
    except Exception as e:
        print(f"  JSON round-trip: FAILED ({e})")

    # Check indicator analysis is present
    has_scores = len(diagnosis.indicator_scores) > 0
    has_recs = len(diagnosis.weight_recommendations) > 0
    print(f"  Indicator scores: {'PASSED' if has_scores else 'MISSING'} ({len(diagnosis.indicator_scores)} indicators)")
    print(f"  Weight recommendations: {'PASSED' if has_recs else 'NONE'} ({len(diagnosis.weight_recommendations)} changes)")

    # 4. Print full diagnosis
    print("\n" + "=" * 80)
    print("FULL COACH DIAGNOSIS")
    print("=" * 80)

    print(f"\nSource: {diagnosis.meta.get('source', 'unknown')}")
    print(f"Model: {diagnosis.meta.get('llm_model', 'N/A')}")

    # Summary
    print("\n--- SUMMARY ---")
    s = diagnosis.summary
    print(f"  Trades: {s['total_trades']}  |  Win Rate: {s['win_rate']:.1f}%")
    print(f"  Net P&L: ₹{s['net_pnl']:,.2f}  |  Profit Factor: {s['profit_factor']}")
    print(f"  Best Trade: ₹{s['best_trade']:,.2f}  |  Worst: ₹{s['worst_trade']:,.2f}")

    # Winning patterns
    print(f"\n--- WINNING PATTERNS ({len(diagnosis.winning_patterns)}) ---")
    for p in diagnosis.winning_patterns:
        print(f"  [{p.get('confidence', 0):.0%}] {p['pattern']}")
        print(f"        Trades: {', '.join(p.get('trade_ids', []))}")

    # Losing patterns
    print(f"\n--- LOSING PATTERNS ({len(diagnosis.losing_patterns)}) ---")
    for p in diagnosis.losing_patterns:
        print(f"  [{p.get('confidence', 0):.0%}] {p['pattern']}")
        print(f"        Trades: {', '.join(p.get('trade_ids', []))}")

    # Mistakes
    print(f"\n--- MISTAKES ({len(diagnosis.mistakes)}) ---")
    for m in diagnosis.mistakes:
        print(f"  [{m.type}] x{m.count}  |  P&L Impact: ₹{m.pnl_impact:,.2f}")
        print(f"    {m.description}")
        print(f"    Examples: {', '.join(m.examples)}")

    # Opportunities
    print(f"\n--- OPPORTUNITIES ({len(diagnosis.opportunities)}) ---")
    for o in diagnosis.opportunities:
        print(f"  [{o.suggested_change_type}] {o.hypothesis}")
        print(f"    Mechanism: {o.expected_mechanism}")
        print(f"    Confidence: {o.confidence:.0%}")

    # News
    print(f"\n--- NEWS SUMMARY ({len(diagnosis.news_summary)}) ---")
    for n in diagnosis.news_summary:
        syms = ", ".join(n.affected_symbols) if n.affected_symbols else "market-wide"
        print(f"  [{n.impact_level}] {n.event} ({syms})")

    # Indicator diagnosis (summary)
    print(f"\n--- INDICATOR DIAGNOSIS (summary) ---")
    for k, v in diagnosis.indicator_diagnosis.items():
        print(f"  {k}: {v}")

    # Per-indicator scores
    print(f"\n--- PER-INDICATOR SCORES ({len(diagnosis.indicator_scores)}) ---")
    if diagnosis.indicator_scores:
        print(f"  {'Indicator':<20} {'Weight':>8} {'WinAvg':>8} {'LossAvg':>8} {'Corr':>8} {'Verdict':<10}")
        print(f"  {'-'*18:<20} {'-'*6:>8} {'-'*6:>8} {'-'*6:>8} {'-'*6:>8} {'-'*8:<10}")
        for sc in diagnosis.indicator_scores:
            print(
                f"  {sc.name:<20} {sc.current_weight:>8.4f} "
                f"{sc.avg_value_winners:>8.4f} {sc.avg_value_losers:>8.4f} "
                f"{sc.win_correlation:>+8.4f} {sc.verdict:<10}"
            )

    # Weight recommendations
    print(f"\n--- WEIGHT RECOMMENDATIONS ({len(diagnosis.weight_recommendations)}) ---")
    if diagnosis.weight_recommendations:
        print(f"  {'Indicator':<20} {'Current':>8} {'Recommended':>12} {'Change':>8} {'Conf':>6}")
        print(f"  {'-'*18:<20} {'-'*6:>8} {'-'*10:>12} {'-'*6:>8} {'-'*4:>6}")
        for wr in diagnosis.weight_recommendations:
            print(
                f"  {wr.indicator:<20} {wr.current_weight:>8.4f} "
                f"{wr.recommended_weight:>12.4f} {wr.change:>+8.4f} "
                f"{wr.confidence:>5.0%}"
            )
            print(f"    Reason: {wr.reason}")
    else:
        print("  No weight changes recommended (all indicators performing as expected)")

    # Regime diagnosis
    print(f"\n--- REGIME DIAGNOSIS ---")
    for k, v in diagnosis.regime_diagnosis.items():
        print(f"  {k}: {v}")

    # Save output
    output_path = Path("coach_diagnosis_output.json")
    with open(output_path, "w") as f:
        f.write(diagnosis.to_json())
    print(f"\n  Full JSON saved to: {output_path}")

    # Final verdict
    print("\n" + "=" * 80)
    if is_valid and has_scores:
        print("PHASE 2 TEST: PASSED")
        print("Coach v1 produces valid JSON with per-indicator analysis & weight recommendations.")
    elif is_valid:
        print("PHASE 2 TEST: PARTIAL")
        print("JSON valid but indicator scores missing.")
    else:
        print("PHASE 2 TEST: FAILED (see validation errors above)")
    print("=" * 80)

    return is_valid


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
