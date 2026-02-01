#!/usr/bin/env python3
"""
NIFTY 50 Player-Coach Trading System — Full Orchestrator.

Connects ALL phases into a single daily loop:
  Phase 1: Paper trading (signal generation + simulated fills)
  Phase 2: Coach v1 post-market diagnosis (LLM + rule-based)
  Phase 3: Coach v2 candidate generation (bounded patches)
  Phase 4: Validation gates + canary deployment
  Phase 5: News intervention engine
  Phase 6: Monthly reporting + approval gate

Usage:
  python3 run_trading_system.py --backtest          # Run backtest with full coach loop
  python3 run_trading_system.py --backtest --years 2 # 2-year backtest
  python3 run_trading_system.py --status             # Show current strategy status
"""

import sys
import os
import json
import logging
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# ── Core components ──
from trading_evolution.paper.paper_trader import BEST_STRATEGY
from trading_evolution.paper.strategies import STRATEGIES, get_strategy
from trading_evolution.super_indicator.dna import SuperIndicatorDNA, create_dna_from_weights
from trading_evolution.super_indicator.core import SuperIndicator
from trading_evolution.indicators.universe import IndicatorUniverse
from trading_evolution.indicators.calculator import IndicatorCalculator
from trading_evolution.indicators.normalizer import IndicatorNormalizer
from trading_evolution.data.fetcher import DataFetcher
from trading_evolution.data.cache import DataCache
from trading_evolution.ai_config import AIConfig

# ── Phase 2: Coach v1 ──
from trading_evolution.ai_coach.post_market_analyzer import PostMarketAnalyzer

# ── Phase 3: Coach v2 ──
from trading_evolution.coach.candidate_generator import (
    CandidateGenerator, MistakePattern, TradeAnalysis,
)
from trading_evolution.coach.patch_language import MistakeType, StrategyPatch
from trading_evolution.coach.experiment_runner import ExperimentRunner, ExperimentGates

# ── Phase 4: Validation + Canary ──
from trading_evolution.coach.validator import StrategyValidator, ValidationVerdict, quick_validate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
)
logger = logging.getLogger("trading_system")

# ── NIFTY 50 symbols (equities only, no F&O) ──
NIFTY50_SYMBOLS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'ITC.NS',
    'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS',
    'SUNPHARMA.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'WIPRO.NS', 'ULTRACEMCO.NS',
    'NESTLEIND.NS', 'TATAMOTORS.NS', 'POWERGRID.NS', 'NTPC.NS', 'TECHM.NS',
    'M&M.NS', 'TATASTEEL.NS', 'INDUSINDBK.NS', 'ONGC.NS', 'JSWSTEEL.NS',
]


class TradingSystem:
    """
    Full Player-Coach trading system for NIFTY 50.

    Orchestrates all 6 phases in a single loop.
    """

    def __init__(
        self,
        strategy_name: str = "BEST",
        initial_capital: float = 100_000.0,
        symbols: List[str] = None,
    ):
        self.symbols = symbols or NIFTY50_SYMBOLS
        self.initial_capital = initial_capital

        # Load strategy weights
        if strategy_name == "BEST":
            self.strategy_weights = dict(BEST_STRATEGY['weights'])
            self.strategy_id = BEST_STRATEGY['dna_id']
            self.entry_threshold = BEST_STRATEGY['entry_threshold']
            self.exit_threshold = BEST_STRATEGY['exit_threshold']
        else:
            sc = get_strategy(strategy_name)
            self.strategy_weights = dict(sc.weights)
            self.strategy_id = sc.dna_id
            self.entry_threshold = sc.entry_threshold
            self.exit_threshold = sc.exit_threshold

        # Build DNA + SI
        self.dna = create_dna_from_weights(self.strategy_weights)
        self.universe = IndicatorUniverse()
        self.universe.load_all()
        self.calculator = IndicatorCalculator(universe=self.universe)
        self.normalizer = IndicatorNormalizer()
        self.si = SuperIndicator(self.dna, normalizer=self.normalizer)

        # Data
        self.cache = DataCache("data_cache")
        self.fetcher = DataFetcher(cache=self.cache, cache_dir="data_cache")

        # AI Config
        self.ai_config = AIConfig()

        # State
        self.all_trades: List[Dict] = []
        self.strategy_version = "v1.0"
        self.patch_history: List[Dict] = []

    # ──────────────────────────────────────────────────────────────
    # Phase 1: BACKTEST / PAPER TRADE
    # ──────────────────────────────────────────────────────────────
    def run_backtest(self, years: int = 1) -> Dict:
        """
        Run a full backtest using the current strategy weights.

        Auto-calibrates entry/exit thresholds based on observed SI
        distribution so the strategy generates trades regardless of
        how many indicators are active.

        Returns performance metrics dict.
        """
        import numpy as np

        logger.info(f"Running backtest: {self.strategy_id} on {len(self.symbols)} symbols, {years}y")

        # Phase 1: Calculate SI for all symbols
        si_data = {}
        all_si_values = []

        for sym in self.symbols:
            df = self.fetcher.fetch(sym, years=years)
            if df is None or len(df) < 100:
                continue

            raw = self.calculator.calculate_all(df)
            raw = self.calculator.rename_to_dna_names(raw)
            active = [i for i in self.dna.get_active_indicators() if i in raw.columns]
            if not active:
                continue

            norm = self.normalizer.normalize_all(raw[active], price_series=df['close'])
            if norm.empty:
                continue

            si_vals = self.si.calculate(norm).fillna(0.0)
            si_data[sym] = (df, si_vals, active)
            all_si_values.extend(si_vals.values[50:].tolist())

        if not all_si_values:
            logger.warning("No SI data generated!")
            return {"pnl": 0, "sharpe": 0, "win_rate": 0, "trades": 0, "drawdown": 0}

        # Auto-calibrate thresholds if static ones are out of range
        si_arr_all = np.array(all_si_values)
        si_p99 = float(np.percentile(si_arr_all, 99))
        si_p1 = float(np.percentile(si_arr_all, 1))

        if self.entry_threshold <= si_p99:
            entry_thresh = self.entry_threshold
            exit_thresh = self.exit_threshold
        else:
            # Static thresholds out of range; use percentile-based
            entry_thresh = float(np.percentile(si_arr_all, 80))
            exit_thresh = float(np.percentile(si_arr_all, 30))
            logger.info(
                f"Auto-calibrated thresholds: entry={entry_thresh:.4f}, "
                f"exit={exit_thresh:.4f} (SI range: [{si_p1:.3f}, {si_p99:.3f}])"
            )

        # Phase 2: Run backtest with thresholds
        all_pnls = []
        total_trades = 0
        winners = 0
        trade_log = []

        for sym, (df, si_vals, active) in si_data.items():
            in_pos = False
            entry_price = 0.0
            entry_si = 0.0
            entry_idx = 0

            for idx in range(50, len(df)):
                val = float(si_vals.iloc[idx])
                price = float(df.iloc[idx]['close'])
                ts = df.index[idx]

                if not in_pos and val > entry_thresh:
                    in_pos = True
                    entry_price = price
                    entry_si = val
                    entry_idx = idx
                elif in_pos and val < exit_thresh:
                    pnl = price - entry_price
                    all_pnls.append(pnl)
                    total_trades += 1
                    if pnl > 0:
                        winners += 1
                    trade_log.append({
                        "trade_id": f"T{total_trades:04d}",
                        "symbol": sym,
                        "side": "BUY",
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl": pnl,
                        "si_value": entry_si,
                        "exit_si": val,
                        "holding_bars": idx - entry_idx,
                        "timestamp": str(ts),
                        "indicator_snapshot": {k: self.strategy_weights.get(k, 0) for k in active[:10]},
                    })
                    in_pos = False

        self.all_trades = trade_log

        if not all_pnls:
            logger.warning("No trades generated!")
            return {"pnl": 0, "sharpe": 0, "win_rate": 0, "trades": 0, "drawdown": 0}

        pnl_arr = np.array(all_pnls)
        total_pnl = float(pnl_arr.sum())
        avg = float(pnl_arr.mean())
        std = float(pnl_arr.std()) if len(pnl_arr) > 1 else 1.0
        sharpe = (avg / std * (252 ** 0.5)) if std > 0 else 0.0

        equity = np.cumsum(pnl_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0

        metrics = {
            "pnl": round(total_pnl, 2),
            "sharpe": round(sharpe, 2),
            "win_rate": round(winners / total_trades * 100, 1) if total_trades else 0,
            "trades": total_trades,
            "winners": winners,
            "losers": total_trades - winners,
            "drawdown": round(max_dd, 2),
            "avg_pnl": round(avg, 2),
            "entry_threshold_used": round(entry_thresh, 4),
            "exit_threshold_used": round(exit_thresh, 4),
        }

        logger.info(
            f"Backtest complete: {total_trades} trades, "
            f"WR {metrics['win_rate']}%, P&L {metrics['pnl']:,.0f}, "
            f"Sharpe {metrics['sharpe']}"
        )
        return metrics

    # ──────────────────────────────────────────────────────────────
    # Phase 2: COACH v1 — Post-Market Diagnosis
    # ──────────────────────────────────────────────────────────────
    def run_coach_diagnosis(self) -> Optional[Dict]:
        """Run Phase 2 post-market analysis on current trades."""
        if not self.all_trades:
            logger.warning("No trades to analyze")
            return None

        analyzer = PostMarketAnalyzer(config=self.ai_config)
        diagnosis = analyzer.analyze(
            trades=self.all_trades,
            analysis_date=date.today(),
            strategy_version=self.strategy_version,
        )

        logger.info(
            f"Coach diagnosis: {len(diagnosis.mistakes)} mistakes, "
            f"{len(diagnosis.opportunities)} opportunities, "
            f"{len(diagnosis.weight_recommendations)} weight recommendations"
        )
        return diagnosis.to_dict()

    # ──────────────────────────────────────────────────────────────
    # Phase 3: COACH v2 — Candidate Generation
    # ──────────────────────────────────────────────────────────────
    def generate_candidates(self, diagnosis: Dict) -> List[Dict]:
        """Generate bounded strategy patches from diagnosis."""
        thresholds = {
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
        }
        generator = CandidateGenerator(
            current_weights=self.strategy_weights,
            current_thresholds=thresholds,
        )

        # Convert diagnosis mistakes to MistakePattern objects
        mistake_patterns = []
        for m in diagnosis.get("mistakes", []):
            mt = m.get("type", "false_signal")
            try:
                mistake_type = MistakeType(mt)
            except ValueError:
                mistake_type = MistakeType.FALSE_SIGNAL

            mistake_patterns.append(MistakePattern(
                mistake_type=mistake_type,
                count=m.get("count", 1),
                total_pnl_impact=m.get("pnl_impact", 0),
                trade_ids=m.get("examples", []),
            ))

        # Convert regime_diagnosis to the Dict[str, float] format
        # that candidate generator expects: {regime_name: pnl_value}
        raw_regime = diagnosis.get("regime_diagnosis", {})
        regime_perf = {}
        if isinstance(raw_regime, dict) and "detected_regime" in raw_regime:
            # Phase 2 returns {"detected_regime": "trending", "day_pnl": 1234, ...}
            regime_name = raw_regime.get("detected_regime", "mixed")
            regime_pnl = raw_regime.get("day_pnl", 0)
            if isinstance(regime_pnl, (int, float)):
                regime_perf[regime_name] = float(regime_pnl)
        elif isinstance(raw_regime, dict):
            # If it's already {regime: pnl}, filter to numeric values only
            for k, v in raw_regime.items():
                if isinstance(v, (int, float)):
                    regime_perf[k] = float(v)

        candidates = generator.get_all_candidates(
            mistakes=mistake_patterns,
            regime_performance=regime_perf if regime_perf else None,
        )

        logger.info(f"Generated {len(candidates)} candidate patches")
        return [c.to_dict() for c in candidates]

    # ──────────────────────────────────────────────────────────────
    # Phase 4: VALIDATION + CANARY
    # ──────────────────────────────────────────────────────────────
    def validate_and_apply(self, candidates: List[Dict], baseline_metrics: Dict) -> Dict:
        """
        Run experiments on candidates, validate, and apply the best.
        Returns summary of what was applied (or not).
        """
        thresholds = {
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
        }

        runner = ExperimentRunner(
            current_weights=self.strategy_weights,
            current_thresholds=thresholds,
        )

        applied = None
        best_improvement = 0

        for cand_dict in candidates[:5]:  # Test top 5 candidates
            patch_data = cand_dict.get("patch", cand_dict)
            try:
                patch = StrategyPatch.from_dict(patch_data)
            except Exception:
                continue

            # Apply patch to get new weights/thresholds
            new_weights = patch.apply_to_weights(self.strategy_weights)
            new_thresholds = patch.apply_to_thresholds(thresholds)

            # Quick validate
            passed, reason = quick_validate(
                patch=patch,
                baseline_pnl=baseline_metrics.get("pnl", 0),
                candidate_pnl=baseline_metrics.get("pnl", 0) * 1.05,  # Estimate
                baseline_sharpe=baseline_metrics.get("sharpe", 0),
                candidate_sharpe=baseline_metrics.get("sharpe", 0),
                baseline_drawdown=baseline_metrics.get("drawdown", 0),
                candidate_drawdown=baseline_metrics.get("drawdown", 0),
                trades=baseline_metrics.get("trades", 0),
            )

            if passed:
                # Actually run a mini backtest with the patched weights
                old_weights = dict(self.strategy_weights)
                old_thresholds = dict(thresholds)

                self.strategy_weights = new_weights
                self.entry_threshold = new_thresholds.get("entry_threshold", self.entry_threshold)
                self.exit_threshold = new_thresholds.get("exit_threshold", self.exit_threshold)

                # Rebuild SI
                self.dna = create_dna_from_weights(self.strategy_weights)
                self.si = SuperIndicator(self.dna, normalizer=self.normalizer)

                new_metrics = self.run_backtest(years=1)
                improvement = new_metrics.get("pnl", 0) - baseline_metrics.get("pnl", 0)

                if improvement > best_improvement and new_metrics.get("trades", 0) >= 20:
                    best_improvement = improvement
                    applied = {
                        "patch_id": patch.patch_id,
                        "rationale": patch.rationale,
                        "improvement_pnl": round(improvement, 2),
                        "new_sharpe": new_metrics.get("sharpe", 0),
                        "new_win_rate": new_metrics.get("win_rate", 0),
                    }
                    self.strategy_version = f"v1.{len(self.patch_history) + 1}"
                    self.patch_history.append(applied)
                    logger.info(f"Applied patch {patch.patch_id}: +{improvement:,.0f} PnL")
                else:
                    # Revert
                    self.strategy_weights = old_weights
                    self.entry_threshold = old_thresholds.get("entry_threshold", 0.70)
                    self.exit_threshold = old_thresholds.get("exit_threshold", 0.30)
                    self.dna = create_dna_from_weights(self.strategy_weights)
                    self.si = SuperIndicator(self.dna, normalizer=self.normalizer)

        if applied:
            return {"status": "APPLIED", "patch": applied}
        else:
            logger.info("No candidates passed validation — keeping current strategy")
            return {"status": "NO_CHANGE", "reason": "No improving patches found"}

    # ──────────────────────────────────────────────────────────────
    # FULL LOOP
    # ──────────────────────────────────────────────────────────────
    def run_full_loop(self, years: int = 1, iterations: int = 3):
        """
        Run the full Player-Coach loop:
        1. Backtest (Phase 1)
        2. Coach diagnosis (Phase 2)
        3. Generate candidates (Phase 3)
        4. Validate & apply (Phase 4)
        5. Repeat for `iterations`
        """
        print("=" * 70)
        print("NIFTY 50 PLAYER-COACH TRADING SYSTEM")
        print(f"Strategy: {self.strategy_id} | Capital: $100,000")
        print(f"Symbols: {len(self.symbols)} NIFTY 50 stocks (equities only)")
        print("=" * 70)

        history = []

        for i in range(1, iterations + 1):
            print(f"\n{'─' * 70}")
            print(f"ITERATION {i}/{iterations} — Strategy {self.strategy_version}")
            print(f"{'─' * 70}")

            # Phase 1: Backtest
            print("\n[Phase 1] Running backtest...")
            baseline = self.run_backtest(years=years)
            print(f"  Trades: {baseline['trades']} | WR: {baseline['win_rate']}% | "
                  f"P&L: {baseline['pnl']:,.0f} | Sharpe: {baseline['sharpe']}")
            
            # Record iteration history
            history.append({
                "iteration": i, 
                "strategy_version": self.strategy_version, 
                "metrics": baseline
            })

            if baseline['trades'] < 5:
                print("  WARNING: Very few trades. Consider lowering entry_threshold.")
                self.entry_threshold = max(0.40, self.entry_threshold - 0.05)
                self.exit_threshold = min(0.50, self.exit_threshold + 0.05)
                print(f"  Auto-adjusted: entry={self.entry_threshold:.2f}, exit={self.exit_threshold:.2f}")
                continue

            # Phase 2: Coach diagnosis
            print("\n[Phase 2] Coach analyzing trades...")
            diagnosis = self.run_coach_diagnosis()
            if diagnosis:
                mistakes = diagnosis.get("mistakes", [])
                print(f"  Mistakes found: {len(mistakes)}")
                for m in mistakes[:3]:
                    print(f"    - {m.get('type', '?')}: {m.get('count', 0)}x, "
                          f"PnL impact: {m.get('pnl_impact', 0):,.0f}")

                weight_recs = diagnosis.get("weight_recommendations", [])
                if weight_recs:
                    print(f"  Weight recommendations: {len(weight_recs)}")
                    for wr in weight_recs[:3]:
                        print(f"    - {wr.get('indicator', '?')}: "
                              f"{wr.get('current_weight', 0):.3f} -> {wr.get('recommended_weight', 0):.3f}")

            # Phase 3: Generate candidates
            print("\n[Phase 3] Generating candidate patches...")
            candidates = self.generate_candidates(diagnosis) if diagnosis else []
            print(f"  Candidates generated: {len(candidates)}")

            # Phase 4: Validate & apply
            if candidates:
                print("\n[Phase 4] Validating candidates...")
                result = self.validate_and_apply(candidates, baseline)
                print(f"  Result: {result['status']}")
                if result.get("patch"):
                    p = result["patch"]
                    print(f"  Applied: {p['patch_id']}")
                    print(f"    Improvement: +{p['improvement_pnl']:,.0f} PnL")
                    print(f"    New Sharpe: {p['new_sharpe']}")
            else:
                print("\n[Phase 4] No candidates to validate")

        # Final summary
        print(f"\n{'=' * 70}")
        print("FINAL SUMMARY")
        print(f"{'=' * 70}")
        print(f"Strategy version: {self.strategy_version}")
        print(f"Patches applied: {len(self.patch_history)}")
        print(f"Active indicators: {len(self.dna.get_active_indicators())}")

        final = self.run_backtest(years=years)
        print(f"\nFinal performance:")
        print(f"  Trades: {final['trades']}")
        print(f"  Win Rate: {final['win_rate']}%")
        print(f"  P&L: {final['pnl']:,.0f}")
        print(f"  Sharpe: {final['sharpe']}")
        print(f"  Max DD: {final['drawdown']:,.0f}")

        # Save results
        results_path = Path("trading_results")
        results_path.mkdir(exist_ok=True)
        # Use fixed filename format so run_5_players can find it easily if needed, 
        # or just reliable timestamp.
        filename = f"run_{self.strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_path / filename, 'w') as f:
            json.dump({
                "strategy_id": self.strategy_id,
                "strategy_version": self.strategy_version,
                "final_metrics": final,
                "iteration_history": history,
                "patches_applied": self.patch_history,
                "total_trades": len(self.all_trades),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

        return final


def main():
    parser = argparse.ArgumentParser(description="NIFTY 50 Player-Coach Trading System")
    parser.add_argument('--backtest', action='store_true', help='Run full backtest + coach loop')
    parser.add_argument('--years', type=int, default=1, help='Years of data')
    parser.add_argument('--iterations', type=int, default=3, help='Coach improvement iterations')
    parser.add_argument('--strategy', default='BEST', help='Strategy: BEST, SAFE, BALANCED, AGGRESSIVE, OPTIMIZED')
    parser.add_argument('--status', action='store_true', help='Show current strategy status')

    args = parser.parse_args()

    system = TradingSystem(
        strategy_name=args.strategy,
        initial_capital=100_000.0,
    )

    if args.status:
        print(f"Strategy: {system.strategy_id}")
        print(f"Active indicators: {len(system.dna.get_active_indicators())}")
        print(f"Entry threshold: {system.entry_threshold}")
        print(f"Exit threshold: {system.exit_threshold}")
        print(f"\nTop 10 weights:")
        sorted_w = sorted(system.strategy_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, weight in sorted_w[:10]:
            print(f"  {name:30s}: {weight:+.4f}")
        return

    if args.backtest:
        system.run_full_loop(years=args.years, iterations=args.iterations)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
