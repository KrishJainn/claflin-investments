"""
Experiment Runner for Coach v2.

Tests candidate patches through backtesting with gates:
- Must pass minimum trade threshold
- Must improve key metrics
- Must not exceed drawdown limits

Player doesn't auto-apply unless ALL gates pass.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

from .patch_language import StrategyPatch
from .candidate_generator import CandidateExperiment


@dataclass
class ExperimentGates:
    """Gates that must pass before applying a patch."""
    
    # Minimum requirements
    min_trades: int = 20
    min_win_rate: float = 0.35
    min_profit_factor: float = 1.0
    
    # Improvement requirements
    require_pnl_improvement: bool = True
    require_sharpe_improvement: bool = True
    min_pnl_improvement_pct: float = 5.0
    
    # Risk limits
    max_drawdown_pct: float = 15.0
    max_drawdown_increase_pct: float = 3.0  # Can't increase DD by more than 3%


@dataclass
class ExperimentResult:
    """Result of running a patch experiment."""
    
    patch_id: str
    
    # Baseline (without patch)
    baseline_pnl: float = 0.0
    baseline_sharpe: float = 0.0
    baseline_win_rate: float = 0.0
    baseline_drawdown: float = 0.0
    baseline_trades: int = 0
    
    # With patch
    patched_pnl: float = 0.0
    patched_sharpe: float = 0.0
    patched_win_rate: float = 0.0
    patched_drawdown: float = 0.0
    patched_trades: int = 0
    
    # Analysis
    pnl_improvement: float = 0.0
    pnl_improvement_pct: float = 0.0
    sharpe_improvement: float = 0.0
    drawdown_change: float = 0.0
    
    # Gates
    gates_passed: List[str] = field(default_factory=list)
    gates_failed: List[str] = field(default_factory=list)
    all_gates_passed: bool = False
    
    # Recommendation
    recommendation: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "patch_id": self.patch_id,
            "baseline": {
                "pnl": self.baseline_pnl,
                "sharpe": self.baseline_sharpe,
                "win_rate": self.baseline_win_rate,
                "drawdown": self.baseline_drawdown,
                "trades": self.baseline_trades,
            },
            "patched": {
                "pnl": self.patched_pnl,
                "sharpe": self.patched_sharpe,
                "win_rate": self.patched_win_rate,
                "drawdown": self.patched_drawdown,
                "trades": self.patched_trades,
            },
            "improvements": {
                "pnl": self.pnl_improvement,
                "pnl_pct": self.pnl_improvement_pct,
                "sharpe": self.sharpe_improvement,
                "drawdown_change": self.drawdown_change,
            },
            "gates_passed": self.gates_passed,
            "gates_failed": self.gates_failed,
            "all_gates_passed": self.all_gates_passed,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
        }


class ExperimentRunner:
    """
    Runs experiments to test candidate patches.

    Process:
    1. Run baseline backtest
    2. Apply patch and run modified backtest
    3. Compare results against gates
    4. Generate recommendation
    """

    def __init__(
        self,
        current_weights: Dict[str, float],
        current_thresholds: Dict[str, float],
        gates: ExperimentGates = None,
        backtest_fn=None,
    ):
        """
        Initialize runner.

        Args:
            current_weights: Current strategy weights
            current_thresholds: Current threshold values
            gates: Gate configuration
            backtest_fn: Callable(weights, thresholds) -> dict with pnl/sharpe/win_rate/drawdown/trades
        """
        self.current_weights = current_weights
        self.current_thresholds = current_thresholds
        self.gates = gates or ExperimentGates()
        self._backtest_fn = backtest_fn

        # Results storage
        self._results: List[ExperimentResult] = []
    
    def run_experiment(
        self,
        candidate: CandidateExperiment,
        baseline_metrics: Dict[str, float] = None,
    ) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            candidate: Candidate to test
            baseline_metrics: Pre-computed baseline metrics (optional)
            
        Returns:
            Experiment result with gate evaluation
        """
        patch = candidate.patch
        
        # Create result
        result = ExperimentResult(patch_id=patch.patch_id)
        
        # Get or compute baseline
        if baseline_metrics:
            result.baseline_pnl = baseline_metrics.get("pnl", 0)
            result.baseline_sharpe = baseline_metrics.get("sharpe", 0)
            result.baseline_win_rate = baseline_metrics.get("win_rate", 0)
            result.baseline_drawdown = baseline_metrics.get("drawdown", 0)
            result.baseline_trades = baseline_metrics.get("trades", 0)
        
        # Apply patch and simulate backtest
        patched_weights = patch.apply_to_weights(self.current_weights)
        patched_thresholds = patch.apply_to_thresholds(self.current_thresholds)
        
        # Run backtest with patched strategy
        # (This would call actual backtester - simplified here)
        patched_metrics = self._run_backtest(patched_weights, patched_thresholds)
        
        result.patched_pnl = patched_metrics.get("pnl", 0)
        result.patched_sharpe = patched_metrics.get("sharpe", 0)
        result.patched_win_rate = patched_metrics.get("win_rate", 0)
        result.patched_drawdown = patched_metrics.get("drawdown", 0)
        result.patched_trades = patched_metrics.get("trades", 0)
        
        # Calculate improvements
        result.pnl_improvement = result.patched_pnl - result.baseline_pnl
        if result.baseline_pnl != 0:
            result.pnl_improvement_pct = (result.pnl_improvement / abs(result.baseline_pnl)) * 100
        result.sharpe_improvement = result.patched_sharpe - result.baseline_sharpe
        result.drawdown_change = result.patched_drawdown - result.baseline_drawdown
        
        # Evaluate gates
        self._evaluate_gates(result)
        
        # Generate recommendation
        self._generate_recommendation(result, candidate)
        
        self._results.append(result)
        
        return result
    
    def _run_backtest(
        self,
        weights: Dict[str, float],
        thresholds: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Run backtest with given weights and thresholds.

        Uses the injected backtest_fn if available; otherwise falls back
        to a lightweight internal simulation using the evolution engine.
        """
        if self._backtest_fn is not None:
            return self._backtest_fn(weights, thresholds)

        # Fallback: use EvolutionOrchestrator._evaluate_dna via import
        try:
            from ..super_indicator.dna import create_dna_from_weights
            from ..super_indicator.core import SuperIndicator
            from ..indicators.universe import IndicatorUniverse
            from ..indicators.calculator import IndicatorCalculator
            from ..indicators.normalizer import IndicatorNormalizer
            from ..data.fetcher import DataFetcher
            from ..data.cache import DataCache

            dna = create_dna_from_weights(weights)
            universe = IndicatorUniverse()
            universe.load_all()
            calculator = IndicatorCalculator(universe=universe)
            normalizer = IndicatorNormalizer()
            si = SuperIndicator(dna, normalizer=normalizer)

            cache = DataCache("data_cache")
            fetcher = DataFetcher(cache=cache, cache_dir="data_cache")

            # Quick backtest on NIFTY 50 subset
            symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
            all_pnls = []
            total_trades = 0
            winners = 0

            entry_thresh = thresholds.get("entry_threshold", 0.70)
            exit_thresh = thresholds.get("exit_threshold", 0.30)

            for sym in symbols:
                df = fetcher.fetch(sym, years=1)
                if df is None or len(df) < 100:
                    continue

                raw = calculator.calculate_all(df)
                raw = calculator.rename_to_dna_names(raw)
                active = [i for i in dna.get_active_indicators() if i in raw.columns]
                if not active:
                    continue

                norm = normalizer.normalize_all(raw[active], price_series=df['close'])
                if norm.empty:
                    continue

                si_vals = si.calculate(norm).fillna(0.0)

                in_pos = False
                entry_price = 0.0
                for idx in range(50, len(df)):
                    val = float(si_vals.iloc[idx])
                    price = float(df.iloc[idx]['close'])

                    if not in_pos and val > entry_thresh:
                        in_pos = True
                        entry_price = price
                    elif in_pos and val < exit_thresh:
                        pnl = price - entry_price
                        all_pnls.append(pnl)
                        total_trades += 1
                        if pnl > 0:
                            winners += 1
                        in_pos = False

            if not all_pnls:
                return {"pnl": 0, "sharpe": 0, "win_rate": 0, "drawdown": 0, "trades": 0}

            import numpy as _np
            pnl_arr = _np.array(all_pnls)
            total_pnl = float(pnl_arr.sum())
            avg = float(pnl_arr.mean())
            std = float(pnl_arr.std()) if len(pnl_arr) > 1 else 1.0
            sharpe = (avg / std * (252 ** 0.5)) if std > 0 else 0.0

            # Drawdown
            equity = _np.cumsum(pnl_arr)
            peak = _np.maximum.accumulate(equity)
            dd = peak - equity
            max_dd = float(dd.max()) if len(dd) > 0 else 0.0
            dd_pct = (max_dd / max(abs(total_pnl), 1.0)) * 100

            return {
                "pnl": total_pnl,
                "sharpe": round(sharpe, 2),
                "win_rate": round(winners / total_trades, 3) if total_trades > 0 else 0,
                "drawdown": round(dd_pct, 1),
                "trades": total_trades,
            }

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Backtest fallback failed: {e}")
            return {"pnl": 0, "sharpe": 0, "win_rate": 0, "drawdown": 0, "trades": 0}
    
    def _evaluate_gates(self, result: ExperimentResult):
        """Evaluate all gates for an experiment result."""
        
        # Gate 1: Minimum trades
        if result.patched_trades >= self.gates.min_trades:
            result.gates_passed.append("min_trades")
        else:
            result.gates_failed.append(f"min_trades: {result.patched_trades} < {self.gates.min_trades}")
        
        # Gate 2: Minimum win rate
        if result.patched_win_rate >= self.gates.min_win_rate:
            result.gates_passed.append("min_win_rate")
        else:
            result.gates_failed.append(f"min_win_rate: {result.patched_win_rate:.1%} < {self.gates.min_win_rate:.1%}")
        
        # Gate 3: P&L improvement required
        if self.gates.require_pnl_improvement:
            if result.pnl_improvement_pct >= self.gates.min_pnl_improvement_pct:
                result.gates_passed.append("pnl_improvement")
            else:
                result.gates_failed.append(f"pnl_improvement: {result.pnl_improvement_pct:.1f}% < {self.gates.min_pnl_improvement_pct:.1f}%")
        
        # Gate 4: Sharpe improvement required
        if self.gates.require_sharpe_improvement:
            if result.sharpe_improvement > 0:
                result.gates_passed.append("sharpe_improvement")
            else:
                result.gates_failed.append(f"sharpe_improvement: {result.sharpe_improvement:.2f} <= 0")
        
        # Gate 5: Max drawdown
        if result.patched_drawdown <= self.gates.max_drawdown_pct:
            result.gates_passed.append("max_drawdown")
        else:
            result.gates_failed.append(f"max_drawdown: {result.patched_drawdown:.1f}% > {self.gates.max_drawdown_pct:.1f}%")
        
        # Gate 6: Drawdown increase limit
        if result.drawdown_change <= self.gates.max_drawdown_increase_pct:
            result.gates_passed.append("drawdown_increase")
        else:
            result.gates_failed.append(f"drawdown_increase: {result.drawdown_change:.1f}% > {self.gates.max_drawdown_increase_pct:.1f}%")
        
        # All gates passed?
        result.all_gates_passed = len(result.gates_failed) == 0
    
    def _generate_recommendation(
        self,
        result: ExperimentResult,
        candidate: CandidateExperiment,
    ):
        """Generate recommendation based on result."""
        
        if result.all_gates_passed:
            result.recommendation = "APPROVE"
            result.confidence = 0.8
            
            if result.pnl_improvement_pct > 20:
                result.recommendation = "STRONG_APPROVE"
                result.confidence = 0.9
        
        elif len(result.gates_failed) == 1:
            result.recommendation = "CONDITIONAL_APPROVE"
            result.confidence = 0.5
        
        elif result.pnl_improvement > 0:
            result.recommendation = "NEEDS_REVIEW"
            result.confidence = 0.3
        
        else:
            result.recommendation = "REJECT"
            result.confidence = 0.7
    
    def run_all_experiments(
        self,
        candidates: List[CandidateExperiment],
        baseline_metrics: Dict[str, float] = None,
    ) -> List[ExperimentResult]:
        """
        Run experiments for all candidates.
        
        Args:
            candidates: List of candidates to test
            baseline_metrics: Pre-computed baseline
            
        Returns:
            List of results sorted by improvement
        """
        results = []
        
        for candidate in candidates:
            result = self.run_experiment(candidate, baseline_metrics)
            results.append(result)
        
        # Sort by PnL improvement
        results.sort(key=lambda x: x.pnl_improvement, reverse=True)
        
        return results
    
    def get_approved_patches(self) -> List[Tuple[StrategyPatch, ExperimentResult]]:
        """Get patches that passed all gates."""
        approved = []
        
        for result in self._results:
            if result.all_gates_passed:
                # Find the original patch
                # (Would need to track this - simplified here)
                pass
        
        return approved
    
    def save_results(self, path: str = "./experiment_results.json"):
        """Save all results to file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "gates": {
                "min_trades": self.gates.min_trades,
                "min_win_rate": self.gates.min_win_rate,
                "min_pnl_improvement_pct": self.gates.min_pnl_improvement_pct,
                "max_drawdown_pct": self.gates.max_drawdown_pct,
            },
            "results": [r.to_dict() for r in self._results],
            "summary": {
                "total_experiments": len(self._results),
                "approved": sum(1 for r in self._results if r.recommendation == "APPROVE"),
                "strong_approved": sum(1 for r in self._results if r.recommendation == "STRONG_APPROVE"),
                "rejected": sum(1 for r in self._results if r.recommendation == "REJECT"),
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
