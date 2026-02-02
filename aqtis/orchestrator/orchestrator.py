"""
AQTIS Multi-Agent Orchestrator.

Coordinates all agents and manages trading workflows:
- Pre-trade: signal analysis -> backtest -> risk check -> execute
- Post-trade: outcome recording -> post-mortem -> strategy updates
- Daily: research scan -> model check -> rolling backtests
- Cross-run: learning persistence across simulation runs

Enhanced with 5-player coach model cross-run learning:
- Simulation run tracking (start/end with P&L)
- Strategy snapshot recording for best-config retrieval
- Indicator-regime stats updates after each trading day
- Coach session integration for bounded weight patches
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from aqtis.memory.memory_layer import MemoryLayer
from aqtis.llm.base import LLMProvider
from aqtis.agents.strategy_generator import StrategyGeneratorAgent
from aqtis.agents.backtester import BacktestingAgent
from aqtis.agents.risk_manager import RiskManagementAgent
from aqtis.agents.researcher import ResearchAgent
from aqtis.agents.post_mortem import PostMortemAgent
from aqtis.agents.prediction_tracker import PredictionTrackingAgent

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Coordinates all AQTIS agents through structured workflows.
    """

    def __init__(
        self,
        memory: MemoryLayer,
        llm: Optional[LLMProvider] = None,
        config: Dict = None,
    ):
        self.memory = memory
        self.llm = llm
        self.config = config or {}

        # Initialize all agents
        self.agents = {
            "strategy_generator": StrategyGeneratorAgent(memory, llm),
            "backtester": BacktestingAgent(memory, llm),
            "risk_manager": RiskManagementAgent(
                memory, llm,
                risk_limits=config.get("risk_limits") if config else None,
            ),
            "researcher": ResearchAgent(memory, llm),
            "post_mortem": PostMortemAgent(memory, llm),
            "prediction_tracker": PredictionTrackingAgent(memory, llm),
        }

    # ─────────────────────────────────────────────────────────────────
    # PRE-TRADE WORKFLOW
    # ─────────────────────────────────────────────────────────────────

    def pre_trade_workflow(self, market_signal: Dict) -> Dict:
        """
        Orchestrate pre-trade decision making.

        Flow:
        1. Strategy Generator: identify opportunity
        2. Backtester: instant backtest
        3. Risk Manager: validate trade
        4. Prediction Tracker: log prediction
        5. Return decision
        """
        logger.info("Starting pre-trade workflow")

        # 1. Strategy Generator identifies opportunity
        opportunity = self.agents["strategy_generator"].run({
            "action": "analyze_signal",
            "signal": market_signal,
        })

        if opportunity.get("error"):
            return {"decision": "error", "reason": opportunity["error"]}

        if not opportunity.get("should_trade", False):
            return {
                "decision": "skip",
                "reason": opportunity.get("reason", "No opportunity"),
            }

        strategy = opportunity.get("strategy", {})

        # 2. Backtest the opportunity
        backtest_result = self.agents["backtester"].run({
            "action": "instant",
            "strategy": strategy,
            "signal": market_signal,
        })

        if backtest_result.get("insufficient_data"):
            logger.warning("Insufficient backtest data, proceeding with caution")

        # 3. Risk validation
        portfolio_value = self.config.get("portfolio_value", 100000)
        confidence = backtest_result.get("confidence", 0.5)

        proposed_trade = {
            "asset": market_signal.get("asset", ""),
            "strategy_id": strategy.get("strategy_id", ""),
            "action": market_signal.get("action", "BUY"),
            "entry_price": market_signal.get("price", 0),
            "confidence": confidence,
            "predicted_return": backtest_result.get("expected_return", 0),
            "portfolio_value": portfolio_value,
        }

        risk_check = self.agents["risk_manager"].run({
            "action": "validate",
            "trade": proposed_trade,
        })

        if not risk_check.get("approved", False):
            return {
                "decision": "reject",
                "reason": risk_check.get("rejection_reasons", ["Risk check failed"]),
                "checks": risk_check.get("checks", {}),
            }

        # 4. Calculate position size
        position_result = self.agents["risk_manager"].run({
            "action": "position_size",
            "prediction": {
                "predicted_confidence": confidence,
                "predicted_return": backtest_result.get("expected_return", 0),
                "asset": market_signal.get("asset", ""),
            },
            "portfolio_value": portfolio_value,
        })

        position_size = position_result.get("position_size", 0)

        # 5. Log prediction
        prediction_result = self.agents["prediction_tracker"].run({
            "action": "record_prediction",
            "prediction": {
                "strategy_id": strategy.get("strategy_id", ""),
                "asset": market_signal.get("asset", ""),
                "predicted_return": backtest_result.get("expected_return", 0),
                "predicted_confidence": confidence,
                "win_probability": backtest_result.get("win_probability", 0.5),
                "predicted_hold_seconds": int(backtest_result.get("expected_hold_seconds", 0)),
                "primary_model": "ensemble",
                "market_features": market_signal,
            },
        })

        prediction_id = prediction_result.get("prediction_id")

        return {
            "decision": "execute",
            "prediction_id": prediction_id,
            "position_size": position_size,
            "strategy": strategy,
            "backtest": backtest_result,
            "risk_check": risk_check,
            "details": {
                "asset": market_signal.get("asset"),
                "action": market_signal.get("action", "BUY"),
                "confidence": confidence,
                "expected_return": backtest_result.get("expected_return", 0),
            },
        }

    # ─────────────────────────────────────────────────────────────────
    # POST-TRADE WORKFLOW
    # ─────────────────────────────────────────────────────────────────

    def post_trade_workflow(self, trade_id: str) -> Dict:
        """
        Orchestrate post-trade analysis.

        Flow:
        1. Record prediction outcome
        2. Deep post-mortem analysis
        3. Strategy updates if actionable insights found
        """
        logger.info(f"Starting post-trade workflow for {trade_id}")

        trade = self.memory.get_trade(trade_id)
        if not trade:
            return {"error": f"Trade {trade_id} not found"}

        # 1. Record prediction outcome
        if trade.get("prediction_id"):
            self.agents["prediction_tracker"].run({
                "action": "record_outcome",
                "prediction_id": trade["prediction_id"],
                "outcome": {
                    "actual_return": trade.get("pnl_percent", 0),
                    "actual_hold_seconds": trade.get("hold_duration_seconds", 0),
                    "actual_max_drawdown": trade.get("max_adverse_excursion", 0),
                },
            })

        # 2. Deep analysis
        analysis = self.agents["post_mortem"].run({
            "action": "analyze_trade",
            "trade_id": trade_id,
        })

        # 3. Strategy updates if needed
        insights = analysis.get("insights")
        if insights and isinstance(insights, dict) and insights.get("actionable_changes"):
            logger.info(f"Actionable changes found for strategy {trade.get('strategy_id')}")
            # Queue strategy improvement (don't auto-apply)

        return {
            "trade_id": trade_id,
            "analysis": analysis,
            "prediction_updated": bool(trade.get("prediction_id")),
        }

    # ─────────────────────────────────────────────────────────────────
    # DAILY ROUTINE
    # ─────────────────────────────────────────────────────────────────

    def daily_routine(self) -> Dict:
        """
        Run daily maintenance tasks.

        Flow:
        1. Research scan
        2. Model degradation check
        3. Rolling backtests for active strategies
        4. Weekly review (if Monday)
        """
        logger.info("Starting daily routine")
        results = {}

        # 1. Research scan
        research = self.agents["researcher"].run({"action": "scan"})
        results["research"] = {
            "papers_scanned": research.get("papers_scanned", 0),
            "relevant_papers": research.get("relevant_papers", 0),
        }

        # 2. Model degradation check
        degradation = self.agents["prediction_tracker"].run({
            "action": "detect_degradation",
        })
        results["degradation"] = degradation.get("degradation_alerts", [])

        if results["degradation"]:
            logger.warning(f"Model degradation detected: {results['degradation']}")

        # 3. Rolling backtests for active strategies
        strategies = self.memory.get_active_strategies()
        backtest_results = {}
        for strategy in strategies:
            bt = self.agents["backtester"].run({
                "action": "rolling",
                "strategy": strategy,
            })
            backtest_results[strategy.get("strategy_id", "")] = {
                "degradation_detected": bt.get("degradation_detected", False),
                "recent_sharpe": bt.get("recent_sharpe", 0),
            }
        results["backtests"] = backtest_results

        # 4. Weekly review (Monday)
        if datetime.now().weekday() == 0:
            review = self.agents["post_mortem"].run({"action": "weekly_review"})
            results["weekly_review"] = review

        return results

    # ─────────────────────────────────────────────────────────────────
    # CROSS-RUN LEARNING (from 5-player coach model)
    # ─────────────────────────────────────────────────────────────────

    def start_simulation_run(
        self, run_number: int, session_id: str, days: int, symbols: int, config: Dict = None
    ) -> int:
        """
        Record the start of a new simulation run.

        This enables cross-run learning where each run builds on
        knowledge from previous runs.
        """
        run_id = self.memory.start_run(
            run_number=run_number,
            session_id=session_id,
            days=days,
            symbols=symbols,
            config=config or {},
        )
        logger.info(f"Started simulation run #{run_number} (id={run_id})")
        return run_id

    def end_simulation_run(self, run_id: int, team_pnl: float, team_return: float, notes: str = ""):
        """
        Record the end of a simulation run with results.

        Triggers post-run analysis including:
        - Indicator-regime stats update
        - Strategy snapshot for best-config retrieval
        """
        self.memory.end_run(run_id, team_pnl, team_return, notes)
        logger.info(f"Ended simulation run {run_id}: P&L=${team_pnl:,.0f}, Return={team_return:.2f}%")

        # Update indicator-regime stats from this run's trades
        strategies = self.memory.get_active_strategies()
        for strategy in strategies:
            sid = strategy.get("strategy_id", "")
            if sid:
                self.memory.update_indicator_regime_stats(sid)

    def record_strategy_snapshot(
        self, strategy_id: str, run_number: int, weights: Dict,
        sharpe: float, win_rate: float, net_pnl: float, extra: Dict = None,
        total_trades: int = 0, max_drawdown: float = 0, profit_factor: float = 0,
    ) -> int:
        """
        Record a strategy configuration snapshot.

        The best snapshot across all runs is retrieved during strategy
        improvement to restore proven configurations.
        """
        entry_threshold = (extra or {}).get("entry_threshold", 0)
        return self.memory.record_strategy_snapshot(
            strategy_id=strategy_id,
            snapshot_type="end",
            label=f"run_{run_number}",
            weights=weights,
            entry_threshold=entry_threshold,
            total_trades=total_trades,
            win_rate=win_rate,
            net_pnl=net_pnl,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            regime_performance=extra,
        )

    def end_of_day_learning(self, trading_date: str = "") -> Dict:
        """
        Run end-of-day learning cycle (inspired by 5-player coach feedback loop).

        Flow:
        1. Update indicator-regime stats for all active strategies
        2. Run post-mortem on today's trades
        3. Record coach sessions with improvement advice
        4. Store strategy snapshots if performance improved
        """
        logger.info("Running end-of-day learning cycle")
        results = {}

        if not trading_date:
            trading_date = datetime.now().strftime("%Y-%m-%d")

        strategies = self.memory.get_active_strategies()

        for strategy in strategies:
            sid = strategy.get("strategy_id", "")
            if not sid:
                continue

            # 1. Update indicator-regime stats
            self.memory.update_indicator_regime_stats(sid)

            # 2. Get today's trades for this strategy
            today_trades = self.memory.get_trades(
                strategy_id=sid, start_date=trading_date
            )

            if not today_trades:
                continue

            # 3. Run post-mortem on each trade
            for trade in today_trades:
                trade_id = trade.get("trade_id")
                if trade_id:
                    self.agents["post_mortem"].run({
                        "action": "analyze_trade",
                        "trade_id": trade_id,
                    })

            # 4. Calculate today's performance
            pnls = [t.get("pnl", 0) or 0 for t in today_trades]
            wins = sum(1 for p in pnls if p > 0)
            total_pnl = sum(pnls)
            win_rate = wins / len(today_trades) if today_trades else 0

            results[sid] = {
                "trades": len(today_trades),
                "wins": wins,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
            }

        results["timestamp"] = datetime.now().isoformat()
        return results

    def get_cross_run_summary(self) -> str:
        """Get a human-readable cross-run learning summary."""
        return self.memory.context.build_cross_run_summary()

    # ─────────────────────────────────────────────────────────────────
    # INTELLIGENT SIGNAL FILTERING (0 LLM tokens)
    # ─────────────────────────────────────────────────────────────────

    def lightweight_signal_check(self, signal: Dict) -> Dict:
        """
        Fast pre-filter using accumulated memory. Zero LLM calls.

        Reads indicator-regime stats, similar trades, coach sessions,
        prediction calibration, and asset history to adjust signal score.
        """
        raw_score = signal.get("score", 0)
        symbol = signal.get("symbol", signal.get("asset", ""))
        regime = signal.get("regime", signal.get("market_regime", "unknown"))
        raw_signals = signal.get("signals", {})

        adjustments = []
        adjusted_score = raw_score

        # 1. Indicator-regime reweighting
        try:
            regime_stats = self.memory.get_top_indicators_for_regime(regime, limit=20)
            if regime_stats:
                regime_eff = {}
                for stat in regime_stats:
                    ind = stat.get("indicator")
                    if ind and stat.get("total_trades", 0) >= 5:
                        regime_eff[ind] = stat.get("contribution_score", 0)

                aligned = sum(1 for k in raw_signals if regime_eff.get(k, 0) > 0.1)
                misaligned = sum(1 for k in raw_signals if regime_eff.get(k, 0) < -0.1)

                if aligned > 3:
                    adjusted_score += 0.05 if raw_score > 0 else -0.05
                    adjustments.append(f"+0.05 ({aligned} aligned indicators)")
                if misaligned > 2:
                    adjusted_score -= 0.08 if raw_score > 0 else -0.08
                    adjustments.append(f"-0.08 ({misaligned} misaligned indicators)")
        except Exception:
            pass

        # 2. Regime win-rate gate
        try:
            all_regime_stats = self.memory.get_indicator_regime_stats(regime)
            if all_regime_stats:
                valid = [s for s in all_regime_stats if s.get("total_trades", 0) >= 5]
                if valid:
                    avg_wr = np.mean([s.get("win_rate", 0.5) for s in valid])
                    if avg_wr < 0.30:
                        adjusted_score *= 0.5
                        adjustments.append(f"*0.5 bad regime WR={avg_wr:.0%}")
        except Exception:
            pass

        # 3. Asset-specific history
        try:
            asset_trades = self.memory.get_trades(asset=symbol, limit=20)
            if len(asset_trades) >= 5:
                wins = sum(1 for t in asset_trades if (t.get("pnl") or 0) > 0)
                wr = wins / len(asset_trades)
                if wr < 0.25:
                    adjusted_score *= 0.6
                    adjustments.append(f"*0.6 poor asset {symbol} WR={wr:.0%}")
                elif wr > 0.70:
                    adjusted_score *= 1.15
                    adjustments.append(f"*1.15 strong asset {symbol} WR={wr:.0%}")
        except Exception:
            pass

        # 4. Similar trade outcome check
        try:
            similar = self.memory.get_similar_trades(signal, top_k=20)
            if len(similar) >= 5:
                sim_wr = sum(1 for t in similar if (t.get("pnl") or 0) > 0) / len(similar)
                if sim_wr < 0.30:
                    adjusted_score *= 0.7
                    adjustments.append(f"*0.7 similar trades WR={sim_wr:.0%}")
                elif sim_wr > 0.65:
                    adjusted_score *= 1.1
                    adjustments.append(f"*1.1 similar trades WR={sim_wr:.0%}")
        except Exception:
            pass

        # 5. Coach advice integration
        try:
            recent_coach = self.memory.get_recent_coach_sessions(limit=3)
            for session in recent_coach:
                advice = (session.get("advice") or "").lower()
                s_regime = session.get("regime", "")
                if s_regime == regime and "avoid" in advice:
                    adjusted_score *= 0.7
                    adjustments.append(f"*0.7 coach: avoid {regime}")
                    break
                if "selective" in advice or "raise threshold" in advice:
                    if abs(adjusted_score) < 0.20:
                        adjusted_score *= 0.8
                        adjustments.append("*0.8 coach: be selective")
        except Exception:
            pass

        # 6. Prediction calibration
        try:
            raw_conf = min(abs(adjusted_score), 0.8)
            calibrated = self.agents["prediction_tracker"].get_calibrated_confidence(
                raw_conf, {"market_regime": regime, "asset": symbol}
            )
        except Exception:
            calibrated = min(abs(adjusted_score), 0.8)

        # 7. Circuit breaker
        if self.agents["risk_manager"].circuit_breaker_active:
            return {
                "adjusted_score": 0, "should_trade": False,
                "confidence": 0, "reason": "Circuit breaker active",
                "adjustments": adjustments,
            }

        adjusted_score = float(np.clip(adjusted_score, -1.0, 1.0))

        return {
            "adjusted_score": adjusted_score,
            "original_score": float(raw_score),
            "confidence": float(calibrated),
            "should_trade": abs(adjusted_score) > 0.10,
            "adjustments": adjustments,
        }

    def apply_daily_learnings(self) -> Dict:
        """
        Extract weight adjustments from recent coach sessions and
        indicator-regime stats. Zero LLM calls — reads memory only.
        """
        weight_adjustments = {}
        reasoning = []

        # Read most recent coach session
        try:
            sessions = self.memory.get_recent_coach_sessions(limit=1)
            if sessions:
                session = sessions[0]
                recs = session.get("weight_recs_json") or session.get("patch_json")
                if recs:
                    try:
                        recs_dict = json.loads(recs) if isinstance(recs, str) else recs
                        for ind, delta in recs_dict.items():
                            weight_adjustments[ind] = max(-0.05, min(0.05, float(delta)))
                        reasoning.append(f"Coach recs: {len(recs_dict)} indicators adjusted")
                    except (json.JSONDecodeError, TypeError):
                        pass
        except Exception:
            pass

        # Read indicator-regime stats to find consistently bad indicators
        try:
            stats = self.memory.get_indicator_regime_stats()
            if stats:
                for s in stats:
                    if s.get("total_trades", 0) >= 10:
                        ind = s["indicator"]
                        if s.get("avg_pnl", 0) < -5 and ind not in weight_adjustments:
                            weight_adjustments[ind] = -0.03
                            reasoning.append(f"Demote {ind} (avg P&L={s['avg_pnl']:.0f})")
                        elif s.get("avg_pnl", 0) > 5 and ind not in weight_adjustments:
                            weight_adjustments[ind] = 0.03
                            reasoning.append(f"Promote {ind} (avg P&L={s['avg_pnl']:.0f})")
        except Exception:
            pass

        # Threshold adjustment based on recent win rate
        threshold_adj = 0.0
        try:
            trades = self.memory.get_trades(limit=50)
            if len(trades) >= 10:
                wr = sum(1 for t in trades if (t.get("pnl") or 0) > 0) / len(trades)
                if wr < 0.40:
                    threshold_adj = 0.02
                    reasoning.append(f"Raise threshold (WR={wr:.0%})")
                elif wr > 0.65:
                    threshold_adj = -0.01
                    reasoning.append(f"Lower threshold (WR={wr:.0%})")
        except Exception:
            pass

        return {
            "weight_adjustments": weight_adjustments,
            "threshold_adjustment": threshold_adj,
            "reasoning": reasoning,
        }

    def get_best_weights(self, strategy_id: str = "aqtis_multi_indicator") -> Optional[Dict]:
        """Load best strategy snapshot weights from memory."""
        try:
            snap = self.memory.get_best_strategy_snapshot(strategy_id)
            if snap:
                weights = json.loads(snap.get("weights_json", "{}"))
                if weights:
                    return {
                        "weights": weights,
                        "entry_threshold": snap.get("entry_threshold", 0.15),
                        "sharpe": snap.get("sharpe", 0),
                        "source": "snapshot",
                    }
        except Exception:
            pass
        return None

    def mid_run_strategy_review(self, metrics: Dict, current_weights: Dict) -> Dict:
        """
        Single LLM call to evaluate mid-run performance and suggest adjustments.
        """
        if not self.llm:
            return {}

        # Build compact context
        ctx = self.memory.context.build_strategy_context("aqtis_multi_indicator")
        regime_breakdown = metrics.get("regime_breakdown", {})
        regime_str = ", ".join(
            f"{r}: {s.get('trades', 0)}t/{s.get('wins', 0)}w"
            for r, s in regime_breakdown.items()
        ) if regime_breakdown else "N/A"

        top_weights = sorted(current_weights.items(), key=lambda x: -abs(x[1]))[:8]
        weights_str = ", ".join(f"{k}={v:.3f}" for k, v in top_weights)

        prompt = f"""You are AQTIS strategy optimizer. Review mid-run performance and suggest weight changes.

PERFORMANCE SO FAR:
- P&L: ${metrics.get('total_pnl', 0):+,.2f}, Sharpe: {metrics.get('sharpe_ratio', 0):.2f}
- Win Rate: {metrics.get('win_rate', 0):.0%}, Trades: {metrics.get('total_trades', 0)}
- Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
- Regimes: {regime_str}

TOP WEIGHTS: {weights_str}

MEMORY CONTEXT:
{ctx}

Respond in JSON with:
1. "weight_changes": dict of indicator_name -> new_weight (max 8, bounds [0.01, 0.50])
2. "threshold": entry threshold (0.08-0.35)
3. "reasoning": 2 sentences on what to adjust and why
"""
        try:
            result = self.llm.generate_json(prompt)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.warning(f"Mid-run review failed: {e}")
            return {}

    # ─────────────────────────────────────────────────────────────────
    # STATUS
    # ─────────────────────────────────────────────────────────────────

    def get_agent_statuses(self) -> Dict:
        """Get status of all agents."""
        return {name: agent.get_status() for name, agent in self.agents.items()}

    def get_system_health(self) -> Dict:
        """Get overall system health."""
        statuses = self.get_agent_statuses()
        memory_stats = self.memory.get_stats()

        # Include cross-run intelligence
        pnl_trend = self.memory.get_cross_run_pnl_trend()

        return {
            "agents": statuses,
            "memory": memory_stats,
            "circuit_breaker": self.agents["risk_manager"].circuit_breaker_active,
            "cross_run_trend": pnl_trend,
            "timestamp": datetime.now().isoformat(),
        }
