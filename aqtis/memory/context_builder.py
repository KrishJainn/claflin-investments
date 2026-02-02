"""
Context builder for LLM prompts.

Queries the memory database and generates concise, actionable
summaries that fit within LLM token budgets.

Ported from the 5-player coach model's context_builder and adapted
for the AQTIS multi-agent architecture.
"""

import json
import logging
from collections import Counter
from typing import Dict, List, Optional

from .database import StructuredDB

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Generates concise LLM-ready context from the AQTIS memory database."""

    def __init__(self, db: StructuredDB):
        self.db = db

    def build_strategy_context(self, strategy_id: str) -> str:
        """Build context for strategy optimization.

        Returns ~800 chars summarizing:
        - Best historical config for this strategy
        - Top/worst assets
        - Win rate by market regime
        - Which indicators work in which regimes
        - Coach advice trends
        """
        lines = []
        run_count = self.db.get_run_count()

        lines.append(f"MEMORY: {run_count} simulation run(s) completed")

        # Best strategy snapshot
        best = self.db.get_best_strategy_snapshot(strategy_id)
        if best:
            lines.append(
                f"Best config: Sharpe {best.get('sharpe', 0):.2f}, "
                f"WR {best.get('win_rate', 0):.1f}%, P&L ${best.get('net_pnl', 0):,.0f}"
            )
            try:
                weights = json.loads(best.get("weights_json", "{}"))
                top_w = sorted(weights.items(), key=lambda x: -abs(x[1]))[:4]
                w_str = ", ".join(f"{k}={v:.2f}" for k, v in top_w)
                lines.append(f"  Top weights: {w_str}")
            except (json.JSONDecodeError, TypeError):
                pass

        # Asset performance
        asset_stats = self.db.get_strategy_asset_performance(strategy_id)
        if asset_stats:
            best_assets = [s for s in asset_stats if s["total_pnl"] > 0][:3]
            worst_assets = [s for s in reversed(asset_stats) if s["total_pnl"] < 0][:3]
            if best_assets:
                parts = [f"{s['asset']} (+${s['total_pnl']:,.0f})" for s in best_assets]
                lines.append(f"Best assets: {', '.join(parts)}")
            if worst_assets:
                parts = [f"{s['asset']} (${s['total_pnl']:,.0f})" for s in worst_assets]
                lines.append(f"Worst assets: {', '.join(parts)}")

        # Regime performance
        regime_stats = self.db.get_strategy_regime_performance(strategy_id)
        if regime_stats:
            regime_parts = []
            for rs in regime_stats:
                if rs["trades"] >= 5:
                    wr = rs["wins"] / rs["trades"] * 100 if rs["trades"] > 0 else 0
                    regime_parts.append(f"{rs['market_regime']} WR={wr:.0f}%")
            if regime_parts:
                lines.append(f"By regime: {', '.join(regime_parts[:4])}")

        # Indicator-regime effectiveness
        ind_stats = self.db.get_indicator_regime_stats()
        if ind_stats:
            best_inds = [s for s in ind_stats
                         if s["total_trades"] >= 10 and s["avg_pnl"] > 0][:3]
            if best_inds:
                parts = [
                    f"{s['indicator']} in {s['regime']} (+${s['avg_pnl']:.0f}/trade)"
                    for s in best_inds
                ]
                lines.append(f"Top indicators: {', '.join(parts)}")

        # Coach advice trends
        sessions = self.db.get_recent_coach_sessions(strategy_id, limit=5)
        if sessions:
            advice_snippets = [s["advice"][:50] for s in sessions if s.get("advice")]
            if advice_snippets:
                lines.append(f"Recent coach: \"{advice_snippets[0]}\"")

        # Cross-run P&L trend
        pnl_trend = self.db.get_cross_run_pnl_trend()
        if len(pnl_trend) >= 2:
            trend_str = " -> ".join(f"${r['team_pnl']:,.0f}" for r in pnl_trend[-5:])
            lines.append(f"P&L trend: {trend_str}")

        return "\n".join(lines) if lines else "No historical data available."

    def build_coach_context(self, strategy_id: str, trading_date: str = "") -> str:
        """Build context for the daily coach prompt.

        Returns ~500 chars summarizing:
        - Last 3 coach patches and their outcomes
        - Asset alerts (consistently winning/losing)
        - Recent market regime trend
        - Strategy's average performance stats
        """
        lines = []

        lines.append(f"HISTORY for {strategy_id}:")

        # Last 3 coach patches with outcomes
        effectiveness = self.db.get_coach_advice_effectiveness(strategy_id)
        if effectiveness:
            recent = effectiveness[-3:]
            patch_parts = []
            for e in recent:
                outcome = "helped" if e["helped"] else "hurt"
                patch_parts.append(f"{e['regime']}->{e['advice'][:30]}({outcome})")
            if patch_parts:
                lines.append(f"Last patches: {'; '.join(patch_parts)}")

        # Asset alerts
        asset_stats = self.db.get_strategy_asset_performance(strategy_id)
        if asset_stats:
            alerts = []
            for s in asset_stats:
                if s["trades"] >= 3:
                    wr = s["wins"] / s["trades"] if s["trades"] > 0 else 0
                    if wr <= 0.2:
                        alerts.append(f"{s['asset']} {s['wins']}/{s['trades']} wins (avoid)")
                    elif wr >= 0.7:
                        alerts.append(f"{s['asset']} {s['wins']}/{s['trades']} wins (prefer)")
            if alerts:
                lines.append(f"Asset alerts: {', '.join(alerts[:4])}")

        # Recent market regime trend
        states = self.db.get_market_state_history(days=7)
        if states:
            regimes = [s.get("vol_regime", "unknown") for s in states if s.get("vol_regime")]
            if regimes:
                regime_counts = Counter(regimes)
                r_str = ", ".join(f"{cnt} {reg}" for reg, cnt in regime_counts.most_common(3))
                lines.append(f"Recent regime: {r_str} (last {len(regimes)} days)")

        # Strategy average stats
        all_trades = self.db.get_trades(strategy_id=strategy_id, limit=200)
        if all_trades:
            total = len(all_trades)
            wins = sum(1 for t in all_trades if (t.get("pnl") or 0) > 0)
            avg_win = sum(t["pnl"] for t in all_trades if (t.get("pnl") or 0) > 0)
            avg_loss = sum(t["pnl"] for t in all_trades if (t.get("pnl") or 0) <= 0)
            n_win = max(1, wins)
            n_loss = max(1, total - wins)
            lines.append(
                f"Stats (last {total}): WR={wins / total * 100:.0f}%, "
                f"avg win=${avg_win / n_win:,.0f}, avg loss=${avg_loss / n_loss:,.0f}"
            )

        return "\n".join(lines)

    def build_cross_run_summary(self) -> str:
        """Build a cross-run comparison summary."""
        pnl_trend = self.db.get_cross_run_pnl_trend()
        if not pnl_trend:
            return "No completed runs yet."

        lines = [f"CROSS-RUN SUMMARY ({len(pnl_trend)} runs):"]

        for r in pnl_trend:
            lines.append(
                f"  Run {r['run_number']}: ${r['team_pnl']:,.0f} ({r['team_return']:.2f}%)"
            )

        best = max(pnl_trend, key=lambda x: x["team_pnl"])
        lines.append(f"Best: Run {best['run_number']} (${best['team_pnl']:,.0f})")

        if len(pnl_trend) >= 2:
            first_pnl = pnl_trend[0]["team_pnl"]
            last_pnl = pnl_trend[-1]["team_pnl"]
            if first_pnl != 0:
                improvement = (last_pnl - first_pnl) / abs(first_pnl) * 100
                lines.append(f"Overall improvement: {improvement:+.1f}%")

        return "\n".join(lines)

    def build_pre_trade_context(self, signal: Dict) -> str:
        """Build context for pre-trade decision making.

        Combines regime intelligence, indicator effectiveness, and
        historical similar trade outcomes.
        """
        lines = []
        regime = signal.get("market_regime", "unknown")

        # Top indicators for this regime
        top_inds = self.db.get_top_indicators_for_regime(regime, limit=5)
        if top_inds:
            parts = [
                f"{i['indicator']} (WR={i['win_rate']:.0%}, score={i['contribution_score']:.2f})"
                for i in top_inds
            ]
            lines.append(f"Best indicators for {regime}: {', '.join(parts)}")

        # Asset-specific history
        asset = signal.get("asset")
        if asset:
            trades = self.db.get_trades(asset=asset, limit=20)
            if trades:
                total = len(trades)
                wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
                avg_pnl = sum(t.get("pnl") or 0 for t in trades) / total
                lines.append(
                    f"{asset} history: {total} trades, WR={wins / total * 100:.0f}%, "
                    f"avg P&L=${avg_pnl:,.0f}"
                )

        # Recent coach wisdom
        sessions = self.db.get_recent_coach_sessions(limit=3)
        if sessions:
            for s in sessions[:2]:
                if s.get("advice"):
                    lines.append(f"Coach ({s.get('regime', '?')}): {s['advice'][:60]}")

        return "\n".join(lines) if lines else ""

    def build_full_optimization_context(self, strategy_id: str) -> str:
        """Build comprehensive context for strategy redesign.

        Combines all available intelligence sources.
        """
        parts = []

        strategy_ctx = self.build_strategy_context(strategy_id)
        if strategy_ctx and strategy_ctx != "No historical data available.":
            parts.append(strategy_ctx)

        coach_ctx = self.build_coach_context(strategy_id)
        if coach_ctx:
            parts.append(coach_ctx)

        cross_run = self.build_cross_run_summary()
        if cross_run and cross_run != "No completed runs yet.":
            parts.append(cross_run)

        return "\n\n".join(parts) if parts else "No optimization context available."
