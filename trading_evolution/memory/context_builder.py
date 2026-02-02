"""
Context builder for LLM prompts.

Queries the memory database and generates concise, actionable
summaries that fit within LLM token budgets.
"""

import json
import logging
from typing import Dict, List, Optional

from .memory_db import MemoryDB

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Generates concise LLM-ready context from the memory database."""

    def __init__(self, db: MemoryDB):
        self.db = db

    def build_optimizer_context(self, player_id: str) -> str:
        """Build context for the pre-simulation strategy optimizer.

        Returns ~800 chars summarizing:
        - Best historical config for this player
        - Top/worst symbols
        - Win rate by market regime
        - Which indicators work in which regimes
        - Coach advice trends
        """
        lines = []
        run_count = self.db.get_run_count()
        if run_count == 0:
            return "No previous run data available."

        lines.append(f"MEMORY: {run_count} previous run(s) for {player_id}")

        # Best strategy snapshot
        best = self.db.get_best_strategy_snapshot(player_id)
        if best:
            lines.append(
                f"Best config: Sharpe {best['sharpe']:.2f}, "
                f"WR {best['win_rate']:.1f}%, P&L ${best['net_pnl']:,.0f} "
                f"(Run {best['run_id']})"
            )
            try:
                weights = json.loads(best["weights_json"])
                top_w = sorted(weights.items(), key=lambda x: -abs(x[1]))[:4]
                w_str = ", ".join(f"{k}={v:.2f}" for k, v in top_w)
                lines.append(f"  Top weights: {w_str}")
            except (json.JSONDecodeError, TypeError):
                pass

        # Symbol performance
        sym_stats = self.db.get_player_symbol_stats(player_id)
        if sym_stats:
            best_syms = [s for s in sym_stats if s["total_pnl"] > 0][:3]
            worst_syms = [s for s in reversed(sym_stats) if s["total_pnl"] < 0][:3]
            if best_syms:
                parts = [f"{s['symbol'].replace('.NS','')} (+${s['total_pnl']:,.0f})"
                         for s in best_syms]
                lines.append(f"Best symbols: {', '.join(parts)}")
            if worst_syms:
                parts = [f"{s['symbol'].replace('.NS','')} (${s['total_pnl']:,.0f})"
                         for s in worst_syms]
                lines.append(f"Worst symbols: {', '.join(parts)}")

        # Regime performance
        regime_stats = self.db.get_player_regime_stats(player_id)
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
                parts = [f"{s['indicator']} in {s['regime']} (+${s['avg_pnl']:.0f}/trade)"
                         for s in best_inds]
                lines.append(f"Top indicators: {', '.join(parts)}")

        # Coach advice trends
        sessions = self.db.get_recent_coach_sessions(player_id, limit=5)
        if sessions:
            advice_snippets = [s["advice"][:50] for s in sessions if s.get("advice")]
            if advice_snippets:
                lines.append(f"Recent coach: \"{advice_snippets[0]}\"")

        # Cross-run P&L trend
        pnl_trend = self.db.get_cross_run_pnl_trend()
        if len(pnl_trend) >= 2:
            trend_str = " → ".join(f"${r['team_pnl']:,.0f}" for r in pnl_trend[-5:])
            lines.append(f"Team P&L trend: {trend_str}")

        return "\n".join(lines)

    def build_coach_context(self, player_id: str,
                            trading_date: str = "") -> str:
        """Build context for the daily expert coach prompt.

        Returns ~500 chars summarizing:
        - Last 3 coach patches and their outcomes
        - Symbol alerts (consistently winning/losing)
        - Recent market regime trend
        - Player's average performance stats
        """
        lines = []
        run_count = self.db.get_run_count()
        if run_count == 0:
            return ""

        lines.append(f"HISTORY for {player_id}:")

        # Last 3 coach patches with outcomes
        effectiveness = self.db.get_coach_advice_effectiveness(player_id)
        if effectiveness:
            recent = effectiveness[-3:]
            patch_parts = []
            for e in recent:
                outcome = "helped" if e["helped"] else "hurt"
                patch_parts.append(
                    f"{e['regime']}→{e['advice'][:30]}({outcome})"
                )
            if patch_parts:
                lines.append(f"Last patches: {'; '.join(patch_parts)}")

        # Symbol alerts
        sym_stats = self.db.get_player_symbol_stats(player_id)
        if sym_stats:
            alerts = []
            for s in sym_stats:
                if s["trades"] >= 3:
                    wr = s["wins"] / s["trades"] if s["trades"] > 0 else 0
                    sym_short = s["symbol"].replace(".NS", "")
                    if wr <= 0.2:
                        alerts.append(f"{sym_short} {s['wins']}/{s['trades']} wins (avoid)")
                    elif wr >= 0.7:
                        alerts.append(f"{sym_short} {s['wins']}/{s['trades']} wins (prefer)")
            if alerts:
                lines.append(f"Symbol alerts: {', '.join(alerts[:4])}")

        # Recent market regime trend
        snapshots = self.db.get_recent_market_snapshots(limit=5)
        if snapshots:
            regimes = [s["nifty_regime"] for s in snapshots if s.get("nifty_regime")]
            if regimes:
                from collections import Counter
                regime_counts = Counter(regimes)
                r_str = ", ".join(f"{cnt} {reg}" for reg, cnt in regime_counts.most_common(3))
                lines.append(f"Recent regime: {r_str} (last {len(regimes)} days)")

        # Player average stats
        all_trades = self.db.get_player_trades(player_id, limit=200)
        if all_trades:
            total = len(all_trades)
            wins = sum(1 for t in all_trades if (t.get("net_pnl") or 0) > 0)
            avg_win = sum(t["net_pnl"] for t in all_trades if (t.get("net_pnl") or 0) > 0)
            avg_loss = sum(t["net_pnl"] for t in all_trades if (t.get("net_pnl") or 0) <= 0)
            n_win = max(1, wins)
            n_loss = max(1, total - wins)
            lines.append(
                f"Stats (last {total}): WR={wins/total*100:.0f}%, "
                f"avg win=${avg_win/n_win:,.0f}, avg loss=${avg_loss/n_loss:,.0f}"
            )

        return "\n".join(lines)

    def build_cross_run_summary(self) -> str:
        """Build a cross-run comparison summary.

        Returns ~400 chars summarizing P&L trends and key turning points.
        """
        pnl_trend = self.db.get_cross_run_pnl_trend()
        if not pnl_trend:
            return "No completed runs yet."

        lines = [f"CROSS-RUN SUMMARY ({len(pnl_trend)} runs):"]

        # P&L progression
        for r in pnl_trend:
            lines.append(
                f"  Run {r['run_number']}: ${r['team_pnl']:,.0f} ({r['team_return']:.2f}%)"
            )

        # Best run
        best = max(pnl_trend, key=lambda x: x["team_pnl"])
        lines.append(
            f"Best: Run {best['run_number']} (${best['team_pnl']:,.0f})"
        )

        # Improvement trend
        if len(pnl_trend) >= 2:
            first_pnl = pnl_trend[0]["team_pnl"]
            last_pnl = pnl_trend[-1]["team_pnl"]
            if first_pnl != 0:
                improvement = (last_pnl - first_pnl) / abs(first_pnl) * 100
                lines.append(f"Overall improvement: {improvement:+.1f}%")

        return "\n".join(lines)
