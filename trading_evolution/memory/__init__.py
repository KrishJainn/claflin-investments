"""
Memory & Context Layer for the 5-Player Trading System.

Provides persistent storage of all trades, coach analyses, market state,
and strategy configurations across simulation runs. Generates concise
context summaries for LLM prompts.
"""

import json
import uuid
import logging
from typing import Dict, List, Optional

from .memory_db import MemoryDB
from .context_builder import ContextBuilder

logger = logging.getLogger(__name__)


class MemoryManager:
    """Orchestrates memory reads and writes for the trading simulation."""

    def __init__(self, db_path: str = "trading_memory.db"):
        self.db = MemoryDB(db_path)
        self.context = ContextBuilder(self.db)
        self.current_run_id: Optional[int] = None
        self.session_id: str = str(uuid.uuid4())[:8]
        logger.info(f"[Memory] Initialized (session={self.session_id}, db={db_path})")

    # ─────────────────────────────────────────────────────────────────
    # WRITE METHODS — called during simulation
    # ─────────────────────────────────────────────────────────────────

    def start_run(self, run_number: int, days: int, symbols: int,
                  config: Dict = None) -> int:
        """Record the start of a new simulation run."""
        self.current_run_id = self.db.start_run(
            run_number=run_number,
            session_id=self.session_id,
            days=days,
            symbols=symbols,
            config=config or {}
        )
        logger.info(f"[Memory] Started run {run_number} (run_id={self.current_run_id})")
        return self.current_run_id

    def end_run(self, team_pnl: float, team_return: float):
        """Record the completion of a simulation run and update stats."""
        if self.current_run_id is None:
            return
        self.db.end_run(self.current_run_id, team_pnl, team_return)
        self.db.update_indicator_regime_stats()
        run_count = self.db.get_run_count()
        logger.info(
            f"[Memory] Completed run (run_id={self.current_run_id}, "
            f"P&L=${team_pnl:,.0f}, total_runs={run_count})"
        )

    def record_trade(self, player_id: str, trade: Dict,
                     market_regime: str = "unknown"):
        """Record a single trade."""
        if self.current_run_id is None:
            return
        self.db.record_trade(self.current_run_id, player_id, trade, market_regime)

    def record_trades_batch(self, player_id: str, trades: List[Dict],
                            market_regime: str = "unknown"):
        """Record multiple trades at once (more efficient for daily batches)."""
        if self.current_run_id is None or not trades:
            return
        self.db.record_trades_batch(
            self.current_run_id, player_id, trades, market_regime
        )

    def record_coach_session(self, player_id: str, trading_date: str,
                             regime: str, advice: str,
                             weight_changes: int = 0,
                             entry_change: float = 0.0,
                             exit_change: float = 0.0,
                             mistakes: List = None,
                             weight_recs: List = None):
        """Record a coach analysis and patch application."""
        if self.current_run_id is None:
            return
        self.db.record_coach_session(
            self.current_run_id, player_id, trading_date,
            regime, advice, weight_changes, entry_change, exit_change,
            mistakes, weight_recs
        )

    def record_market_snapshot(self, trading_date: str,
                               advancers: int = 0, decliners: int = 0,
                               avg_change_pct: float = 0.0,
                               market_bias: str = "NEUTRAL",
                               top_movers: List[Dict] = None,
                               nifty_regime: str = "unknown"):
        """Record daily market state."""
        if self.current_run_id is None:
            return
        self.db.record_market_snapshot(
            self.current_run_id, trading_date,
            advancers, decliners, avg_change_pct, market_bias,
            top_movers, nifty_regime
        )

    def record_strategy_snapshot(self, player_id: str, snapshot_type: str,
                                  label: str, weights: Dict,
                                  entry_threshold: float,
                                  exit_threshold: float,
                                  min_hold_bars: int,
                                  total_trades: int = 0,
                                  win_rate: float = 0.0,
                                  net_pnl: float = 0.0,
                                  sharpe: float = 0.0):
        """Record a player's strategy configuration (start or end of run)."""
        if self.current_run_id is None:
            return
        self.db.record_strategy_snapshot(
            self.current_run_id, player_id, snapshot_type, label,
            weights, entry_threshold, exit_threshold, min_hold_bars,
            total_trades, win_rate, net_pnl, sharpe
        )

    # ─────────────────────────────────────────────────────────────────
    # READ METHODS — for LLM prompts
    # ─────────────────────────────────────────────────────────────────

    def get_optimizer_context(self, player_id: str) -> str:
        """Get historical context for the strategy optimizer prompt."""
        return self.context.build_optimizer_context(player_id)

    def get_coach_context(self, player_id: str,
                          trading_date: str = "") -> str:
        """Get historical context for the daily coach prompt."""
        return self.context.build_coach_context(player_id, trading_date)

    def get_cross_run_summary(self) -> str:
        """Get cross-run comparison summary."""
        return self.context.build_cross_run_summary()

    def get_best_strategies(self, top_n: int = 3) -> List[Dict]:
        """Get the top N best strategy configurations by Sharpe."""
        results = []
        for pid in [f"PLAYER_{i}" for i in range(1, 6)]:
            best = self.db.get_best_strategy_snapshot(pid)
            if best:
                results.append(best)
        results.sort(key=lambda x: x.get("sharpe", 0), reverse=True)
        return results[:top_n]

    def get_run_count(self) -> int:
        """Get the total number of completed runs in memory."""
        return self.db.get_run_count()
