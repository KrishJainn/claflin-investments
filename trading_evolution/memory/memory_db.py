"""
Persistent memory database for the 5-Player trading simulation.

Stores every trade, coach session, market snapshot, strategy config,
and indicator effectiveness data across all simulation runs.
"""

import json
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MemoryDB:
    """SQLite-backed persistent memory for trading simulation."""

    def __init__(self, db_path: str = "trading_memory.db"):
        self.db_path = Path(db_path)
        self._initialize_schema()

    @contextmanager
    def get_connection(self):
        """Thread-safe connection context manager with WAL mode."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self):
        """Create all tables and indexes if they don't exist."""
        with self.get_connection() as conn:
            conn.executescript("""
                -- 1. Simulation runs
                CREATE TABLE IF NOT EXISTS sim_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_number INTEGER,
                    session_id TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    days INTEGER,
                    symbols INTEGER,
                    team_pnl REAL,
                    team_return REAL,
                    config_json TEXT
                );

                -- 2. Every trade from every player
                CREATE TABLE IF NOT EXISTS player_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    player_id TEXT,
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    net_pnl REAL,
                    exit_reason TEXT,
                    si_at_entry REAL,
                    si_at_exit REAL,
                    atr_at_entry REAL,
                    holding_bars INTEGER,
                    market_regime TEXT,
                    indicator_snapshot TEXT,
                    FOREIGN KEY (run_id) REFERENCES sim_runs(run_id)
                );

                -- 3. Coach sessions
                CREATE TABLE IF NOT EXISTS coach_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    player_id TEXT,
                    trading_date TEXT,
                    regime TEXT,
                    advice TEXT,
                    weight_changes INTEGER,
                    entry_change REAL,
                    exit_change REAL,
                    mistakes_json TEXT,
                    weight_recs_json TEXT,
                    FOREIGN KEY (run_id) REFERENCES sim_runs(run_id)
                );

                -- 4. Daily market snapshots
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    trading_date TEXT,
                    advancers INTEGER,
                    decliners INTEGER,
                    avg_change_pct REAL,
                    market_bias TEXT,
                    top_movers_json TEXT,
                    nifty_regime TEXT,
                    FOREIGN KEY (run_id) REFERENCES sim_runs(run_id)
                );

                -- 5. Strategy snapshots at start/end of each run
                CREATE TABLE IF NOT EXISTS strategy_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    player_id TEXT,
                    snapshot_type TEXT,
                    label TEXT,
                    weights_json TEXT,
                    entry_threshold REAL,
                    exit_threshold REAL,
                    min_hold_bars INTEGER,
                    total_trades INTEGER,
                    win_rate REAL,
                    net_pnl REAL,
                    sharpe REAL,
                    FOREIGN KEY (run_id) REFERENCES sim_runs(run_id)
                );

                -- 6. Indicator-regime performance stats
                CREATE TABLE IF NOT EXISTS indicator_regime_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator TEXT,
                    regime TEXT,
                    total_trades INTEGER,
                    avg_pnl REAL,
                    win_rate REAL,
                    contribution_score REAL,
                    updated_at TEXT
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_trades_player
                    ON player_trades(player_id, run_id);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol
                    ON player_trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_regime
                    ON player_trades(market_regime);
                CREATE INDEX IF NOT EXISTS idx_coach_player
                    ON coach_sessions(player_id, run_id);
                CREATE INDEX IF NOT EXISTS idx_market_date
                    ON market_snapshots(trading_date);
                CREATE INDEX IF NOT EXISTS idx_strategy_player
                    ON strategy_snapshots(player_id, run_id);
                CREATE INDEX IF NOT EXISTS idx_ind_regime
                    ON indicator_regime_stats(indicator, regime);
            """)

    # ─────────────────────────────────────────────────────────────────
    # WRITE METHODS
    # ─────────────────────────────────────────────────────────────────

    def start_run(self, run_number: int, session_id: str,
                  days: int, symbols: int, config: Dict) -> int:
        """Record the start of a new simulation run. Returns run_id."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO sim_runs
                   (run_number, session_id, started_at, days, symbols, config_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_number, session_id, datetime.now().isoformat(),
                 days, symbols, json.dumps(config, default=str))
            )
            return cursor.lastrowid

    def end_run(self, run_id: int, team_pnl: float, team_return: float):
        """Record the completion of a simulation run."""
        with self.get_connection() as conn:
            conn.execute(
                """UPDATE sim_runs
                   SET completed_at = ?, team_pnl = ?, team_return = ?
                   WHERE run_id = ?""",
                (datetime.now().isoformat(), team_pnl, team_return, run_id)
            )

    def record_trade(self, run_id: int, player_id: str, trade: Dict,
                     market_regime: str = "unknown"):
        """Record a single trade."""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO player_trades
                   (run_id, player_id, symbol, direction, entry_price,
                    exit_price, entry_time, exit_time, net_pnl, exit_reason,
                    si_at_entry, si_at_exit, atr_at_entry, holding_bars,
                    market_regime, indicator_snapshot)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, player_id,
                 trade.get("symbol", ""),
                 trade.get("side", trade.get("direction", "")),
                 trade.get("price", trade.get("entry_price", 0)),
                 trade.get("exit_price", 0),
                 str(trade.get("timestamp", trade.get("entry_time", ""))),
                 str(trade.get("exit_time", "")),
                 trade.get("pnl", trade.get("net_pnl", 0)),
                 trade.get("exit_reason", ""),
                 trade.get("si_value", trade.get("si_at_entry", 0)),
                 trade.get("exit_si", trade.get("si_at_exit", 0)),
                 trade.get("atr", trade.get("atr_at_entry", 0)),
                 trade.get("holding_bars", 0),
                 market_regime,
                 json.dumps(trade.get("indicator_snapshot", {})))
            )

    def record_trades_batch(self, run_id: int, player_id: str,
                            trades: List[Dict], market_regime: str = "unknown"):
        """Record multiple trades at once (more efficient)."""
        with self.get_connection() as conn:
            rows = []
            for t in trades:
                rows.append((
                    run_id, player_id,
                    t.get("symbol", ""),
                    t.get("side", t.get("direction", "")),
                    t.get("price", t.get("entry_price", 0)),
                    t.get("exit_price", 0),
                    str(t.get("timestamp", t.get("entry_time", ""))),
                    str(t.get("exit_time", "")),
                    t.get("pnl", t.get("net_pnl", 0)),
                    t.get("exit_reason", ""),
                    t.get("si_value", t.get("si_at_entry", 0)),
                    t.get("exit_si", t.get("si_at_exit", 0)),
                    t.get("atr", t.get("atr_at_entry", 0)),
                    t.get("holding_bars", 0),
                    market_regime,
                    json.dumps(t.get("indicator_snapshot", {}))
                ))
            conn.executemany(
                """INSERT INTO player_trades
                   (run_id, player_id, symbol, direction, entry_price,
                    exit_price, entry_time, exit_time, net_pnl, exit_reason,
                    si_at_entry, si_at_exit, atr_at_entry, holding_bars,
                    market_regime, indicator_snapshot)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows
            )

    def record_coach_session(self, run_id: int, player_id: str,
                             trading_date: str, regime: str, advice: str,
                             weight_changes: int, entry_change: float,
                             exit_change: float, mistakes: List = None,
                             weight_recs: List = None):
        """Record a coach analysis + patch application."""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO coach_sessions
                   (run_id, player_id, trading_date, regime, advice,
                    weight_changes, entry_change, exit_change,
                    mistakes_json, weight_recs_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, player_id, trading_date, regime, advice,
                 weight_changes, entry_change, exit_change,
                 json.dumps(mistakes or []),
                 json.dumps(weight_recs or []))
            )

    def record_market_snapshot(self, run_id: int, trading_date: str,
                               advancers: int, decliners: int,
                               avg_change_pct: float, market_bias: str,
                               top_movers: List[Dict] = None,
                               nifty_regime: str = "unknown"):
        """Record daily market state."""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO market_snapshots
                   (run_id, trading_date, advancers, decliners,
                    avg_change_pct, market_bias, top_movers_json, nifty_regime)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, trading_date, advancers, decliners,
                 avg_change_pct, market_bias,
                 json.dumps(top_movers or []),
                 nifty_regime)
            )

    def record_strategy_snapshot(self, run_id: int, player_id: str,
                                  snapshot_type: str, label: str,
                                  weights: Dict, entry_threshold: float,
                                  exit_threshold: float, min_hold_bars: int,
                                  total_trades: int = 0, win_rate: float = 0,
                                  net_pnl: float = 0, sharpe: float = 0):
        """Record a player's strategy configuration snapshot."""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO strategy_snapshots
                   (run_id, player_id, snapshot_type, label, weights_json,
                    entry_threshold, exit_threshold, min_hold_bars,
                    total_trades, win_rate, net_pnl, sharpe)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, player_id, snapshot_type, label,
                 json.dumps(weights, default=str),
                 entry_threshold, exit_threshold, min_hold_bars,
                 total_trades, win_rate, net_pnl, sharpe)
            )

    def update_indicator_regime_stats(self, run_id: int = None):
        """Recalculate indicator-regime performance from trade data."""
        with self.get_connection() as conn:
            # Clear old stats
            conn.execute("DELETE FROM indicator_regime_stats")

            # Get all trades with indicator snapshots
            where = "WHERE run_id = ?" if run_id else ""
            params = (run_id,) if run_id else ()
            rows = conn.execute(
                f"""SELECT net_pnl, market_regime, indicator_snapshot
                    FROM player_trades {where}""",
                params
            ).fetchall()

            # Aggregate per indicator per regime
            stats = {}  # (indicator, regime) -> {pnls: [], wins: 0, total: 0}
            for row in rows:
                regime = row["market_regime"] or "unknown"
                pnl = row["net_pnl"] or 0
                try:
                    snapshot = json.loads(row["indicator_snapshot"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    continue

                for ind_name, ind_val in snapshot.items():
                    key = (ind_name, regime)
                    if key not in stats:
                        stats[key] = {"pnls": [], "wins": 0, "total": 0}
                    stats[key]["pnls"].append(pnl)
                    stats[key]["total"] += 1
                    if pnl > 0:
                        stats[key]["wins"] += 1

            # Insert aggregated stats
            now = datetime.now().isoformat()
            insert_rows = []
            for (ind, regime), s in stats.items():
                if s["total"] < 3:
                    continue
                avg_pnl = sum(s["pnls"]) / len(s["pnls"])
                win_rate = s["wins"] / s["total"] if s["total"] > 0 else 0
                # Contribution score: normalized avg_pnl
                contribution = avg_pnl / max(1, abs(avg_pnl)) if avg_pnl != 0 else 0
                insert_rows.append((
                    ind, regime, s["total"], round(avg_pnl, 2),
                    round(win_rate, 4), round(contribution, 4), now
                ))

            conn.executemany(
                """INSERT INTO indicator_regime_stats
                   (indicator, regime, total_trades, avg_pnl, win_rate,
                    contribution_score, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                insert_rows
            )
            logger.info(f"Updated {len(insert_rows)} indicator-regime stats")

    # ─────────────────────────────────────────────────────────────────
    # READ METHODS
    # ─────────────────────────────────────────────────────────────────

    def get_all_runs(self) -> List[Dict]:
        """Get all completed simulation runs."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM sim_runs
                   WHERE completed_at IS NOT NULL
                   ORDER BY run_id"""
            ).fetchall()
            return [dict(r) for r in rows]

    def get_run_count(self) -> int:
        """Get total number of completed runs."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM sim_runs WHERE completed_at IS NOT NULL"
            ).fetchone()
            return row["cnt"] if row else 0

    def get_player_trades(self, player_id: str,
                          run_id: int = None,
                          limit: int = None) -> List[Dict]:
        """Get trades for a player, optionally filtered by run."""
        with self.get_connection() as conn:
            query = "SELECT * FROM player_trades WHERE player_id = ?"
            params = [player_id]
            if run_id:
                query += " AND run_id = ?"
                params.append(run_id)
            query += " ORDER BY id DESC"
            if limit:
                query += f" LIMIT {limit}"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_player_symbol_stats(self, player_id: str) -> List[Dict]:
        """Get per-symbol performance for a player across all runs."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT symbol,
                          COUNT(*) as trades,
                          SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                          ROUND(AVG(net_pnl), 2) as avg_pnl,
                          ROUND(SUM(net_pnl), 2) as total_pnl
                   FROM player_trades
                   WHERE player_id = ?
                   GROUP BY symbol
                   ORDER BY total_pnl DESC""",
                (player_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_player_regime_stats(self, player_id: str) -> List[Dict]:
        """Get per-regime performance for a player across all runs."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT market_regime,
                          COUNT(*) as trades,
                          SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
                          ROUND(AVG(net_pnl), 2) as avg_pnl,
                          ROUND(SUM(net_pnl), 2) as total_pnl
                   FROM player_trades
                   WHERE player_id = ?
                   GROUP BY market_regime
                   ORDER BY total_pnl DESC""",
                (player_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_best_strategy_snapshot(self, player_id: str) -> Optional[Dict]:
        """Get the best-performing strategy config for a player."""
        with self.get_connection() as conn:
            row = conn.execute(
                """SELECT * FROM strategy_snapshots
                   WHERE player_id = ? AND snapshot_type = 'end'
                     AND total_trades > 0
                   ORDER BY sharpe DESC
                   LIMIT 1""",
                (player_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_recent_coach_sessions(self, player_id: str,
                                   limit: int = 5) -> List[Dict]:
        """Get the most recent coach sessions for a player."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM coach_sessions
                   WHERE player_id = ?
                   ORDER BY id DESC
                   LIMIT ?""",
                (player_id, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_market_snapshots(self, limit: int = 5) -> List[Dict]:
        """Get the most recent market snapshots."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM market_snapshots
                   ORDER BY id DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_indicator_regime_stats(self, regime: str = None) -> List[Dict]:
        """Get indicator performance stats, optionally filtered by regime."""
        with self.get_connection() as conn:
            if regime:
                rows = conn.execute(
                    """SELECT * FROM indicator_regime_stats
                       WHERE regime = ?
                       ORDER BY avg_pnl DESC""",
                    (regime,)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM indicator_regime_stats
                       ORDER BY avg_pnl DESC"""
                ).fetchall()
            return [dict(r) for r in rows]

    def get_cross_run_pnl_trend(self) -> List[Dict]:
        """Get team P&L trend across all runs."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT run_id, run_number, team_pnl, team_return
                   FROM sim_runs
                   WHERE completed_at IS NOT NULL
                   ORDER BY run_id"""
            ).fetchall()
            return [dict(r) for r in rows]

    def get_coach_advice_effectiveness(self, player_id: str) -> List[Dict]:
        """Check if coach patches helped or hurt by comparing P&L before/after."""
        with self.get_connection() as conn:
            sessions = conn.execute(
                """SELECT cs.id, cs.trading_date, cs.regime, cs.advice,
                          cs.weight_changes, cs.run_id
                   FROM coach_sessions cs
                   WHERE cs.player_id = ?
                   ORDER BY cs.id""",
                (player_id,)
            ).fetchall()

            results = []
            for sess in sessions:
                # Get P&L in the 3 days before and after this patch
                date = sess["trading_date"]
                run_id = sess["run_id"]

                before = conn.execute(
                    """SELECT COALESCE(SUM(net_pnl), 0) as pnl
                       FROM player_trades
                       WHERE player_id = ? AND run_id = ?
                         AND entry_time < ?""",
                    (player_id, run_id, date)
                ).fetchone()

                after = conn.execute(
                    """SELECT COALESCE(SUM(net_pnl), 0) as pnl
                       FROM player_trades
                       WHERE player_id = ? AND run_id = ?
                         AND entry_time >= ?""",
                    (player_id, run_id, date)
                ).fetchone()

                results.append({
                    "date": date,
                    "regime": sess["regime"],
                    "advice": (sess["advice"] or "")[:80],
                    "pnl_before": round(before["pnl"], 2) if before else 0,
                    "pnl_after": round(after["pnl"], 2) if after else 0,
                    "helped": (after["pnl"] if after else 0) > (before["pnl"] if before else 0),
                })
            return results
