"""
Learning Log System for AQTIS

Tracks how the algorithm improves over time by analyzing:
1. Strategy performance evolution
2. Indicator effectiveness by market regime
3. Model prediction accuracy trends
4. Coach intervention effectiveness
5. Cross-run learning metrics

This provides visibility into what the system is learning and where it's improving.
"""

import os
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LearningEpoch:
    """Represents a single learning epoch (usually one day or one backtest run)."""
    epoch_id: int
    timestamp: datetime
    epoch_type: str  # 'backtest', 'paper_trade', 'live_trade'

    # Strategy metrics
    strategy_name: str
    strategy_version: int

    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    net_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0

    # Learning metrics
    top_indicators: List[str] = field(default_factory=list)
    worst_indicators: List[str] = field(default_factory=list)
    market_regime: str = "unknown"
    regime_accuracy: float = 0.0

    # Improvement tracking
    pnl_improvement: float = 0.0  # vs previous epoch
    sharpe_improvement: float = 0.0
    win_rate_improvement: float = 0.0

    # Coach interaction
    coach_applied: bool = False
    coach_changes: Dict[str, Any] = field(default_factory=dict)

    # Additional context
    symbols_traded: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class IndicatorLearning:
    """Tracks learning about a specific indicator."""
    indicator_name: str
    category: str  # momentum, trend, volatility, volume

    # Performance by regime
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # {regime: {win_rate, avg_pnl, signal_accuracy, sample_size}}

    # Time-series performance
    rolling_win_rate: List[float] = field(default_factory=list)
    rolling_signal_accuracy: List[float] = field(default_factory=list)

    # Optimal settings discovered
    best_weight: float = 0.0
    best_threshold: float = 0.0

    # Learning timestamps
    first_seen: datetime = None
    last_updated: datetime = None
    total_signals: int = 0


@dataclass
class StrategyLearning:
    """Tracks learning about a specific strategy."""
    strategy_id: str
    strategy_name: str

    # Version history with metrics
    version_history: List[Dict[str, Any]] = field(default_factory=list)
    # [{version, timestamp, win_rate, sharpe, changes_made}]

    # Performance trend
    pnl_history: List[float] = field(default_factory=list)
    sharpe_history: List[float] = field(default_factory=list)
    win_rate_history: List[float] = field(default_factory=list)

    # Best configuration found
    best_config: Dict[str, Any] = field(default_factory=dict)
    best_sharpe: float = 0.0
    best_win_rate: float = 0.0

    # Learning insights
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)


class LearningLog:
    """
    Central learning log that tracks algorithm improvement over time.

    Features:
    - Records each learning epoch (backtest, paper trade, live trade)
    - Tracks indicator effectiveness by market regime
    - Monitors strategy evolution and improvement
    - Provides visualizations of learning progress
    - Persists all data to SQLite for historical analysis

    Usage:
        log = LearningLog(db_path="./learning_log.db")

        # Record a learning epoch
        log.record_epoch(
            epoch_type="backtest",
            strategy_name="SuperIndicator_v1",
            metrics={...}
        )

        # Get learning summary
        summary = log.get_learning_summary(days=30)

        # View improvement trends
        trends = log.get_improvement_trends()
    """

    def __init__(self, db_path: str = "./learning_log.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self._init_database()
        logger.info(f"LearningLog initialized at {self.db_path}")

    def _init_database(self) -> None:
        """Initialize SQLite database with learning tables."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Learning epochs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_epochs (
                    epoch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    epoch_type TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    strategy_version INTEGER DEFAULT 1,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    net_pnl REAL DEFAULT 0,
                    sharpe_ratio REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    profit_factor REAL DEFAULT 0,
                    market_regime TEXT DEFAULT 'unknown',
                    regime_accuracy REAL DEFAULT 0,
                    pnl_improvement REAL DEFAULT 0,
                    sharpe_improvement REAL DEFAULT 0,
                    win_rate_improvement REAL DEFAULT 0,
                    coach_applied INTEGER DEFAULT 0,
                    coach_changes TEXT DEFAULT '{}',
                    top_indicators TEXT DEFAULT '[]',
                    worst_indicators TEXT DEFAULT '[]',
                    symbols_traded TEXT DEFAULT '[]',
                    notes TEXT DEFAULT ''
                )
            """)

            # Indicator learning table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indicator_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    market_regime TEXT NOT NULL,
                    epoch_id INTEGER,
                    timestamp TEXT NOT NULL,
                    win_rate REAL DEFAULT 0,
                    avg_pnl REAL DEFAULT 0,
                    signal_accuracy REAL DEFAULT 0,
                    signal_count INTEGER DEFAULT 0,
                    weight_used REAL DEFAULT 0,
                    FOREIGN KEY (epoch_id) REFERENCES learning_epochs(epoch_id)
                )
            """)

            # Strategy learning table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    config TEXT DEFAULT '{}',
                    win_rate REAL DEFAULT 0,
                    sharpe_ratio REAL DEFAULT 0,
                    net_pnl REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    changes_made TEXT DEFAULT '',
                    is_improvement INTEGER DEFAULT 0
                )
            """)

            # Learning insights table (human-readable learnings)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    evidence TEXT DEFAULT '{}',
                    actionable INTEGER DEFAULT 0,
                    applied INTEGER DEFAULT 0
                )
            """)

            # Model accuracy tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_accuracy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    predicted_value REAL,
                    actual_value REAL,
                    error REAL,
                    confidence REAL,
                    market_regime TEXT
                )
            """)

            # Create indexes for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_epochs_timestamp ON learning_epochs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_epochs_strategy ON learning_epochs(strategy_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_name ON indicator_learning(indicator_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_regime ON indicator_learning(market_regime)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_name ON strategy_learning(strategy_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_type ON learning_insights(insight_type)")

            conn.commit()

    def record_epoch(
        self,
        epoch_type: str,
        strategy_name: str,
        strategy_version: int = 1,
        total_trades: int = 0,
        winning_trades: int = 0,
        net_pnl: float = 0.0,
        sharpe_ratio: float = 0.0,
        max_drawdown: float = 0.0,
        profit_factor: float = 0.0,
        market_regime: str = "unknown",
        regime_accuracy: float = 0.0,
        coach_applied: bool = False,
        coach_changes: Dict[str, Any] = None,
        top_indicators: List[str] = None,
        worst_indicators: List[str] = None,
        symbols_traded: List[str] = None,
        indicator_performance: Dict[str, Dict[str, float]] = None,
        notes: str = "",
    ) -> int:
        """
        Record a learning epoch (backtest run, paper trade day, etc.)

        Returns the epoch_id for reference.
        """
        with self._lock:
            # Calculate improvements vs previous epoch
            prev_epoch = self._get_previous_epoch(strategy_name)

            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            pnl_improvement = net_pnl - prev_epoch.get("net_pnl", 0) if prev_epoch else 0
            sharpe_improvement = sharpe_ratio - prev_epoch.get("sharpe_ratio", 0) if prev_epoch else 0
            win_rate_improvement = win_rate - prev_epoch.get("win_rate", 0) if prev_epoch else 0

            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO learning_epochs (
                        timestamp, epoch_type, strategy_name, strategy_version,
                        total_trades, winning_trades, win_rate, net_pnl,
                        sharpe_ratio, max_drawdown, profit_factor,
                        market_regime, regime_accuracy,
                        pnl_improvement, sharpe_improvement, win_rate_improvement,
                        coach_applied, coach_changes,
                        top_indicators, worst_indicators, symbols_traded, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    epoch_type,
                    strategy_name,
                    strategy_version,
                    total_trades,
                    winning_trades,
                    win_rate,
                    net_pnl,
                    sharpe_ratio,
                    max_drawdown,
                    profit_factor,
                    market_regime,
                    regime_accuracy,
                    pnl_improvement,
                    sharpe_improvement,
                    win_rate_improvement,
                    1 if coach_applied else 0,
                    json.dumps(coach_changes or {}),
                    json.dumps(top_indicators or []),
                    json.dumps(worst_indicators or []),
                    json.dumps(symbols_traded or []),
                    notes,
                ))

                epoch_id = cursor.lastrowid

                # Record indicator performance for this epoch
                if indicator_performance:
                    for indicator, perf in indicator_performance.items():
                        cursor.execute("""
                            INSERT INTO indicator_learning (
                                indicator_name, category, market_regime, epoch_id,
                                timestamp, win_rate, avg_pnl, signal_accuracy,
                                signal_count, weight_used
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            indicator,
                            perf.get("category", "unknown"),
                            market_regime,
                            epoch_id,
                            datetime.now().isoformat(),
                            perf.get("win_rate", 0),
                            perf.get("avg_pnl", 0),
                            perf.get("signal_accuracy", 0),
                            perf.get("signal_count", 0),
                            perf.get("weight", 0),
                        ))

                conn.commit()

                # Generate insights if significant changes
                self._generate_insights(epoch_id, pnl_improvement, sharpe_improvement, win_rate_improvement)

                logger.info(f"Recorded learning epoch {epoch_id}: {strategy_name} - PnL: ${net_pnl:.2f}, Sharpe: {sharpe_ratio:.2f}")
                return epoch_id

    def record_strategy_version(
        self,
        strategy_id: str,
        strategy_name: str,
        version: int,
        config: Dict[str, Any],
        win_rate: float,
        sharpe_ratio: float,
        net_pnl: float,
        max_drawdown: float,
        changes_made: str = "",
    ) -> None:
        """Record a new strategy version with its configuration and results."""
        with self._lock:
            # Check if this is an improvement
            prev_version = self._get_previous_strategy_version(strategy_name)
            is_improvement = (
                sharpe_ratio > prev_version.get("sharpe_ratio", 0) and
                win_rate >= prev_version.get("win_rate", 0) * 0.95  # Allow 5% win rate drop if Sharpe improved
            ) if prev_version else True

            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO strategy_learning (
                        strategy_id, strategy_name, version, timestamp, config,
                        win_rate, sharpe_ratio, net_pnl, max_drawdown,
                        changes_made, is_improvement
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy_id,
                    strategy_name,
                    version,
                    datetime.now().isoformat(),
                    json.dumps(config),
                    win_rate,
                    sharpe_ratio,
                    net_pnl,
                    max_drawdown,
                    changes_made,
                    1 if is_improvement else 0,
                ))
                conn.commit()

    def record_prediction(
        self,
        model_name: str,
        prediction_type: str,
        predicted_value: float,
        actual_value: float,
        confidence: float = 0.5,
        market_regime: str = "unknown",
    ) -> None:
        """Record a model prediction and its actual outcome for accuracy tracking."""
        error = abs(predicted_value - actual_value)

        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO model_accuracy (
                        timestamp, model_name, prediction_type,
                        predicted_value, actual_value, error,
                        confidence, market_regime
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    model_name,
                    prediction_type,
                    predicted_value,
                    actual_value,
                    error,
                    confidence,
                    market_regime,
                ))
                conn.commit()

    def record_insight(
        self,
        insight_type: str,
        category: str,
        title: str,
        description: str,
        confidence: float = 0.5,
        evidence: Dict[str, Any] = None,
        actionable: bool = False,
    ) -> int:
        """
        Record a learning insight discovered by the system.

        insight_type: 'indicator_discovery', 'regime_pattern', 'strategy_weakness',
                      'optimization_opportunity', 'risk_pattern', 'correlation_found'
        """
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO learning_insights (
                        timestamp, insight_type, category, title,
                        description, confidence, evidence, actionable
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    insight_type,
                    category,
                    title,
                    description,
                    confidence,
                    json.dumps(evidence or {}),
                    1 if actionable else 0,
                ))
                conn.commit()
                return cursor.lastrowid

    def _get_previous_epoch(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get the most recent previous epoch for a strategy."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM learning_epochs
                WHERE strategy_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (strategy_name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def _get_previous_strategy_version(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get the most recent strategy version."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM strategy_learning
                WHERE strategy_name = ?
                ORDER BY version DESC
                LIMIT 1
            """, (strategy_name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def _generate_insights(
        self,
        epoch_id: int,
        pnl_improvement: float,
        sharpe_improvement: float,
        win_rate_improvement: float,
    ) -> None:
        """Generate automatic insights based on epoch results."""
        # Significant PnL improvement
        if pnl_improvement > 1000:  # $1000+ improvement
            self.record_insight(
                insight_type="performance_breakthrough",
                category="pnl",
                title=f"Significant PnL Improvement: ${pnl_improvement:.2f}",
                description=f"Strategy achieved ${pnl_improvement:.2f} more profit than previous epoch.",
                confidence=0.8,
                evidence={"epoch_id": epoch_id, "improvement": pnl_improvement},
                actionable=False,
            )

        # Significant Sharpe improvement
        if sharpe_improvement > 0.5:
            self.record_insight(
                insight_type="risk_adjusted_improvement",
                category="sharpe",
                title=f"Sharpe Ratio Improved by {sharpe_improvement:.2f}",
                description="Risk-adjusted returns have significantly improved.",
                confidence=0.8,
                evidence={"epoch_id": epoch_id, "improvement": sharpe_improvement},
                actionable=False,
            )

        # Performance degradation warning
        if pnl_improvement < -500 or sharpe_improvement < -0.3:
            self.record_insight(
                insight_type="performance_degradation",
                category="warning",
                title="Performance Degradation Detected",
                description=f"PnL dropped by ${abs(pnl_improvement):.2f}, Sharpe dropped by {abs(sharpe_improvement):.2f}",
                confidence=0.9,
                evidence={"epoch_id": epoch_id, "pnl_drop": pnl_improvement, "sharpe_drop": sharpe_improvement},
                actionable=True,
            )

    # ============================================================
    # QUERY METHODS
    # ============================================================

    def get_learning_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get a comprehensive summary of learning over the specified period.
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get epoch statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_epochs,
                    SUM(total_trades) as total_trades,
                    AVG(win_rate) as avg_win_rate,
                    AVG(sharpe_ratio) as avg_sharpe,
                    SUM(net_pnl) as total_pnl,
                    AVG(max_drawdown) as avg_drawdown,
                    SUM(CASE WHEN pnl_improvement > 0 THEN 1 ELSE 0 END) as improving_epochs,
                    SUM(CASE WHEN coach_applied = 1 THEN 1 ELSE 0 END) as coach_epochs
                FROM learning_epochs
                WHERE timestamp >= ?
            """, (cutoff,))
            epoch_stats = dict(cursor.fetchone())

            # Get best and worst performing strategies
            cursor.execute("""
                SELECT strategy_name, AVG(sharpe_ratio) as avg_sharpe, SUM(net_pnl) as total_pnl
                FROM learning_epochs
                WHERE timestamp >= ?
                GROUP BY strategy_name
                ORDER BY avg_sharpe DESC
                LIMIT 5
            """, (cutoff,))
            top_strategies = [dict(row) for row in cursor.fetchall()]

            # Get top performing indicators
            cursor.execute("""
                SELECT indicator_name, AVG(win_rate) as avg_win_rate, AVG(signal_accuracy) as avg_accuracy
                FROM indicator_learning
                WHERE timestamp >= ?
                GROUP BY indicator_name
                HAVING COUNT(*) >= 5
                ORDER BY avg_win_rate DESC
                LIMIT 10
            """, (cutoff,))
            top_indicators = [dict(row) for row in cursor.fetchall()]

            # Get worst performing indicators
            cursor.execute("""
                SELECT indicator_name, AVG(win_rate) as avg_win_rate, AVG(signal_accuracy) as avg_accuracy
                FROM indicator_learning
                WHERE timestamp >= ?
                GROUP BY indicator_name
                HAVING COUNT(*) >= 5
                ORDER BY avg_win_rate ASC
                LIMIT 5
            """, (cutoff,))
            worst_indicators = [dict(row) for row in cursor.fetchall()]

            # Get recent insights
            cursor.execute("""
                SELECT * FROM learning_insights
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 10
            """, (cutoff,))
            recent_insights = [dict(row) for row in cursor.fetchall()]

            # Calculate improvement rate
            improvement_rate = (
                epoch_stats["improving_epochs"] / epoch_stats["total_epochs"]
                if epoch_stats["total_epochs"] > 0 else 0
            )

            return {
                "period_days": days,
                "epoch_stats": epoch_stats,
                "improvement_rate": improvement_rate,
                "top_strategies": top_strategies,
                "top_indicators": top_indicators,
                "worst_indicators": worst_indicators,
                "recent_insights": recent_insights,
                "generated_at": datetime.now().isoformat(),
            }

    def get_improvement_trends(self, strategy_name: str = None, periods: int = 20) -> Dict[str, Any]:
        """
        Get improvement trends over time.

        Returns rolling averages and trend direction for key metrics.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT timestamp, win_rate, sharpe_ratio, net_pnl, max_drawdown,
                       pnl_improvement, sharpe_improvement, win_rate_improvement
                FROM learning_epochs
            """
            params = []
            if strategy_name:
                query += " WHERE strategy_name = ?"
                params.append(strategy_name)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(periods)

            cursor.execute(query, params)
            rows = [dict(row) for row in cursor.fetchall()]

            if not rows:
                return {"message": "No data available"}

            # Reverse to get chronological order
            rows = rows[::-1]

            # Calculate trends
            win_rates = [r["win_rate"] for r in rows]
            sharpes = [r["sharpe_ratio"] for r in rows]
            pnls = [r["net_pnl"] for r in rows]

            def calculate_trend(values):
                if len(values) < 2:
                    return "stable"
                # Simple linear regression slope
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                if slope > 0.01:
                    return "improving"
                elif slope < -0.01:
                    return "declining"
                return "stable"

            return {
                "periods_analyzed": len(rows),
                "win_rate": {
                    "current": win_rates[-1] if win_rates else 0,
                    "average": np.mean(win_rates) if win_rates else 0,
                    "trend": calculate_trend(win_rates),
                    "history": win_rates,
                },
                "sharpe_ratio": {
                    "current": sharpes[-1] if sharpes else 0,
                    "average": np.mean(sharpes) if sharpes else 0,
                    "trend": calculate_trend(sharpes),
                    "history": sharpes,
                },
                "pnl": {
                    "current": pnls[-1] if pnls else 0,
                    "total": sum(pnls),
                    "average": np.mean(pnls) if pnls else 0,
                    "trend": calculate_trend(pnls),
                    "history": pnls,
                },
                "timestamps": [r["timestamp"] for r in rows],
            }

    def get_indicator_analysis(self, indicator_name: str = None, market_regime: str = None) -> Dict[str, Any]:
        """
        Get detailed analysis of indicator performance.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Build query based on filters
            conditions = []
            params = []

            if indicator_name:
                conditions.append("indicator_name = ?")
                params.append(indicator_name)
            if market_regime:
                conditions.append("market_regime = ?")
                params.append(market_regime)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # Get indicator performance by regime
            cursor.execute(f"""
                SELECT
                    indicator_name,
                    market_regime,
                    COUNT(*) as sample_size,
                    AVG(win_rate) as avg_win_rate,
                    AVG(signal_accuracy) as avg_accuracy,
                    AVG(avg_pnl) as avg_pnl,
                    SUM(signal_count) as total_signals
                FROM indicator_learning
                WHERE {where_clause}
                GROUP BY indicator_name, market_regime
                ORDER BY avg_win_rate DESC
            """, params)

            results = [dict(row) for row in cursor.fetchall()]

            # Organize by indicator
            by_indicator = defaultdict(lambda: {"regimes": {}, "overall": {}})
            for row in results:
                ind = row["indicator_name"]
                regime = row["market_regime"]
                by_indicator[ind]["regimes"][regime] = {
                    "win_rate": row["avg_win_rate"],
                    "accuracy": row["avg_accuracy"],
                    "avg_pnl": row["avg_pnl"],
                    "signals": row["total_signals"],
                    "samples": row["sample_size"],
                }

            # Calculate overall stats per indicator
            for ind in by_indicator:
                regimes = by_indicator[ind]["regimes"]
                total_signals = sum(r["signals"] for r in regimes.values())
                if total_signals > 0:
                    weighted_win_rate = sum(
                        r["win_rate"] * r["signals"] for r in regimes.values()
                    ) / total_signals
                    by_indicator[ind]["overall"] = {
                        "weighted_win_rate": weighted_win_rate,
                        "total_signals": total_signals,
                        "regimes_analyzed": len(regimes),
                    }

            return dict(by_indicator)

    def get_model_accuracy_trends(self, model_name: str = None, days: int = 30) -> Dict[str, Any]:
        """Get model prediction accuracy trends over time."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT
                    model_name,
                    prediction_type,
                    AVG(error) as avg_error,
                    AVG(confidence) as avg_confidence,
                    COUNT(*) as predictions,
                    AVG(CASE WHEN error < 0.1 THEN 1 ELSE 0 END) as accuracy_rate
                FROM model_accuracy
                WHERE timestamp >= ?
            """
            params = [cutoff]

            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)

            query += " GROUP BY model_name, prediction_type"

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_coach_effectiveness(self) -> Dict[str, Any]:
        """Analyze the effectiveness of coach interventions."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get epochs with coach applied vs without
            cursor.execute("""
                SELECT
                    coach_applied,
                    COUNT(*) as epochs,
                    AVG(win_rate) as avg_win_rate,
                    AVG(sharpe_ratio) as avg_sharpe,
                    AVG(net_pnl) as avg_pnl,
                    AVG(pnl_improvement) as avg_improvement
                FROM learning_epochs
                GROUP BY coach_applied
            """)

            results = {row["coach_applied"]: dict(row) for row in cursor.fetchall()}

            with_coach = results.get(1, {})
            without_coach = results.get(0, {})

            return {
                "with_coach": with_coach,
                "without_coach": without_coach,
                "coach_impact": {
                    "win_rate_diff": with_coach.get("avg_win_rate", 0) - without_coach.get("avg_win_rate", 0),
                    "sharpe_diff": with_coach.get("avg_sharpe", 0) - without_coach.get("avg_sharpe", 0),
                    "pnl_diff": with_coach.get("avg_pnl", 0) - without_coach.get("avg_pnl", 0),
                },
                "conclusion": self._interpret_coach_effectiveness(with_coach, without_coach),
            }

    def _interpret_coach_effectiveness(self, with_coach: Dict, without_coach: Dict) -> str:
        """Generate human-readable interpretation of coach effectiveness."""
        if not with_coach or not without_coach:
            return "Insufficient data to evaluate coach effectiveness."

        sharpe_diff = with_coach.get("avg_sharpe", 0) - without_coach.get("avg_sharpe", 0)
        pnl_diff = with_coach.get("avg_pnl", 0) - without_coach.get("avg_pnl", 0)

        if sharpe_diff > 0.2 and pnl_diff > 0:
            return "Coach interventions are significantly improving performance."
        elif sharpe_diff > 0:
            return "Coach interventions show positive impact on risk-adjusted returns."
        elif sharpe_diff < -0.2:
            return "Coach interventions may need recalibration - performance declining."
        else:
            return "Coach impact is neutral - more data needed for conclusive analysis."

    # ============================================================
    # REPORTING METHODS
    # ============================================================

    def generate_learning_report(self, days: int = 30) -> str:
        """Generate a human-readable learning report."""
        summary = self.get_learning_summary(days)
        trends = self.get_improvement_trends(periods=min(days, 50))
        coach_eff = self.get_coach_effectiveness()

        report = []
        report.append("=" * 60)
        report.append(f"  AQTIS LEARNING REPORT - Last {days} Days")
        report.append("=" * 60)
        report.append(f"\nGenerated: {summary['generated_at']}")

        # Overview
        report.append("\n" + "-" * 40)
        report.append("  OVERVIEW")
        report.append("-" * 40)
        stats = summary["epoch_stats"]
        report.append(f"  Total Learning Epochs: {stats['total_epochs']}")
        report.append(f"  Total Trades Analyzed: {stats['total_trades']}")
        report.append(f"  Average Win Rate: {stats['avg_win_rate']*100:.1f}%")
        report.append(f"  Average Sharpe Ratio: {stats['avg_sharpe']:.2f}")
        report.append(f"  Total P&L: ${stats['total_pnl']:.2f}")
        report.append(f"  Improvement Rate: {summary['improvement_rate']*100:.1f}%")

        # Trends
        report.append("\n" + "-" * 40)
        report.append("  IMPROVEMENT TRENDS")
        report.append("-" * 40)
        report.append(f"  Win Rate Trend: {trends['win_rate']['trend'].upper()}")
        report.append(f"    Current: {trends['win_rate']['current']*100:.1f}%")
        report.append(f"    Average: {trends['win_rate']['average']*100:.1f}%")
        report.append(f"  Sharpe Ratio Trend: {trends['sharpe_ratio']['trend'].upper()}")
        report.append(f"    Current: {trends['sharpe_ratio']['current']:.2f}")
        report.append(f"    Average: {trends['sharpe_ratio']['average']:.2f}")
        report.append(f"  P&L Trend: {trends['pnl']['trend'].upper()}")
        report.append(f"    Total: ${trends['pnl']['total']:.2f}")

        # Top Strategies
        report.append("\n" + "-" * 40)
        report.append("  TOP PERFORMING STRATEGIES")
        report.append("-" * 40)
        for i, strat in enumerate(summary["top_strategies"][:5], 1):
            report.append(f"  {i}. {strat['strategy_name']}")
            report.append(f"     Sharpe: {strat['avg_sharpe']:.2f}, P&L: ${strat['total_pnl']:.2f}")

        # Top Indicators
        report.append("\n" + "-" * 40)
        report.append("  TOP PERFORMING INDICATORS")
        report.append("-" * 40)
        for i, ind in enumerate(summary["top_indicators"][:5], 1):
            report.append(f"  {i}. {ind['indicator_name']}")
            report.append(f"     Win Rate: {ind['avg_win_rate']*100:.1f}%, Accuracy: {ind['avg_accuracy']*100:.1f}%")

        # Indicators to Improve
        report.append("\n" + "-" * 40)
        report.append("  INDICATORS NEEDING IMPROVEMENT")
        report.append("-" * 40)
        for ind in summary["worst_indicators"][:3]:
            report.append(f"  - {ind['indicator_name']}: {ind['avg_win_rate']*100:.1f}% win rate")

        # Coach Effectiveness
        report.append("\n" + "-" * 40)
        report.append("  COACH EFFECTIVENESS")
        report.append("-" * 40)
        report.append(f"  {coach_eff['conclusion']}")
        impact = coach_eff.get("coach_impact", {})
        if impact:
            report.append(f"  Sharpe Impact: {impact.get('sharpe_diff', 0):+.2f}")
            report.append(f"  P&L Impact: ${impact.get('pnl_diff', 0):+.2f}")

        # Recent Insights
        report.append("\n" + "-" * 40)
        report.append("  RECENT LEARNING INSIGHTS")
        report.append("-" * 40)
        for insight in summary["recent_insights"][:5]:
            actionable = " [ACTIONABLE]" if insight["actionable"] else ""
            report.append(f"  - {insight['title']}{actionable}")
            report.append(f"    {insight['description'][:80]}...")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def export_to_dataframe(self, table: str = "learning_epochs", days: int = None) -> pd.DataFrame:
        """Export learning data to pandas DataFrame for analysis."""
        with sqlite3.connect(str(self.db_path)) as conn:
            query = f"SELECT * FROM {table}"
            if days:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                query += f" WHERE timestamp >= '{cutoff}'"
            return pd.read_sql_query(query, conn)

    def get_all_insights(self, actionable_only: bool = False) -> List[Dict[str, Any]]:
        """Get all learning insights, optionally filtered to actionable ones."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM learning_insights"
            if actionable_only:
                query += " WHERE actionable = 1"
            query += " ORDER BY timestamp DESC"

            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def mark_insight_applied(self, insight_id: int) -> None:
        """Mark an insight as having been applied."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE learning_insights SET applied = 1 WHERE id = ?",
                    (insight_id,)
                )
                conn.commit()
