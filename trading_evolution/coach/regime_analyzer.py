"""
Market Regime Analyzer module.

Detects market regimes and analyzes strategy performance by regime.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class RegimePerformance:
    """Performance metrics for a market regime."""
    regime: MarketRegime
    trade_count: int
    win_rate: float
    avg_profit: float
    total_profit: float
    sharpe_ratio: float
    long_win_rate: float
    short_win_rate: float
    pct_of_time: float  # Percentage of time in this regime


class RegimeAnalyzer:
    """
    Analyzes market regimes and strategy performance by regime.

    Regime Detection:
    - TRENDING_UP: ADX > 25, +DI > -DI, price above 50 SMA
    - TRENDING_DOWN: ADX > 25, -DI > +DI, price below 50 SMA
    - RANGING: ADX < 20, price oscillating around 50 SMA
    - VOLATILE: High ATR percentile (>80th), no clear trend
    """

    def __init__(self,
                 adx_trend_threshold: float = 25,
                 adx_range_threshold: float = 20,
                 volatility_percentile: float = 80):
        """
        Initialize regime analyzer.

        Args:
            adx_trend_threshold: ADX above this indicates trending
            adx_range_threshold: ADX below this indicates ranging
            volatility_percentile: ATR percentile above this is volatile
        """
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.volatility_percentile = volatility_percentile

    def detect_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market regime for each bar in DataFrame.

        Args:
            df: DataFrame with columns including ADX, DI+, DI-, ATR, SMA_50, close

        Returns:
            Series with MarketRegime values
        """
        regimes = pd.Series(MarketRegime.UNKNOWN.value, index=df.index)

        # Get required columns (case-insensitive search)
        cols = {c.lower(): c for c in df.columns}

        adx_col = None
        dip_col = None  # DI+
        din_col = None  # DI-
        atr_col = None
        sma_col = None
        close_col = cols.get('close', 'close')

        # Find ADX column
        for key in cols:
            if 'adx' in key and 'dm' not in key:
                adx_col = cols[key]
                break

        # Find DI columns
        for key in cols:
            if 'dmp' in key or 'di+' in key or key.endswith('_dip'):
                dip_col = cols[key]
            elif 'dmn' in key or 'di-' in key or key.endswith('_din'):
                din_col = cols[key]

        # Find ATR column
        for key in cols:
            if 'atr' in key:
                atr_col = cols[key]
                break

        # Find SMA column (prefer 50)
        for key in cols:
            if 'sma_50' in key or 'sma50' in key:
                sma_col = cols[key]
                break
        if not sma_col:
            for key in cols:
                if 'sma' in key:
                    sma_col = cols[key]
                    break

        # Calculate ATR percentile if we have ATR
        if atr_col and atr_col in df.columns:
            atr_percentile = df[atr_col].rolling(252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
                raw=False
            )
        else:
            atr_percentile = pd.Series(50, index=df.index)

        # Classify each bar
        for i in range(len(df)):
            idx = df.index[i]

            adx = df.loc[idx, adx_col] if adx_col and adx_col in df.columns else 20
            di_plus = df.loc[idx, dip_col] if dip_col and dip_col in df.columns else 25
            di_minus = df.loc[idx, din_col] if din_col and din_col in df.columns else 25
            close = df.loc[idx, close_col] if close_col in df.columns else 0
            sma = df.loc[idx, sma_col] if sma_col and sma_col in df.columns else close
            atr_pct = atr_percentile.iloc[i] if not pd.isna(atr_percentile.iloc[i]) else 50

            # Handle NaN
            if pd.isna(adx):
                adx = 20
            if pd.isna(di_plus):
                di_plus = 25
            if pd.isna(di_minus):
                di_minus = 25
            if pd.isna(sma):
                sma = close

            # Determine regime
            regime = self._classify_regime(
                adx, di_plus, di_minus, close, sma, atr_pct
            )
            regimes.iloc[i] = regime.value

        return regimes

    def _classify_regime(self,
                         adx: float,
                         di_plus: float,
                         di_minus: float,
                         close: float,
                         sma: float,
                         atr_percentile: float) -> MarketRegime:
        """Classify market regime based on indicators."""
        # High volatility takes precedence
        if atr_percentile > self.volatility_percentile:
            return MarketRegime.VOLATILE

        # Trending
        if adx > self.adx_trend_threshold:
            if di_plus > di_minus and close > sma:
                return MarketRegime.TRENDING_UP
            elif di_minus > di_plus and close < sma:
                return MarketRegime.TRENDING_DOWN

        # Ranging
        if adx < self.adx_range_threshold:
            return MarketRegime.RANGING

        return MarketRegime.UNKNOWN

    def get_regime_at(self,
                      df: pd.DataFrame,
                      timestamp: pd.Timestamp) -> MarketRegime:
        """Get regime at a specific timestamp."""
        regimes = self.detect_regime(df)
        if timestamp in regimes.index:
            return MarketRegime(regimes.loc[timestamp])
        return MarketRegime.UNKNOWN

    def analyze_performance_by_regime(self,
                                      trades: List[Dict],
                                      market_data: pd.DataFrame) -> List[RegimePerformance]:
        """
        Analyze strategy performance for each market regime.

        Args:
            trades: List of completed trades
            market_data: DataFrame with price and indicator data

        Returns:
            List of RegimePerformance for each regime
        """
        if not trades:
            return []

        # Detect regimes
        regimes = self.detect_regime(market_data)

        # Assign regime to each trade based on entry time
        trades_by_regime = {r: [] for r in MarketRegime}

        for trade in trades:
            entry_time = trade.get('entry_time')
            if entry_time and entry_time in regimes.index:
                regime = MarketRegime(regimes.loc[entry_time])
            else:
                regime = MarketRegime.UNKNOWN
            trades_by_regime[regime].append(trade)

        # Calculate regime time percentages
        regime_counts = regimes.value_counts()
        total_bars = len(regimes)

        # Calculate performance for each regime
        performances = []

        for regime in MarketRegime:
            regime_trades = trades_by_regime[regime]

            if not regime_trades:
                continue

            winners = [t for t in regime_trades if t.get('net_pnl', 0) > 0]
            longs = [t for t in regime_trades if t.get('direction') == 'LONG']
            shorts = [t for t in regime_trades if t.get('direction') == 'SHORT']
            long_winners = [t for t in longs if t.get('net_pnl', 0) > 0]
            short_winners = [t for t in shorts if t.get('net_pnl', 0) > 0]

            profits = [t.get('net_pnl', 0) for t in regime_trades]
            pnl_pcts = [t.get('net_pnl_pct', 0) for t in regime_trades]

            # Calculate Sharpe
            if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
                sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(252)
            else:
                sharpe = 0

            # Regime time percentage
            pct_time = regime_counts.get(regime.value, 0) / total_bars if total_bars > 0 else 0

            performances.append(RegimePerformance(
                regime=regime,
                trade_count=len(regime_trades),
                win_rate=len(winners) / len(regime_trades) if regime_trades else 0,
                avg_profit=np.mean(profits),
                total_profit=sum(profits),
                sharpe_ratio=sharpe,
                long_win_rate=len(long_winners) / len(longs) if longs else 0,
                short_win_rate=len(short_winners) / len(shorts) if shorts else 0,
                pct_of_time=pct_time
            ))

        return performances

    def get_regime_recommendations(self,
                                   performances: List[RegimePerformance]) -> Dict[str, str]:
        """
        Generate recommendations based on regime performance.

        Args:
            performances: List of RegimePerformance

        Returns:
            Dict of regime -> recommendation
        """
        recommendations = {}

        for perf in performances:
            if perf.trade_count < 10:
                recommendations[perf.regime.value] = "Insufficient data"
                continue

            rec_parts = []

            # Win rate assessment
            if perf.win_rate > 0.55:
                rec_parts.append("Strategy performs well")
            elif perf.win_rate < 0.45:
                rec_parts.append("Strategy struggles")

            # Long vs short
            if perf.long_win_rate > perf.short_win_rate + 0.1:
                rec_parts.append("favor long trades")
            elif perf.short_win_rate > perf.long_win_rate + 0.1:
                rec_parts.append("favor short trades")

            # Overall
            if perf.avg_profit > 0:
                rec_parts.append(f"avg profit ${perf.avg_profit:.2f}")
            else:
                rec_parts.append(f"avg loss ${abs(perf.avg_profit):.2f}")

            recommendations[perf.regime.value] = "; ".join(rec_parts) if rec_parts else "No specific recommendation"

        return recommendations
