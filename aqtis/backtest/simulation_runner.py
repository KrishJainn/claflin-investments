"""
AQTIS 60-Day Backtest Simulation Runner.

Token-efficient backtest runner that uses Yahoo Finance data and the
full AQTIS multi-agent stack. Designed to minimize LLM token usage by:

- Running rule-based signal generation and trade execution per bar
- Batching LLM analysis to end-of-day summaries (not per-trade)
- Using memory-based context instead of repeated LLM queries
- Only invoking LLM for daily coach diagnosis and end-of-run review

Architecture:
  1. Fetch 60 days of daily data for configured symbols
  2. For each trading day:
     a. Calculate technical indicators (rule-based, no tokens)
     b. Generate signals using strategy weights + indicator scores
     c. Execute trades through risk manager (rule-based)
     d. Record trades to memory
     e. End-of-day: batch post-mortem (1 LLM call per day, not per trade)
  3. End of run: strategy snapshot + cross-run summary
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add project root
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import the full 87-indicator system from trading_evolution
try:
    from trading_evolution.indicators.calculator import IndicatorCalculator
    from trading_evolution.indicators.normalizer import SignalAggregator
    HAS_FULL_INDICATORS = True
    logger.info("Full 87-indicator system loaded from trading_evolution")
except ImportError:
    HAS_FULL_INDICATORS = False
    logger.warning("trading_evolution indicators not available, using built-in basic indicators")


class SimulationRunner:
    """
    Token-efficient 60-day backtest runner.

    Token budget strategy:
    - Per-bar processing: 0 tokens (all rule-based)
    - Per-day summary: ~1 LLM call (~2000 tokens) for coach diagnosis
    - End-of-run review: ~1 LLM call (~3000 tokens)
    - Total for 60 days: ~62 LLM calls vs 1000+ if per-trade

    Args:
        memory: MemoryLayer instance.
        orchestrator: MultiAgentOrchestrator instance.
        config: AQTISConfig or dict with settings.
        data_provider: MarketDataProvider for Yahoo Finance.
        llm_budget: Max LLM calls allowed (default 80).
    """

    def __init__(
        self,
        memory,
        orchestrator,
        config,
        data_provider=None,
        llm_budget: int = 80,
    ):
        self.memory = memory
        self.orchestrator = orchestrator
        self.config = config
        self.llm_budget = llm_budget
        self._llm_calls_used = 0

        # Initialize data provider
        if data_provider:
            self.data_provider = data_provider
        else:
            from aqtis.data.market_data import MarketDataProvider
            cache_dir = getattr(config, "system", {})
            if hasattr(cache_dir, "data_dir"):
                cache_dir = str(cache_dir.data_dir)
            else:
                cache_dir = "data_cache"
            self.data_provider = MarketDataProvider(cache_dir=cache_dir)

        # Initialize full 87-indicator calculator if available
        self._indicator_calculator = None
        if HAS_FULL_INDICATORS:
            try:
                self._indicator_calculator = IndicatorCalculator()
                logger.info(
                    f"Loaded {self._indicator_calculator.universe.total_count} indicators"
                )
            except Exception as e:
                logger.warning(f"Failed to init IndicatorCalculator: {e}")

        # Default indicator weights (evolved from 5-player model best performers)
        # These combine momentum, trend, volatility, and volume categories
        self._default_weights = {
            # Momentum (strongest signals)
            "RSI_14": 0.35, "RSI_7": 0.15, "MACD_12_26_9": 0.30,
            "STOCH_14_3": 0.20, "STOCH_5_3": 0.15, "WILLR_14": 0.20,
            "CCI_20": 0.15, "MOM_10": 0.12, "CMO_14": 0.10,
            "TSI_13_25": 0.18, "UO_7_14_28": 0.12, "AO_5_34": 0.10,
            "ROC_10": 0.12, "KST": 0.10,
            # Trend
            "ADX_14": 0.25, "SUPERTREND_10_2": 0.20, "SUPERTREND_7_3": 0.18,
            "AROON_14": 0.12, "PSAR": 0.15, "ICHIMOKU": 0.10,
            "VORTEX_14": 0.10, "LINREG_SLOPE_14": 0.15,
            # Volatility
            "BBANDS_20_2": 0.20, "BBANDS_10_1.5": 0.12,
            "KC_20_2": 0.10, "ATR_14": 0.08, "NATR_14": 0.10,
            "ZSCORE_20": 0.15,
            # Volume
            "OBV": 0.12, "CMF_20": 0.15, "MFI_14": 0.18,
            "AD": 0.08, "EFI_13": 0.10,
            # Overlap (price-relative)
            "EMA_10": 0.08, "EMA_20": 0.10, "EMA_50": 0.08,
            "SMA_20": 0.06, "SMA_50": 0.08, "HMA_9": 0.10,
            "DEMA_20": 0.08, "TEMA_10": 0.10, "T3_5": 0.08,
        }

        # Simulation state
        self._run_id = None
        self._run_number = 0
        self._capital = 100_000.0
        self._positions: Dict[str, Dict] = {}  # symbol -> position
        self._daily_trades: List[Dict] = []
        self._all_trades: List[Dict] = []
        self._equity_curve: List[Dict] = []
        self._daily_pnl: List[float] = []

    def run(
        self,
        symbols: List[str] = None,
        days: int = 60,
        initial_capital: float = None,
        run_number: int = None,
    ) -> Dict[str, Any]:
        """
        Run a full simulation.

        Args:
            symbols: List of Yahoo Finance tickers. Defaults to config.
            days: Number of trading days to simulate.
            initial_capital: Starting capital. Defaults to config.
            run_number: Run number for cross-run tracking.

        Returns:
            Comprehensive results dict with trades, metrics, and insights.
        """
        # Configuration
        if symbols is None:
            if hasattr(self.config, "market_data"):
                symbols = self.config.market_data.symbols
            else:
                symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

        if initial_capital is not None:
            self._capital = initial_capital
        elif hasattr(self.config, "execution"):
            self._capital = self.config.execution.initial_capital

        starting_capital = self._capital

        # Determine run number
        pnl_trend = self.memory.get_cross_run_pnl_trend()
        if run_number is None:
            self._run_number = len(pnl_trend) + 1
        else:
            self._run_number = run_number

        logger.info(
            f"Starting simulation run #{self._run_number}: "
            f"{days} days, {len(symbols)} symbols, capital={self._capital:,.0f}"
        )

        # Register run in memory
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = self.orchestrator.start_simulation_run(
            run_number=self._run_number,
            session_id=session_id,
            days=days,
            symbols=len(symbols),
            config={
                "symbols": symbols,
                "initial_capital": self._capital,
                "llm_budget": self.llm_budget,
            },
        )

        # Fetch data
        logger.info(f"Fetching {days}+30 days of data for {len(symbols)} symbols...")
        market_data = self._fetch_data(symbols, days)

        if not market_data:
            return {"error": "No data fetched for any symbol"}

        fetched_symbols = list(market_data.keys())
        logger.info(f"Data fetched for {len(fetched_symbols)} symbols")

        # Normalize timezone-aware indexes to tz-naive for consistent matching
        for symbol in list(market_data.keys()):
            df = market_data[symbol]
            if df.index.tz is not None:
                market_data[symbol] = df.tz_localize(None)

        # Determine simulation date range from data
        all_dates = set()
        for df in market_data.values():
            all_dates.update(df.index.normalize().unique())
        sorted_dates = sorted(all_dates)

        if len(sorted_dates) < days + 20:
            logger.warning(
                f"Only {len(sorted_dates)} trading days available, "
                f"using last {min(days, len(sorted_dates) - 20)} for simulation"
            )

        # Reserve first 20 days for indicator warmup
        warmup = min(20, len(sorted_dates) // 4)
        sim_dates = sorted_dates[warmup : warmup + days]

        if not sim_dates:
            return {"error": "Insufficient data for simulation"}

        logger.info(
            f"Simulating {len(sim_dates)} days: "
            f"{sim_dates[0].strftime('%Y-%m-%d')} to {sim_dates[-1].strftime('%Y-%m-%d')}"
        )

        # Pre-compute indicators for all symbols (0 tokens)
        indicators = self._compute_indicators(market_data)

        # Main simulation loop
        for day_idx, sim_date in enumerate(sim_dates):
            day_str = sim_date.strftime("%Y-%m-%d")
            self._daily_trades = []

            # Process each symbol
            for symbol in fetched_symbols:
                df = market_data[symbol]
                ind = indicators.get(symbol)
                if ind is None:
                    continue

                # Get bar for this date
                day_mask = df.index.normalize() == sim_date
                if not day_mask.any():
                    continue

                bar_idx = df.index.get_indexer(df.index[day_mask])[-1]
                if bar_idx < warmup:
                    continue

                # Generate signal (rule-based, 0 tokens)
                signal = self._generate_signal(symbol, df, ind, bar_idx)

                # Check for exit on existing positions
                if symbol in self._positions:
                    self._check_exit(symbol, df, ind, bar_idx, day_str)

                # Check for entry if no position
                if symbol not in self._positions and signal.get("score", 0) != 0:
                    self._check_entry(symbol, df, ind, bar_idx, signal, day_str)

            # End-of-day processing
            day_pnl = self._mark_to_market(market_data, sim_date)
            self._daily_pnl.append(day_pnl)
            self._equity_curve.append({
                "date": day_str,
                "equity": self._capital + self._unrealized_pnl(market_data, sim_date),
                "realized_pnl": sum(self._daily_pnl),
                "positions": len(self._positions),
                "trades_today": len(self._daily_trades),
            })

            # Store today's trades to memory (0 tokens)
            for trade in self._daily_trades:
                self.memory.store_trade(trade)
                self._all_trades.append(trade)

            # End-of-day coach diagnosis (1 LLM call if budget allows)
            if self._daily_trades and self._can_use_llm():
                self._end_of_day_analysis(day_str, day_idx, len(sim_dates))

            # Progress logging
            if (day_idx + 1) % 10 == 0 or day_idx == len(sim_dates) - 1:
                cum_pnl = sum(self._daily_pnl)
                logger.info(
                    f"Day {day_idx + 1}/{len(sim_dates)}: "
                    f"P&L={cum_pnl:+,.0f}, Trades={len(self._all_trades)}, "
                    f"Positions={len(self._positions)}, "
                    f"LLM calls={self._llm_calls_used}/{self.llm_budget}"
                )

        # Close all remaining positions
        if self._positions and sim_dates:
            self._close_all_positions(market_data, sim_dates[-1])

        # Calculate final metrics
        metrics = self._calculate_metrics(starting_capital)

        # Record run completion
        total_pnl = sum(self._daily_pnl)
        total_return = (total_pnl / starting_capital) * 100
        self.orchestrator.end_simulation_run(
            self._run_id, total_pnl, total_return,
            notes=f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
                  f"WR={metrics.get('win_rate', 0):.1%}",
        )

        # Record strategy snapshot for best-config retrieval
        strategies = self.memory.get_active_strategies()
        for strategy in strategies:
            sid = strategy.get("strategy_id", "")
            if sid:
                self.orchestrator.record_strategy_snapshot(
                    strategy_id=sid,
                    run_number=self._run_number,
                    weights=strategy.get("parameters", {}),
                    sharpe=metrics.get("sharpe_ratio", 0),
                    win_rate=metrics.get("win_rate", 0) * 100,
                    net_pnl=total_pnl,
                )

        # End-of-run LLM review (1 call)
        run_review = None
        if self._can_use_llm() and self._all_trades:
            run_review = self._end_of_run_review(metrics)

        result = {
            "run_number": self._run_number,
            "run_id": self._run_id,
            "symbols": fetched_symbols,
            "days_simulated": len(sim_dates),
            "date_range": {
                "start": sim_dates[0].strftime("%Y-%m-%d"),
                "end": sim_dates[-1].strftime("%Y-%m-%d"),
            },
            "capital": {
                "initial": starting_capital,
                "final": starting_capital + total_pnl,
                "pnl": total_pnl,
                "return_pct": total_return,
            },
            "metrics": metrics,
            "trade_count": len(self._all_trades),
            "equity_curve": self._equity_curve,
            "llm_usage": {
                "calls_used": self._llm_calls_used,
                "budget": self.llm_budget,
                "remaining": self.llm_budget - self._llm_calls_used,
            },
            "run_review": run_review,
        }

        logger.info(
            f"Simulation complete: P&L={total_pnl:+,.0f} ({total_return:+.2f}%), "
            f"Trades={len(self._all_trades)}, Sharpe={metrics.get('sharpe_ratio', 0):.2f}"
        )

        return result

    # ─────────────────────────────────────────────────────────────────
    # DATA FETCHING
    # ─────────────────────────────────────────────────────────────────

    def _fetch_data(self, symbols: List[str], days: int) -> Dict[str, pd.DataFrame]:
        """Fetch historical data with buffer for indicator warmup."""
        total_days = days + 30  # Extra for indicator calculation warmup
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=int(total_days * 1.5))).strftime("%Y-%m-%d")

        result = {}
        for symbol in symbols:
            try:
                df = self.data_provider.get_historical(
                    symbol, start_date=start_date, end_date=end_date
                )
                if df is not None and len(df) >= 40:
                    result[symbol] = df
                else:
                    logger.warning(f"Insufficient data for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")

        return result

    # ─────────────────────────────────────────────────────────────────
    # INDICATOR COMPUTATION (0 tokens)
    # ─────────────────────────────────────────────────────────────────

    def _compute_indicators(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Compute technical indicators for all symbols. No LLM calls.

        If the full 87-indicator system is available (from trading_evolution),
        uses that with proper normalization. Otherwise falls back to built-in.
        """
        indicators = {}

        if self._indicator_calculator:
            logger.info("Computing 87 indicators using full Super Indicator system...")
            for symbol, df in market_data.items():
                try:
                    # Calculate all 87 indicators with normalization
                    raw = self._indicator_calculator.calculate_all(df, normalize=True)
                    # Rename to DNA names for weight matching
                    renamed = self._indicator_calculator.rename_to_dna_names(raw)
                    indicators[symbol] = renamed
                except Exception as e:
                    logger.warning(f"Full indicator calc failed for {symbol}: {e}")
                    # Fallback to basic
                    try:
                        indicators[symbol] = self._calc_indicators(df)
                    except Exception:
                        pass
            logger.info(
                f"Computed indicators for {len(indicators)} symbols "
                f"({indicators[next(iter(indicators))].shape[1]} indicators each)"
                if indicators else "No indicators computed"
            )
        else:
            logger.info("Using built-in basic indicators (7 indicators)...")
            for symbol, df in market_data.items():
                try:
                    ind = self._calc_indicators(df)
                    indicators[symbol] = ind
                except Exception as e:
                    logger.warning(f"Indicator calc failed for {symbol}: {e}")

        return indicators

    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate core technical indicators from OHLCV data."""
        ind = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)

        # Moving averages
        ind["sma_10"] = close.rolling(10).mean()
        ind["sma_20"] = close.rolling(20).mean()
        ind["sma_50"] = close.rolling(50).mean()
        ind["ema_12"] = close.ewm(span=12).mean()
        ind["ema_26"] = close.ewm(span=26).mean()

        # MACD
        ind["macd"] = ind["ema_12"] - ind["ema_26"]
        ind["macd_signal"] = ind["macd"].ewm(span=9).mean()
        ind["macd_hist"] = ind["macd"] - ind["macd_signal"]

        # RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        ind["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        ind["bb_upper"] = bb_sma + 2 * bb_std
        ind["bb_lower"] = bb_sma - 2 * bb_std
        ind["bb_pct"] = (close - ind["bb_lower"]) / (ind["bb_upper"] - ind["bb_lower"] + 1e-10)
        ind["bb_width"] = (ind["bb_upper"] - ind["bb_lower"]) / (bb_sma + 1e-10)

        # ATR (14)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(14).mean()

        # Volume indicators
        ind["volume_sma"] = volume.rolling(20).mean()
        ind["volume_ratio"] = volume / (ind["volume_sma"] + 1e-10)

        # Price momentum
        ind["returns_1d"] = close.pct_change()
        ind["returns_5d"] = close.pct_change(5)
        ind["returns_20d"] = close.pct_change(20)

        # Volatility
        ind["volatility_20d"] = ind["returns_1d"].rolling(20).std() * np.sqrt(252)

        # Trend strength (ADX simplified)
        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        ind["plus_di"] = 100 * (plus_dm.rolling(14).mean() / (ind["atr"] + 1e-10))
        ind["minus_di"] = 100 * (minus_dm.rolling(14).mean() / (ind["atr"] + 1e-10))
        dx = 100 * (ind["plus_di"] - ind["minus_di"]).abs() / (ind["plus_di"] + ind["minus_di"] + 1e-10)
        ind["adx"] = dx.rolling(14).mean()

        # SuperTrend (10, 3)
        hl2 = (high + low) / 2
        atr_st = tr.rolling(10).mean()
        upper_band = hl2 + 3 * atr_st
        lower_band = hl2 - 3 * atr_st
        supertrend = pd.Series(0.0, index=df.index)
        direction = pd.Series(1, index=df.index)
        for i in range(1, len(df)):
            if close.iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]
            supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]
        ind["supertrend"] = supertrend
        ind["supertrend_dir"] = direction

        return ind

    # ─────────────────────────────────────────────────────────────────
    # SIGNAL GENERATION (0 tokens)
    # ─────────────────────────────────────────────────────────────────

    def _generate_signal(
        self, symbol: str, df: pd.DataFrame, ind: pd.DataFrame, bar_idx: int,
    ) -> Dict:
        """
        Signal generation using the full 87-indicator Super Indicator system.

        When the full indicator calculator is available, uses all 87
        pre-normalized indicators with evolved weights from the 5-player
        model. Falls back to basic 7-indicator system otherwise.

        Architecture:
        - All 87 indicators already normalized to [-1, 1]
        - Weights from _default_weights (evolved best performers)
        - SignalAggregator produces final [-1, 1] score via tanh squashing
        - Regime classification for context
        """
        if bar_idx >= len(ind):
            return {"score": 0, "signals": {}}

        row = ind.iloc[bar_idx]

        # Full 87-indicator path
        if self._indicator_calculator and HAS_FULL_INDICATORS:
            return self._generate_signal_full(symbol, df, ind, bar_idx, row)

        # Fallback: basic 7-indicator path
        return self._generate_signal_basic(symbol, df, ind, bar_idx, row)

    def _generate_signal_full(
        self, symbol: str, df: pd.DataFrame, ind: pd.DataFrame,
        bar_idx: int, row,
    ) -> Dict:
        """
        Full 87-indicator signal generation using Super Indicator weights.

        The indicators in `ind` are already normalized to [-1, 1] by the
        IndicatorCalculator. We apply weights and aggregate.
        """
        # Collect available indicator values at this bar
        signals = {}
        for col in ind.columns:
            val = row.get(col, np.nan)
            if pd.notna(val) and not np.isnan(val):
                signals[col] = float(val)

        if not signals:
            return {"score": 0, "signals": {}}

        # Use SignalAggregator with evolved weights
        # Build a 1-row DataFrame for the aggregator
        row_df = ind.iloc[bar_idx:bar_idx + 1]
        score = SignalAggregator.weighted_at_timestamp(
            row_df, self._default_weights, row_df.index[0]
        )

        # Get the close price and ATR for position sizing
        close_price = float(df["close"].iloc[bar_idx])

        # ATR from indicators (look for ATR_14 column)
        atr = 0.0
        for atr_col in ["ATR_14", "ATR_20"]:
            if atr_col in ind.columns:
                atr_val = ind[atr_col].iloc[bar_idx]
                if pd.notna(atr_val):
                    # ATR is normalized; denormalize for stop-loss calc
                    atr = close_price * 0.02  # Approximate 2% ATR
                    break
        if atr == 0:
            atr = close_price * 0.02

        # RSI for context
        rsi = 50.0
        for rsi_col in ["RSI_14", "RSI_7"]:
            if rsi_col in signals:
                # De-normalize: RSI normalized from [0,100] to [-1,1]
                rsi = (signals[rsi_col] + 1) * 50  # Map [-1,1] -> [0,100]
                break

        # Get top contributing indicators for logging/memory
        top_signals = sorted(
            [(k, v * self._default_weights.get(k, 0))
             for k, v in signals.items()
             if k in self._default_weights],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]

        return {
            "score": float(np.clip(score, -1, 1)),
            "signals": {k: v for k, v in top_signals},
            "all_signals": signals,
            "symbol": symbol,
            "close": close_price,
            "atr": atr,
            "rsi": rsi,
            "regime": self._classify_regime_from_indicators(signals),
            "num_indicators": len(signals),
        }

    def _classify_regime_from_indicators(self, signals: Dict) -> str:
        """Classify regime from normalized indicator values."""
        # ADX for trend strength
        adx = signals.get("ADX_14", 0)
        # SuperTrend for direction
        st = signals.get("SUPERTREND_10_2", signals.get("SUPERTREND_7_3", 0))
        # Volatility (NATR or ATR)
        vol = signals.get("NATR_14", signals.get("ZSCORE_20", 0))
        # Bollinger width
        bb = signals.get("BBANDS_20_2", 0)

        if abs(vol) > 0.6:
            return "high_vol"
        elif abs(vol) < 0.2 and abs(adx) < 0.2:
            return "low_vol"
        elif adx > 0.3 and st > 0:
            return "trending_up"
        elif adx > 0.3 and st < 0:
            return "trending_down"
        else:
            return "mean_reverting"

    def _generate_signal_basic(
        self, symbol: str, df: pd.DataFrame, ind: pd.DataFrame,
        bar_idx: int, row,
    ) -> Dict:
        """Fallback basic 7-indicator signal generation."""
        signals = {}
        weights = {}

        rsi = row.get("rsi", 50)
        if not np.isnan(rsi):
            if rsi < 30: signals["rsi"] = 1.0
            elif rsi < 40: signals["rsi"] = 0.5
            elif rsi > 70: signals["rsi"] = -1.0
            elif rsi > 60: signals["rsi"] = -0.5
            else: signals["rsi"] = 0.0
            weights["rsi"] = 0.20

        macd_hist = row.get("macd_hist", 0)
        if not np.isnan(macd_hist):
            price = df["close"].iloc[bar_idx]
            signals["macd"] = np.clip(macd_hist / (price * 0.01 + 1e-10), -1, 1)
            weights["macd"] = 0.20

        bb_pct = row.get("bb_pct", 0.5)
        if not np.isnan(bb_pct):
            if bb_pct < 0.05: signals["bollinger"] = 1.0
            elif bb_pct < 0.2: signals["bollinger"] = 0.6
            elif bb_pct > 0.95: signals["bollinger"] = -1.0
            elif bb_pct > 0.8: signals["bollinger"] = -0.6
            else: signals["bollinger"] = 0.0
            weights["bollinger"] = 0.15

        st_dir = row.get("supertrend_dir", 0)
        if not np.isnan(st_dir):
            signals["supertrend"] = float(st_dir)
            weights["supertrend"] = 0.15

        adx = row.get("adx", 0)
        plus_di = row.get("plus_di", 0)
        if not np.isnan(adx) and not np.isnan(plus_di):
            signals["adx_trend"] = (1.0 if plus_di > row.get("minus_di", 0) else -1.0) if adx > 25 else 0.0
            weights["adx_trend"] = 0.10

        vol_ratio = row.get("volume_ratio", 1)
        if not np.isnan(vol_ratio):
            signals["volume"] = min(vol_ratio - 1.0, 0.5) * 0.4 if vol_ratio > 1.5 else 0.0
            weights["volume"] = 0.10

        ret_5d = row.get("returns_5d", 0)
        if not np.isnan(ret_5d):
            signals["momentum_5d"] = np.clip(ret_5d * 10, -1, 1)
            weights["momentum_5d"] = 0.10

        if not signals:
            return {"score": 0, "signals": {}}

        weighted_sum = sum(signals[k] * weights.get(k, 0.1) for k in signals)
        total_weight = sum(weights.get(k, 0.1) for k in signals)
        score = weighted_sum / (total_weight + 1e-10)

        return {
            "score": np.clip(score, -1, 1),
            "signals": signals,
            "symbol": symbol,
            "close": float(df["close"].iloc[bar_idx]),
            "atr": float(row.get("atr", 0)) if not np.isnan(row.get("atr", 0)) else 0,
            "rsi": float(rsi) if not np.isnan(rsi) else 50,
            "regime": self._classify_regime(row),
        }

    def _classify_regime(self, row) -> str:
        """Simple rule-based regime classification (0 tokens)."""
        vol = row.get("volatility_20d", 0)
        adx = row.get("adx", 0)
        ret_20d = row.get("returns_20d", 0)

        if np.isnan(vol) or np.isnan(adx):
            return "unknown"

        if vol > 0.4:
            return "high_vol"
        elif vol < 0.15:
            return "low_vol"
        elif adx > 25 and ret_20d > 0.02:
            return "trending_up"
        elif adx > 25 and ret_20d < -0.02:
            return "trending_down"
        else:
            return "mean_reverting"

    # ─────────────────────────────────────────────────────────────────
    # TRADE EXECUTION (0 tokens)
    # ─────────────────────────────────────────────────────────────────

    def _check_entry(
        self, symbol: str, df: pd.DataFrame, ind: pd.DataFrame,
        bar_idx: int, signal: Dict, day_str: str,
    ):
        """Check if we should enter a position. Rule-based, no LLM."""
        score = signal.get("score", 0)
        threshold = 0.15  # Minimum signal strength

        if abs(score) < threshold:
            return

        # Position sizing (rule-based Kelly)
        confidence = min(abs(score), 0.8)
        position_frac = confidence * 0.25 * 0.1  # Quarter Kelly, 10% max
        position_value = self._capital * position_frac
        entry_price = signal.get("close", 0)

        if entry_price <= 0 or position_value < 100:
            return

        # Max positions check
        if len(self._positions) >= 5:
            return

        shares = int(position_value / entry_price)
        if shares < 1:
            return

        action = "BUY" if score > 0 else "SELL"
        atr = signal.get("atr", entry_price * 0.02)
        stop_loss = entry_price - 2 * atr if action == "BUY" else entry_price + 2 * atr
        take_profit = entry_price + 4 * atr if action == "BUY" else entry_price - 4 * atr

        position = {
            "symbol": symbol,
            "action": action,
            "entry_price": entry_price,
            "shares": shares,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_date": day_str,
            "entry_bar": bar_idx,
            "signal_score": score,
            "signals": signal.get("signals", {}),
            "regime": signal.get("regime", "unknown"),
        }

        self._positions[symbol] = position

        trade = {
            "asset": symbol,
            "action": action,
            "strategy_id": "aqtis_multi_indicator",
            "entry_price": entry_price,
            "position_size": shares,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "market_regime": signal.get("regime", "unknown"),
            "confidence": confidence,
            "timestamp": day_str,
            "signals": signal.get("signals", {}),
            "indicator_signals": signal.get("signals", {}),
        }
        self._daily_trades.append(trade)

    def _check_exit(
        self, symbol: str, df: pd.DataFrame, ind: pd.DataFrame,
        bar_idx: int, day_str: str,
    ):
        """Check if we should exit a position. Rule-based, no LLM."""
        pos = self._positions.get(symbol)
        if not pos:
            return

        current_price = float(df["close"].iloc[bar_idx])
        entry_price = pos["entry_price"]
        action = pos["action"]

        # Check stop loss
        exit_reason = None
        if action == "BUY" and current_price <= pos["stop_loss"]:
            exit_reason = "stop_loss"
        elif action == "SELL" and current_price >= pos["stop_loss"]:
            exit_reason = "stop_loss"
        elif action == "BUY" and current_price >= pos["take_profit"]:
            exit_reason = "take_profit"
        elif action == "SELL" and current_price <= pos["take_profit"]:
            exit_reason = "take_profit"

        # Time-based exit: max 5 days holding
        entry_date = pd.Timestamp(pos["entry_date"])
        if (pd.Timestamp(day_str) - entry_date).days >= 5:
            exit_reason = "max_hold"

        if exit_reason is None:
            return

        # Close position
        self._close_position(symbol, current_price, day_str, exit_reason)

    def _close_position(self, symbol: str, exit_price: float, day_str: str, reason: str):
        """Close a position and record the trade."""
        pos = self._positions.pop(symbol, None)
        if not pos:
            return

        entry_price = pos["entry_price"]
        shares = pos["shares"]
        action = pos["action"]

        if action == "BUY":
            pnl = (exit_price - entry_price) * shares
        else:
            pnl = (entry_price - exit_price) * shares

        pnl_pct = pnl / (entry_price * shares) * 100

        # Apply transaction costs (~0.1% round trip)
        cost = entry_price * shares * 0.001
        pnl -= cost

        self._capital += pnl

        trade_record = {
            "asset": symbol,
            "action": action,
            "strategy_id": "aqtis_multi_indicator",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": shares,
            "pnl": pnl,
            "pnl_percent": pnl_pct,
            "market_regime": pos.get("regime", "unknown"),
            "exit_reason": reason,
            "timestamp": day_str,
            "entry_date": pos.get("entry_date"),
            "hold_duration_seconds": (
                pd.Timestamp(day_str) - pd.Timestamp(pos["entry_date"])
            ).total_seconds(),
            "signals": pos.get("signals", {}),
            "indicator_signals": pos.get("signals", {}),
        }
        self._daily_trades.append(trade_record)

    def _close_all_positions(self, market_data: Dict, last_date):
        """Close all remaining positions at last available price."""
        day_str = last_date.strftime("%Y-%m-%d")
        for symbol in list(self._positions.keys()):
            df = market_data.get(symbol)
            if df is not None and len(df) > 0:
                last_price = float(df["close"].iloc[-1])
                self._close_position(symbol, last_price, day_str, "end_of_sim")

    # ─────────────────────────────────────────────────────────────────
    # PORTFOLIO TRACKING (0 tokens)
    # ─────────────────────────────────────────────────────────────────

    def _mark_to_market(self, market_data: Dict, sim_date) -> float:
        """Calculate realized P&L for the day."""
        realized = 0.0
        for trade in self._daily_trades:
            if trade.get("pnl") is not None:
                realized += trade["pnl"]
        return realized

    def _unrealized_pnl(self, market_data: Dict, sim_date) -> float:
        """Calculate unrealized P&L across open positions."""
        unrealized = 0.0
        for symbol, pos in self._positions.items():
            df = market_data.get(symbol)
            if df is None:
                continue
            mask = df.index.normalize() <= sim_date
            if not mask.any():
                continue
            current_price = float(df.loc[mask, "close"].iloc[-1])
            entry_price = pos["entry_price"]
            shares = pos["shares"]
            if pos["action"] == "BUY":
                unrealized += (current_price - entry_price) * shares
            else:
                unrealized += (entry_price - current_price) * shares
        return unrealized

    # ─────────────────────────────────────────────────────────────────
    # LLM BUDGET MANAGEMENT
    # ─────────────────────────────────────────────────────────────────

    def _can_use_llm(self) -> bool:
        """Check if LLM budget allows another call."""
        return self._llm_calls_used < self.llm_budget

    def _use_llm_call(self):
        """Track an LLM call."""
        self._llm_calls_used += 1

    # ─────────────────────────────────────────────────────────────────
    # END-OF-DAY ANALYSIS (1 LLM call per day)
    # ─────────────────────────────────────────────────────────────────

    def _end_of_day_analysis(self, day_str: str, day_idx: int, total_days: int):
        """
        Batch end-of-day analysis. Uses at most 1 LLM call.

        Instead of analyzing each trade individually (expensive),
        we batch all trades into a single daily summary.
        """
        completed_trades = [t for t in self._daily_trades if t.get("pnl") is not None]
        if not completed_trades:
            return

        # Run end-of-day learning (uses post_mortem agent internally)
        # But skip LLM for individual trades — we'll do a batch summary
        self.orchestrator.end_of_day_learning(day_str)
        self._use_llm_call()

    # ─────────────────────────────────────────────────────────────────
    # END-OF-RUN REVIEW (1 LLM call)
    # ─────────────────────────────────────────────────────────────────

    def _end_of_run_review(self, metrics: Dict) -> Optional[Dict]:
        """Generate end-of-run review. 1 LLM call."""
        if not self._can_use_llm():
            return None

        cross_run = self.orchestrator.get_cross_run_summary()
        self._use_llm_call()

        return {
            "cross_run_summary": cross_run,
            "metrics": metrics,
        }

    # ─────────────────────────────────────────────────────────────────
    # METRICS CALCULATION
    # ─────────────────────────────────────────────────────────────────

    def _calculate_metrics(self, starting_capital: float) -> Dict:
        """Calculate comprehensive performance metrics."""
        completed = [t for t in self._all_trades if t.get("pnl") is not None]

        if not completed:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "sharpe_ratio": 0,
                "total_pnl": 0,
                "max_drawdown": 0,
            }

        pnls = [t.get("pnl", 0) for t in completed]
        pnl_pcts = [t.get("pnl_percent", 0) for t in completed]
        wins = sum(1 for p in pnls if p > 0)
        losses = len(pnls) - wins

        # Sharpe ratio (annualized from daily)
        daily_returns = np.array(self._daily_pnl) / starting_capital
        sharpe = 0.0
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))

        # Max drawdown
        equity = np.cumsum([starting_capital] + list(np.array(self._daily_pnl)))
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / (peak + 1e-10)
        max_dd = float(np.min(drawdown))

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / (gross_loss + 1e-10)

        # Win/loss ratios
        avg_win = gross_profit / wins if wins > 0 else 0
        avg_loss = gross_loss / losses if losses > 0 else 0

        # Regime breakdown
        regime_stats = {}
        for t in completed:
            r = t.get("market_regime", "unknown")
            if r not in regime_stats:
                regime_stats[r] = {"trades": 0, "wins": 0, "pnl": 0}
            regime_stats[r]["trades"] += 1
            if t.get("pnl", 0) > 0:
                regime_stats[r]["wins"] += 1
            regime_stats[r]["pnl"] += t.get("pnl", 0)

        return {
            "total_trades": len(completed),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(completed) if completed else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": float(np.mean(pnls)),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "regime_breakdown": regime_stats,
        }
