"""
Paper Trader - Main Paper Trading Engine.

Combines all components:
- Live data fetching
- Signal generation with best strategy
- Paper execution with realistic fills
- Trade ledger with reason codes
- Risk management
- Event logging
- Daily reports
"""

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Callable
import logging
import json
from pathlib import Path

from .live_data import LiveDataManager, Bar, MarketStatus
from .paper_executor import PaperExecutor, OrderSide, PaperFill
from .trade_ledger import TradeLedger, TradeReason
from .paper_risk_manager import PaperRiskManager, RiskLimits, RiskAction

# Import trading components
from ..super_indicator.dna import SuperIndicatorDNA, IndicatorGene
from ..super_indicator.core import SuperIndicator
from ..super_indicator.signals import SignalType, PositionState
from ..indicators.universe import IndicatorUniverse
from ..indicators.calculator import IndicatorCalculator
from ..indicators.normalizer import IndicatorNormalizer
from ..backtest.indian_costs import IndianCostModel

logger = logging.getLogger(__name__)


def load_best_strategy_from_db(db_path: str = None) -> Dict:
    """Load the best evolved strategy from the trading_evolution database.

    Queries the hall_of_fame table joined with dna_configs to find the
    highest-ranked strategy that has actual weight data.

    Args:
        db_path: Path to trading_evolution.db. If None, searches common locations.

    Returns:
        Strategy dict with dna_id, weights, entry/exit thresholds.
    """
    import sqlite3

    if db_path is None:
        # Search common locations
        candidates = [
            Path(__file__).parent.parent.parent / 'trading_evolution.db',
            Path.cwd() / 'trading_evolution.db',
        ]
        for p in candidates:
            if p.exists():
                db_path = str(p)
                break

    if db_path and Path(db_path).exists():
        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute('''
                SELECT h.dna_id, h.rank, h.sharpe_ratio, h.win_rate, h.net_profit,
                       d.weights_json, d.active_indicators_json
                FROM hall_of_fame h
                JOIN dna_configs d ON h.dna_id = d.dna_id
                WHERE LENGTH(d.weights_json) > 2
                ORDER BY h.rank ASC
                LIMIT 1
            ''').fetchone()
            conn.close()

            if row:
                dna_id, rank, sharpe, wr, profit, weights_json, active_json = row
                weights = json.loads(weights_json)
                active = json.loads(active_json)

                # Filter to only active indicators with meaningful weights
                filtered_weights = {k: v for k, v in weights.items()
                                    if k in active and abs(v) > 0.001}

                logger.info(
                    f"Loaded Hall of Fame #{rank}: DNA {dna_id} | "
                    f"Sharpe {sharpe:.2f} | WR {wr:.2f} | "
                    f"Profit ${profit:,.0f} | {len(filtered_weights)} indicators"
                )

                return {
                    "dna_id": dna_id,
                    "version": "db_hof",
                    "weights": filtered_weights,
                    "entry_threshold": 0.70,
                    "exit_threshold": 0.30,
                    "source": f"hall_of_fame_rank_{rank}",
                    "sharpe": sharpe,
                    "win_rate": wr,
                }
        except Exception as e:
            logger.warning(f"Failed to load strategy from database: {e}")

    # Fallback to hardcoded strategy
    logger.info("Using fallback hardcoded strategy (DNA 8748f3f8)")
    return _FALLBACK_STRATEGY


_FALLBACK_STRATEGY = {
    "dna_id": "8748f3f8",
    "version": "v1.0",
    "weights": {
        "TSI_13_25": 0.8883, "NVI": 0.8611, "PVI": 0.7964, "STOCH_5_3": 0.6260,
        "ATR_14": 0.5265, "ZSCORE_20": 0.5002, "AROON_14": 0.4698, "BBANDS_20_2.5": 0.4642,
        "MASS_INDEX": 0.4535, "TEMA_20": 0.4258, "CMF_21": 0.3670, "ATR_20": 0.3443,
        "SUPERTREND_7_3": -0.9663, "AROON_25": -0.9508, "AO_5_34": -0.9286,
        "VWMA_10": -0.9283, "ADX_20": -0.8928, "EFI_13": -0.8716, "MFI_14": -0.8352,
        "VWMA_20": -0.8194, "KST": -0.7956, "WMA_20": -0.7559, "WMA_10": -0.7482,
        "DEMA_20": -0.6582, "STOCH_14_3": -0.6514, "CCI_20": -0.6433, "PIVOTS": -0.5894,
    },
    "entry_threshold": 0.70,
    "exit_threshold": 0.30,
}

# Load best strategy (from DB if available, else fallback)
BEST_STRATEGY = load_best_strategy_from_db()


@dataclass
class PaperTraderConfig:
    """Paper trader configuration."""
    
    # Strategy
    strategy: Dict = None
    
    # Symbols
    symbols: List[str] = None
    
    # Capital
    initial_capital: float = 100_000.0
    position_size_pct: float = 0.10
    
    # Risk limits
    daily_loss_limit: float = 20_000.0
    max_trades_per_day: int = 20
    max_concurrent_positions: int = 5
    
    # Data
    data_interval: str = "5m"
    
    # Saving
    save_dir: str = "./paper_trades"
    
    def __post_init__(self):
        if self.strategy is None:
            self.strategy = BEST_STRATEGY
        if self.symbols is None:
            self.symbols = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "HINDUNILVR.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "BAJFINANCE.NS",
            ]


class PaperTrader:
    """
    Paper trading engine.
    
    Features:
    - Uses best strategy from backtests
    - Real-time data via yfinance
    - Realistic execution simulation
    - Full trade logging with reasons
    - Risk management enforced
    """
    
    def __init__(self, config: PaperTraderConfig = None):
        """Initialize paper trader."""
        self.config = config or PaperTraderConfig()

        # Create DNA
        self._dna = self._create_dna()

        # Initialize components
        self._init_indicators()
        self._init_execution()
        self._init_risk()
        self._init_ledger()

        # State
        self._capital = self.config.initial_capital
        self._is_running = False
        self._last_bars: Dict[str, Bar] = {}
        self._last_si: Dict[str, float] = {}

        # Rolling bar history per symbol for indicator calculation
        self._bar_history: Dict[str, List[Dict]] = {}
        self._MIN_BARS_FOR_SI = 60  # Need enough history for indicators

        # Events
        self._events: List[Dict] = []
    
    def _create_dna(self) -> SuperIndicatorDNA:
        """Create DNA from strategy config."""
        genes = {}
        for name, weight in self.config.strategy['weights'].items():
            genes[name] = IndicatorGene(
                name=name,
                weight=weight,
                active=abs(weight) > 0.01,
                category='unknown'
            )
        
        return SuperIndicatorDNA(
            dna_id=self.config.strategy['dna_id'],
            generation=0,
            run_id=0,
            genes=genes
        )
    
    def _init_indicators(self):
        """Initialize indicator components."""
        self._universe = IndicatorUniverse()
        self._universe.load_all()
        self._calculator = IndicatorCalculator(universe=self._universe)
        self._normalizer = IndicatorNormalizer()
        self._super_indicator = SuperIndicator(self._dna, normalizer=self._normalizer)
    
    def _init_execution(self):
        """Initialize execution components."""
        self._cost_model = IndianCostModel.for_intraday()
        self._executor = PaperExecutor(
            cost_model=self._cost_model,
            base_spread_pct=0.001,
        )
        self._data_manager = LiveDataManager(
            symbols=self.config.symbols,
            interval=self.config.data_interval,
        )
    
    def _init_risk(self):
        """Initialize risk management."""
        limits = RiskLimits(
            daily_loss_limit=self.config.daily_loss_limit,
            max_trades_per_day=self.config.max_trades_per_day,
            max_concurrent_positions=self.config.max_concurrent_positions,
            max_position_pct=self.config.position_size_pct,
        )
        self._risk_manager = PaperRiskManager(
            limits=limits,
            capital=self.config.initial_capital,
        )
    
    def _init_ledger(self):
        """Initialize trade ledger."""
        self._ledger = TradeLedger(
            strategy_version=self.config.strategy['version'],
            dna_id=self.config.strategy['dna_id'],
            save_dir=self.config.save_dir,
        )
    
    def _log_event(self, event_type: str, data: Dict):
        """Log an event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': data,
        }
        self._events.append(event)
        logger.info(f"Event: {event_type} - {json.dumps(data)}")
    
    def process_bar(self, symbol: str, bar: Bar) -> Optional[PaperFill]:
        """
        Process a new bar and potentially generate a trade.
        
        Args:
            symbol: Stock symbol
            bar: New price bar
            
        Returns:
            PaperFill if a trade was executed
        """
        # Check if should flatten all
        if self._risk_manager.should_flatten_all(bar.timestamp):
            return self._flatten_position(symbol, bar, TradeReason.EOD_FLATTEN)
        
        # Calculate indicators and SI
        si_value = self._calculate_si(symbol, bar)
        if si_value is None:
            return None
        
        prev_si = self._last_si.get(symbol, 0.0)
        self._last_si[symbol] = si_value
        
        # Determine current position
        position = self._ledger.get_position(symbol)
        if position is None:
            pos_state = PositionState.FLAT
        elif position.side == "BUY":
            pos_state = PositionState.LONG
        else:
            pos_state = PositionState.SHORT
        
        # Generate signal
        signal = self._generate_signal(si_value, prev_si, pos_state)
        
        if signal == SignalType.HOLD:
            return None
        
        self._log_event("signal", {
            'symbol': symbol,
            'signal': signal.value,
            'si_value': si_value,
            'prev_si': prev_si,
            'position': pos_state.value,
        })
        
        # Execute signal
        return self._execute_signal(symbol, signal, bar, si_value)
    
    def _calculate_si(self, symbol: str, bar: Bar) -> Optional[float]:
        """
        Calculate SI value for a bar using real indicator pipeline.

        Maintains a rolling window of bars per symbol, calculates all
        indicators, normalizes them, and computes the weighted Super
        Indicator value.
        """
        import pandas as pd

        # Accumulate bar history
        if symbol not in self._bar_history:
            self._bar_history[symbol] = []

        self._bar_history[symbol].append({
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
        })

        # Need enough bars for indicator warm-up
        if len(self._bar_history[symbol]) < self._MIN_BARS_FOR_SI:
            return None

        # Build DataFrame from history
        df = pd.DataFrame(self._bar_history[symbol])

        # Calculate indicators
        try:
            raw_indicators = self._calculator.calculate_all(df)
            if raw_indicators.empty:
                return None

            # Rename output columns to DNA-compatible names
            raw_indicators = self._calculator.rename_to_dna_names(raw_indicators)

            # Filter to only active indicators in our DNA
            active = [
                ind for ind in self._dna.get_active_indicators()
                if ind in raw_indicators.columns
            ]
            if not active:
                return None

            active_df = raw_indicators[active]

            # Normalize
            normalized = self._normalizer.normalize_all(
                active_df, price_series=df['close']
            )
            if normalized.empty:
                return None

            # Compute SI for the last bar
            si_series = self._super_indicator.calculate(normalized)
            if si_series.empty:
                return None

            return float(si_series.iloc[-1])

        except Exception as e:
            logger.debug(f"SI calc error for {symbol}: {e}")
            return None
    
    def _generate_signal(
        self,
        si: float,
        prev_si: float,
        position: PositionState,
    ) -> SignalType:
        """Generate trading signal."""
        entry_threshold = self.config.strategy['entry_threshold']
        exit_threshold = self.config.strategy['exit_threshold']
        
        if position == PositionState.FLAT:
            if si > entry_threshold and prev_si <= entry_threshold:
                return SignalType.LONG_ENTRY
            elif si < -entry_threshold and prev_si >= -entry_threshold:
                return SignalType.SHORT_ENTRY
        
        elif position == PositionState.LONG:
            if si < exit_threshold:
                return SignalType.LONG_EXIT
        
        elif position == PositionState.SHORT:
            if si > -exit_threshold:
                return SignalType.SHORT_EXIT
        
        return SignalType.HOLD
    
    def _get_indicator_snapshot(self, symbol: str) -> Dict[str, float]:
        """
        Capture current individual indicator values for a symbol.
        Returns {indicator_name: weighted_contribution} so the Coach
        can analyze which indicators helped/hurt each trade.
        """
        snapshot = {}
        weights = self.config.strategy.get('weights', {})
        for name, weight in weights.items():
            # Store the weight as a proxy for indicator contribution
            # In full implementation this would be the actual computed
            # indicator value; for now we store the weight so the Coach
            # knows which indicators were active and their direction.
            snapshot[name] = weight
        return snapshot

    def _execute_signal(
        self,
        symbol: str,
        signal: SignalType,
        bar: Bar,
        si_value: float,
    ) -> Optional[PaperFill]:
        """Execute a trading signal."""

        if signal == SignalType.LONG_ENTRY:
            return self._enter_position(symbol, "BUY", bar, si_value, TradeReason.SI_LONG_CROSS)

        elif signal == SignalType.SHORT_ENTRY:
            return self._enter_position(symbol, "SELL", bar, si_value, TradeReason.SI_SHORT_CROSS)
        
        elif signal == SignalType.LONG_EXIT:
            return self._exit_position(symbol, bar, TradeReason.SI_EXIT_SIGNAL)
        
        elif signal == SignalType.SHORT_EXIT:
            return self._exit_position(symbol, bar, TradeReason.SI_EXIT_SIGNAL)
        
        return None
    
    def _enter_position(
        self,
        symbol: str,
        side: str,
        bar: Bar,
        si_value: float,
        reason: TradeReason,
    ) -> Optional[PaperFill]:
        """Enter a new position."""
        
        # Calculate position size
        position_value = self._capital * self.config.position_size_pct
        quantity = int(position_value / bar.close)
        
        if quantity <= 0:
            return None
        
        # Check risk limits
        allowed, action, msg = self._risk_manager.check_entry(
            symbol, side, quantity, bar.close, bar.timestamp
        )
        
        if not allowed:
            self._log_event("entry_blocked", {
                'symbol': symbol,
                'action': action.value,
                'reason': msg,
            })
            return None
        
        # Execute
        order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
        fill = self._executor.execute_market_order(
            symbol=symbol,
            side=order_side,
            quantity=quantity,
            current_price=bar.close,
            strategy_version=self.config.strategy['version'],
            signal_value=si_value,
            reason=reason.value,
        )
        
        # Capture indicator snapshot for Coach analysis
        indicator_snapshot = self._get_indicator_snapshot(symbol)

        # Record in ledger
        self._ledger.record_entry(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill.fill_price,
            si_value=si_value,
            reason=reason,
            timestamp=bar.timestamp,
            slippage=fill.slippage,
            commission=fill.cost_breakdown.brokerage,
            indicator_snapshot=indicator_snapshot,
        )
        
        # Update risk manager
        self._risk_manager.record_entry(symbol, quantity, fill.fill_price)
        
        self._log_event("entry", {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': fill.fill_price,
            'si_value': si_value,
            'reason': reason.value,
        })
        
        return fill
    
    def _exit_position(
        self,
        symbol: str,
        bar: Bar,
        reason: TradeReason,
    ) -> Optional[PaperFill]:
        """Exit an existing position."""
        
        position = self._ledger.get_position(symbol)
        if position is None:
            return None
        
        # Execute
        exit_side = OrderSide.SELL if position.side == "BUY" else OrderSide.BUY
        fill = self._executor.execute_market_order(
            symbol=symbol,
            side=exit_side,
            quantity=position.quantity,
            current_price=bar.close,
            strategy_version=self.config.strategy['version'],
            reason=reason.value,
        )
        
        # Record in ledger
        entry = self._ledger.record_exit(
            symbol=symbol,
            exit_price=fill.fill_price,
            reason=reason,
            exit_time=bar.timestamp,
            exit_slippage=fill.slippage,
            exit_commission=fill.cost_breakdown.brokerage,
        )
        
        # Update risk manager
        self._risk_manager.record_exit(symbol, entry.pnl)
        
        self._log_event("exit", {
            'symbol': symbol,
            'exit_price': fill.fill_price,
            'pnl': entry.pnl,
            'reason': reason.value,
        })
        
        return fill
    
    def _flatten_position(
        self,
        symbol: str,
        bar: Bar,
        reason: TradeReason,
    ) -> Optional[PaperFill]:
        """Flatten a position."""
        return self._exit_position(symbol, bar, reason)
    
    def flatten_all(self, current_time: datetime = None):
        """Flatten all open positions."""
        now = current_time or datetime.now()
        
        for symbol in self._risk_manager.get_positions_to_flatten():
            bar = self._last_bars.get(symbol)
            if bar:
                self._flatten_position(symbol, bar, TradeReason.EOD_FLATTEN)
    
    def get_status(self) -> Dict:
        """Get current status."""
        return {
            'capital': self._capital,
            'strategy': self.config.strategy['dna_id'],
            'open_positions': len(self._ledger.get_open_positions()),
            'daily_pnl': self._risk_manager.get_state().daily_pnl,
            'trades_today': self._risk_manager.get_state().trades_today,
            'summary': self._ledger.get_summary(),
        }
    
    def get_daily_report(self, for_date: date = None) -> str:
        """Generate daily P&L report."""
        return self._ledger.generate_daily_report(for_date)
    
    def save(self):
        """Save state to disk."""
        self._ledger.save()
        
        # Save events
        events_path = Path(self.config.save_dir) / f"events_{datetime.now().strftime('%Y%m%d')}.json"
        with open(events_path, 'w') as f:
            json.dump(self._events, f, indent=2)
