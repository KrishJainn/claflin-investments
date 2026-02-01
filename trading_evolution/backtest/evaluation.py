"""
Evaluation module for backtest analysis.

Provides:
- Comprehensive performance metrics
- Rolling metrics over time
- Regime-based performance analysis
- Trade distribution analysis
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import logging

from .result import BacktestResult, Trade

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    
    # Basic counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Win rate
    win_rate: float = 0.0
    
    # P&L
    gross_pnl: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0
    
    # Returns
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    
    # Trade statistics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade: float = 0.0
    
    # Risk-adjusted
    profit_factor: float = 0.0
    payoff_ratio: float = 0.0  # avg win / avg loss
    expectancy: float = 0.0
    expectancy_ratio: float = 0.0  # expectancy / avg loss
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0  # in bars
    
    # Trade duration
    avg_trade_duration: timedelta = field(default_factory=timedelta)
    avg_winner_duration: timedelta = field(default_factory=timedelta)
    avg_loser_duration: timedelta = field(default_factory=timedelta)
    
    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0  # positive = wins, negative = losses
    
    # R-multiples
    avg_r_multiple: float = 0.0
    max_r_multiple: float = 0.0
    min_r_multiple: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'gross_pnl': self.gross_pnl,
            'total_costs': self.total_costs,
            'net_pnl': self.net_pnl,
            'total_return_pct': self.total_return_pct,
            'annualized_return_pct': self.annualized_return_pct,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'avg_trade': self.avg_trade,
            'profit_factor': self.profit_factor,
            'payoff_ratio': self.payoff_ratio,
            'expectancy': self.expectancy,
            'expectancy_ratio': self.expectancy_ratio,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_drawdown_duration': self.max_drawdown_duration,
            'avg_trade_duration_hours': self.avg_trade_duration.total_seconds() / 3600,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'avg_r_multiple': self.avg_r_multiple,
        }


@dataclass
class RegimePerformance:
    """Performance metrics for a specific market regime."""
    
    regime: str
    metrics: PerformanceMetrics
    trade_count: int
    pct_of_total_trades: float


class Evaluator:
    """
    Comprehensive evaluation of backtest results.
    """
    
    def __init__(self, initial_capital: float = 1_000_000.0):
        """
        Initialize evaluator.
        
        Args:
            initial_capital: Starting capital for return calculations
        """
        self.initial_capital = initial_capital
    
    def calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.DataFrame = None,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive metrics from trades.
        
        Args:
            trades: List of completed trades
            equity_curve: Optional equity curve DataFrame
            
        Returns:
            PerformanceMetrics object
        """
        if not trades:
            return PerformanceMetrics()
        
        metrics = PerformanceMetrics()
        
        # Basic counts
        metrics.total_trades = len(trades)
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        
        metrics.winning_trades = len(winners)
        metrics.losing_trades = len(losers)
        metrics.win_rate = len(winners) / len(trades)
        
        # P&L
        metrics.gross_pnl = sum(t.gross_pnl for t in trades)
        metrics.total_costs = sum(t.total_costs for t in trades)
        metrics.net_pnl = sum(t.net_pnl for t in trades)
        
        # Returns
        metrics.total_return_pct = metrics.net_pnl / self.initial_capital
        
        # Calculate trading days for annualization
        if trades:
            start_date = min(t.entry_time for t in trades)
            end_date = max(t.exit_time for t in trades)
            trading_days = (end_date - start_date).days
            if trading_days > 0:
                years = trading_days / 365
                if years > 0:
                    metrics.annualized_return_pct = ((1 + metrics.total_return_pct) ** (1 / years)) - 1
        
        # Trade statistics
        if winners:
            win_pnls = [t.net_pnl for t in winners]
            metrics.avg_win = sum(win_pnls) / len(win_pnls)
            metrics.largest_win = max(win_pnls)
        
        if losers:
            loss_pnls = [t.net_pnl for t in losers]
            metrics.avg_loss = abs(sum(loss_pnls) / len(loss_pnls))
            metrics.largest_loss = abs(min(loss_pnls))
        
        metrics.avg_trade = metrics.net_pnl / len(trades)
        
        # Profit factor and payoff ratio
        total_wins = sum(t.net_pnl for t in winners) if winners else 0
        total_losses = abs(sum(t.net_pnl for t in losers)) if losers else 0
        
        if total_losses > 0:
            metrics.profit_factor = total_wins / total_losses
        elif total_wins > 0:
            metrics.profit_factor = float('inf')
        
        if metrics.avg_loss > 0:
            metrics.payoff_ratio = metrics.avg_win / metrics.avg_loss
        
        # Expectancy
        metrics.expectancy = metrics.avg_trade
        if metrics.avg_loss > 0:
            metrics.expectancy_ratio = metrics.expectancy / metrics.avg_loss
        
        # Trade durations
        durations = [t.holding_duration for t in trades]
        avg_seconds = sum(d.total_seconds() for d in durations) / len(durations)
        metrics.avg_trade_duration = timedelta(seconds=avg_seconds)
        
        if winners:
            winner_durations = [t.holding_duration for t in winners]
            avg_winner_seconds = sum(d.total_seconds() for d in winner_durations) / len(winner_durations)
            metrics.avg_winner_duration = timedelta(seconds=avg_winner_seconds)
        
        if losers:
            loser_durations = [t.holding_duration for t in losers]
            avg_loser_seconds = sum(d.total_seconds() for d in loser_durations) / len(loser_durations)
            metrics.avg_loser_duration = timedelta(seconds=avg_loser_seconds)
        
        # Streaks
        metrics.max_consecutive_wins, metrics.max_consecutive_losses = self._calculate_streaks(trades)
        
        # R-multiples
        r_multiples = [t.r_multiple for t in trades if t.r_multiple != 0]
        if r_multiples:
            metrics.avg_r_multiple = sum(r_multiples) / len(r_multiples)
            metrics.max_r_multiple = max(r_multiples)
            metrics.min_r_multiple = min(r_multiples)
        
        # Risk metrics from equity curve
        if equity_curve is not None and len(equity_curve) > 0:
            self._calculate_risk_metrics(metrics, equity_curve)
        
        return metrics
    
    def _calculate_streaks(self, trades: List[Trade]) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses."""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade.is_winner:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _calculate_risk_metrics(
        self,
        metrics: PerformanceMetrics,
        equity_curve: pd.DataFrame,
    ):
        """Calculate risk metrics from equity curve."""
        equity = equity_curve['total_equity'] if 'total_equity' in equity_curve.columns else equity_curve.iloc[:, 0]
        
        returns = equity.pct_change().dropna()
        
        if len(returns) > 0:
            # Sharpe ratio (annualized)
            risk_free_rate = 0.05 / 252  # ~5% annual, daily
            excess_returns = returns - risk_free_rate
            
            if returns.std() > 0:
                # Assuming 5-min bars: 78 bars/day * 252 days = 19656 bars/year
                bars_per_year = 252 * 78
                metrics.sharpe_ratio = np.sqrt(bars_per_year) * excess_returns.mean() / returns.std()
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                metrics.sortino_ratio = np.sqrt(bars_per_year) * excess_returns.mean() / downside_returns.std()
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdowns = (cumulative - running_max) / running_max
            
            metrics.max_drawdown_pct = abs(drawdowns.min())
            metrics.max_drawdown = metrics.max_drawdown_pct * self.initial_capital
            
            # Drawdown duration
            in_drawdown = drawdowns < 0
            if in_drawdown.any():
                dd_groups = (in_drawdown != in_drawdown.shift()).cumsum()
                dd_lengths = in_drawdown.groupby(dd_groups).sum()
                metrics.max_drawdown_duration = int(dd_lengths.max())
            
            # Calmar ratio
            if metrics.max_drawdown_pct > 0:
                annual_return = returns.mean() * bars_per_year
                metrics.calmar_ratio = annual_return / metrics.max_drawdown_pct
    
    def rolling_metrics(
        self,
        trades: List[Trade],
        window: int = 30,
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            trades: List of trades
            window: Rolling window size (number of trades)
            
        Returns:
            DataFrame with rolling metrics
        """
        if len(trades) < window:
            return pd.DataFrame()
        
        rows = []
        
        for i in range(window, len(trades) + 1):
            window_trades = trades[i - window:i]
            
            winners = [t for t in window_trades if t.is_winner]
            pnl = sum(t.net_pnl for t in window_trades)
            
            rows.append({
                'trade_num': i,
                'end_time': window_trades[-1].exit_time,
                'win_rate': len(winners) / window,
                'cumulative_pnl': sum(t.net_pnl for t in trades[:i]),
                'rolling_pnl': pnl,
                'rolling_avg_trade': pnl / window,
            })
        
        return pd.DataFrame(rows)
    
    def regime_performance(
        self,
        trades: List[Trade],
    ) -> Dict[str, RegimePerformance]:
        """
        Analyze performance by market regime.
        
        Args:
            trades: List of trades with market_regime field
            
        Returns:
            Dictionary of regime -> RegimePerformance
        """
        # Group trades by regime
        regime_trades: Dict[str, List[Trade]] = {}
        
        for trade in trades:
            regime = trade.market_regime or 'unknown'
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(trade)
        
        # Calculate metrics for each regime
        results = {}
        
        for regime, regime_trade_list in regime_trades.items():
            metrics = self.calculate_metrics(regime_trade_list)
            results[regime] = RegimePerformance(
                regime=regime,
                metrics=metrics,
                trade_count=len(regime_trade_list),
                pct_of_total_trades=len(regime_trade_list) / len(trades) if trades else 0,
            )
        
        return results
    
    def exit_reason_analysis(self, trades: List[Trade]) -> Dict[str, Dict]:
        """Analyze performance by exit reason."""
        reason_trades: Dict[str, List[Trade]] = {}
        
        for trade in trades:
            reason = trade.exit_reason
            if reason not in reason_trades:
                reason_trades[reason] = []
            reason_trades[reason].append(trade)
        
        results = {}
        for reason, reason_trade_list in reason_trades.items():
            winners = [t for t in reason_trade_list if t.is_winner]
            results[reason] = {
                'count': len(reason_trade_list),
                'pct_of_total': len(reason_trade_list) / len(trades) if trades else 0,
                'win_rate': len(winners) / len(reason_trade_list),
                'total_pnl': sum(t.net_pnl for t in reason_trade_list),
                'avg_pnl': sum(t.net_pnl for t in reason_trade_list) / len(reason_trade_list),
            }
        
        return results
    
    def generate_report(self, result: BacktestResult) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            result: BacktestResult to analyze
            
        Returns:
            Formatted report string
        """
        metrics = self.calculate_metrics(result.trades, result.equity_curve)
        regime_perf = self.regime_performance(result.trades)
        exit_analysis = self.exit_reason_analysis(result.trades)
        
        report = f"""
{'='*60}
BACKTEST EVALUATION REPORT
{'='*60}
Strategy: {result.strategy_name} v{result.strategy_version}
Period: {result.start_date} to {result.end_date}
Config Hash: {result.config_hash}
Run ID: {result.run_id}

{'='*60}
PERFORMANCE SUMMARY
{'='*60}
Total Trades:        {metrics.total_trades}
Win Rate:            {metrics.win_rate:.1%}
Profit Factor:       {metrics.profit_factor:.2f}

Gross P&L:           ₹{metrics.gross_pnl:,.2f}
Total Costs:         ₹{metrics.total_costs:,.2f}
Net P&L:             ₹{metrics.net_pnl:,.2f}

Total Return:        {metrics.total_return_pct:.1%}
Annualized Return:   {metrics.annualized_return_pct:.1%}

{'='*60}
TRADE STATISTICS
{'='*60}
Average Win:         ₹{metrics.avg_win:,.2f}
Average Loss:        ₹{metrics.avg_loss:,.2f}
Largest Win:         ₹{metrics.largest_win:,.2f}
Largest Loss:        ₹{metrics.largest_loss:,.2f}
Payoff Ratio:        {metrics.payoff_ratio:.2f}
Expectancy:          ₹{metrics.expectancy:,.2f}

Avg Trade Duration:  {metrics.avg_trade_duration}
Max Consecutive Wins: {metrics.max_consecutive_wins}
Max Consecutive Losses: {metrics.max_consecutive_losses}

{'='*60}
RISK METRICS
{'='*60}
Sharpe Ratio:        {metrics.sharpe_ratio:.2f}
Sortino Ratio:       {metrics.sortino_ratio:.2f}
Calmar Ratio:        {metrics.calmar_ratio:.2f}
Max Drawdown:        {metrics.max_drawdown_pct:.1%} (₹{metrics.max_drawdown:,.2f})

{'='*60}
EXIT REASON ANALYSIS
{'='*60}
"""
        for reason, data in exit_analysis.items():
            report += f"{reason:20s}: {data['count']:4d} trades ({data['pct_of_total']:.0%}), "
            report += f"WR: {data['win_rate']:.0%}, P&L: ₹{data['total_pnl']:,.0f}\n"
        
        if regime_perf:
            report += f"""
{'='*60}
REGIME PERFORMANCE
{'='*60}
"""
            for regime, perf in regime_perf.items():
                report += f"{regime:20s}: {perf.trade_count:4d} trades ({perf.pct_of_total_trades:.0%}), "
                report += f"WR: {perf.metrics.win_rate:.0%}, P&L: ₹{perf.metrics.net_pnl:,.0f}\n"
        
        return report
