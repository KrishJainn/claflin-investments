"""
Equity Curve Plotting module.

Generates matplotlib visualizations of equity over time.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EquityCurveChart:
    """
    Generates equity curve visualizations.

    Features:
    - Equity over time
    - Drawdown visualization
    - Trade markers
    - Benchmark comparison
    """

    def __init__(self,
                 figsize: Tuple[int, int] = (14, 10),
                 style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize equity curve chart generator.

        Args:
            figsize: Figure size (width, height)
            style: Matplotlib style
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')

    def plot_equity_curve(self,
                          equity_data: pd.DataFrame,
                          save_path: str = None,
                          title: str = "Equity Curve",
                          show_drawdown: bool = True,
                          show_trades: bool = True,
                          trades: List[Dict] = None,
                          benchmark: pd.Series = None) -> str:
        """
        Plot equity curve with optional features.

        Args:
            equity_data: DataFrame with 'date' and 'equity' columns
            save_path: Path to save PNG
            title: Chart title
            show_drawdown: Show drawdown subplot
            show_trades: Show trade markers
            trades: List of trades for markers
            benchmark: Benchmark equity series

        Returns:
            Path to saved file
        """
        # Create figure with subplots
        if show_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize,
                                           height_ratios=[3, 1],
                                           sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=self.figsize)
            ax2 = None

        # Prepare data
        if isinstance(equity_data, pd.DataFrame):
            dates = pd.to_datetime(equity_data['date']) if 'date' in equity_data.columns else equity_data.index
            equity = equity_data['equity'].values if 'equity' in equity_data.columns else equity_data.iloc[:, 0].values
        else:
            dates = equity_data.index
            equity = equity_data.values

        # Plot equity curve
        ax1.plot(dates, equity, 'b-', linewidth=1.5, label='Strategy')
        ax1.fill_between(dates, equity[0], equity, alpha=0.3)

        # Add benchmark if provided
        if benchmark is not None:
            # Normalize benchmark to start at same value as equity
            benchmark_normalized = benchmark / benchmark.iloc[0] * equity[0]
            ax1.plot(benchmark.index, benchmark_normalized, 'gray',
                    linewidth=1, alpha=0.7, linestyle='--', label='Benchmark')
            ax1.legend(loc='upper left')

        # Add trade markers
        if show_trades and trades:
            self._add_trade_markers(ax1, trades, dates, equity)

        # Format equity plot
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Equity ($)', fontsize=11)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, alpha=0.3)

        # Add key statistics as text
        final_equity = equity[-1]
        initial_equity = equity[0]
        total_return = (final_equity - initial_equity) / initial_equity * 100
        max_equity = np.max(equity)

        stats_text = (
            f"Final: ${final_equity:,.0f}\n"
            f"Return: {total_return:+.1f}%\n"
            f"Peak: ${max_equity:,.0f}"
        )
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot drawdown
        if show_drawdown and ax2 is not None:
            drawdown = self._calculate_drawdown(equity)
            ax2.fill_between(dates, 0, drawdown * 100, color='red', alpha=0.5)
            ax2.plot(dates, drawdown * 100, 'r-', linewidth=1)

            ax2.set_ylabel('Drawdown (%)', fontsize=11)
            ax2.set_xlabel('Date', fontsize=11)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
            ax2.set_ylim(min(drawdown * 100) * 1.1, 5)
            ax2.grid(True, alpha=0.3)

            # Max drawdown annotation
            max_dd = min(drawdown)
            max_dd_idx = np.argmin(drawdown)
            ax2.annotate(f'Max: {max_dd*100:.1f}%',
                        xy=(dates[max_dd_idx], max_dd * 100),
                        xytext=(10, -20), textcoords='offset points',
                        fontsize=9, color='darkred')

        # Format x-axis dates
        if ax2 is not None:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax1.set_xlabel('Date', fontsize=11)

        plt.tight_layout()

        # Save to file
        if save_path is None:
            save_path = 'equity_curve.png'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        logger.info(f"Saved equity curve to {save_path}")
        return save_path

    def _calculate_drawdown(self, equity: np.ndarray) -> np.ndarray:
        """Calculate drawdown series."""
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return drawdown

    def _add_trade_markers(self,
                           ax: plt.Axes,
                           trades: List[Dict],
                           dates: pd.DatetimeIndex,
                           equity: np.ndarray):
        """Add trade entry/exit markers."""
        # Sample trades if too many
        if len(trades) > 50:
            trades = trades[::len(trades)//50]

        for trade in trades:
            entry_date = pd.to_datetime(trade.get('entry_date'))
            exit_date = pd.to_datetime(trade.get('exit_date'))
            pnl = trade.get('net_pnl', 0)
            direction = trade.get('direction', 'LONG')

            # Find closest equity value
            entry_idx = np.searchsorted(dates, entry_date)
            exit_idx = np.searchsorted(dates, exit_date)

            if entry_idx < len(equity) and exit_idx < len(equity):
                color = 'green' if pnl > 0 else 'red'
                marker = '^' if direction == 'LONG' else 'v'

                # Entry marker
                ax.scatter(dates[entry_idx], equity[entry_idx],
                          c='blue', marker=marker, s=30, alpha=0.6, zorder=5)

                # Exit marker
                ax.scatter(dates[exit_idx], equity[exit_idx],
                          c=color, marker='o', s=30, alpha=0.6, zorder=5)

    def plot_multiple_equity_curves(self,
                                    curves: Dict[str, pd.DataFrame],
                                    save_path: str = None,
                                    title: str = "Equity Curves Comparison") -> str:
        """
        Plot multiple equity curves for comparison.

        Args:
            curves: Dictionary mapping name to equity DataFrame
            save_path: Path to save PNG
            title: Chart title

        Returns:
            Path to saved file
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))

        for i, (name, data) in enumerate(curves.items()):
            if isinstance(data, pd.DataFrame):
                dates = pd.to_datetime(data['date']) if 'date' in data.columns else data.index
                equity = data['equity'].values if 'equity' in data.columns else data.iloc[:, 0].values
            else:
                dates = data.index
                equity = data.values

            # Normalize to start at 100
            equity_normalized = equity / equity[0] * 100

            ax.plot(dates, equity_normalized, color=colors[i],
                   linewidth=1.5, label=name)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Equity (Base 100)', fontsize=11)
        ax.set_xlabel('Date', fontsize=11)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_path is None:
            save_path = 'equity_comparison.png'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        logger.info(f"Saved equity comparison to {save_path}")
        return save_path

    def plot_rolling_returns(self,
                             equity_data: pd.DataFrame,
                             window: int = 20,
                             save_path: str = None) -> str:
        """
        Plot rolling returns over time.

        Args:
            equity_data: DataFrame with equity values
            window: Rolling window size
            save_path: Path to save PNG

        Returns:
            Path to saved file
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        if isinstance(equity_data, pd.DataFrame):
            dates = pd.to_datetime(equity_data['date']) if 'date' in equity_data.columns else equity_data.index
            equity = equity_data['equity'].values if 'equity' in equity_data.columns else equity_data.iloc[:, 0].values
        else:
            dates = equity_data.index
            equity = equity_data.values

        # Calculate daily returns
        returns = np.diff(equity) / equity[:-1]
        returns_dates = dates[1:]

        # Rolling metrics
        returns_series = pd.Series(returns, index=returns_dates)
        rolling_return = returns_series.rolling(window).mean() * 252 * 100  # Annualized
        rolling_vol = returns_series.rolling(window).std() * np.sqrt(252) * 100

        # Plot rolling return
        ax1.plot(rolling_return.index, rolling_return.values, 'b-', linewidth=1)
        ax1.fill_between(rolling_return.index, 0, rolling_return.values,
                        where=rolling_return > 0, color='green', alpha=0.3)
        ax1.fill_between(rolling_return.index, 0, rolling_return.values,
                        where=rolling_return <= 0, color='red', alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.set_ylabel(f'{window}-Day Rolling Return (Ann. %)', fontsize=11)
        ax1.set_title('Rolling Returns and Volatility', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Plot rolling volatility
        ax2.plot(rolling_vol.index, rolling_vol.values, 'orange', linewidth=1)
        ax2.fill_between(rolling_vol.index, 0, rolling_vol.values,
                        color='orange', alpha=0.3)
        ax2.set_ylabel(f'{window}-Day Rolling Vol (Ann. %)', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.grid(True, alpha=0.3)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_path is None:
            save_path = 'rolling_returns.png'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        logger.info(f"Saved rolling returns to {save_path}")
        return save_path


def plot_generation_equity_curves(generation_equities: Dict[int, pd.DataFrame],
                                  save_path: str = None) -> str:
    """
    Plot equity curves for different generations.

    Args:
        generation_equities: Dict mapping generation number to equity DataFrame
        save_path: Path to save PNG

    Returns:
        Path to saved file
    """
    chart = EquityCurveChart()

    # Convert to format expected by plot_multiple_equity_curves
    curves = {f"Gen {gen}": data for gen, data in generation_equities.items()}

    return chart.plot_multiple_equity_curves(
        curves=curves,
        save_path=save_path,
        title="Equity Curves by Generation"
    )


def create_equity_report(equity_data: pd.DataFrame,
                         trades: List[Dict],
                         output_dir: str) -> Dict[str, str]:
    """
    Create complete equity report with multiple charts.

    Args:
        equity_data: DataFrame with equity values
        trades: List of trades
        output_dir: Directory to save charts

    Returns:
        Dictionary mapping chart name to file path
    """
    chart = EquityCurveChart()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = {}

    # Main equity curve
    files['equity_curve'] = chart.plot_equity_curve(
        equity_data=equity_data,
        save_path=str(output_path / 'equity_curve.png'),
        title="Strategy Equity Curve",
        show_drawdown=True,
        show_trades=True,
        trades=trades
    )

    # Rolling returns
    files['rolling_returns'] = chart.plot_rolling_returns(
        equity_data=equity_data,
        window=20,
        save_path=str(output_path / 'rolling_returns.png')
    )

    logger.info(f"Created equity report with {len(files)} charts in {output_dir}")

    return files
