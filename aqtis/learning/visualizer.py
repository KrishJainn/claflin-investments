"""
Learning Progress Visualizer for AQTIS

Provides visual representations of algorithm learning and improvement over time.
Supports both terminal-based ASCII visualizations and optional matplotlib charts.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .learning_log import LearningLog


class LearningVisualizer:
    """
    Visualizes learning progress from the LearningLog.

    Supports:
    - ASCII charts for terminal display
    - Progress bars and sparklines
    - Summary dashboards
    - Optional matplotlib integration for detailed charts
    """

    def __init__(self, learning_log: LearningLog):
        self.log = learning_log

    # ============================================================
    # ASCII VISUALIZATIONS (Terminal-friendly)
    # ============================================================

    def ascii_sparkline(self, values: List[float], width: int = 20) -> str:
        """Create a simple ASCII sparkline from values."""
        if not values:
            return " " * width

        chars = "▁▂▃▄▅▆▇█"
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val if max_val != min_val else 1

        # Sample values if too many
        if len(values) > width:
            step = len(values) / width
            values = [values[int(i * step)] for i in range(width)]
        elif len(values) < width:
            # Pad with last value
            values = values + [values[-1]] * (width - len(values))

        result = ""
        for v in values:
            normalized = (v - min_val) / val_range
            idx = min(int(normalized * (len(chars) - 1)), len(chars) - 1)
            result += chars[idx]

        return result

    def ascii_progress_bar(self, value: float, max_value: float = 1.0, width: int = 20, label: str = "") -> str:
        """Create an ASCII progress bar."""
        if max_value == 0:
            pct = 0
        else:
            pct = min(value / max_value, 1.0)

        filled = int(pct * width)
        empty = width - filled

        bar = "█" * filled + "░" * empty
        pct_str = f"{pct * 100:5.1f}%"

        if label:
            return f"{label}: [{bar}] {pct_str}"
        return f"[{bar}] {pct_str}"

    def ascii_trend_indicator(self, trend: str) -> str:
        """Get ASCII indicator for trend direction."""
        indicators = {
            "improving": "↑ ✓",
            "declining": "↓ ✗",
            "stable": "→ ○",
        }
        return indicators.get(trend, "? ○")

    def format_currency(self, value: float) -> str:
        """Format currency with color indicators."""
        if value >= 0:
            return f"+${value:,.2f}"
        return f"-${abs(value):,.2f}"

    def format_percentage(self, value: float) -> str:
        """Format percentage."""
        return f"{value * 100:.1f}%"

    # ============================================================
    # DASHBOARD DISPLAYS
    # ============================================================

    def display_dashboard(self, days: int = 30) -> str:
        """Generate a comprehensive ASCII dashboard of learning progress."""
        summary = self.log.get_learning_summary(days)
        trends = self.log.get_improvement_trends(periods=min(days, 30))

        lines = []
        lines.append("")
        lines.append("╔" + "═" * 58 + "╗")
        lines.append("║" + " AQTIS LEARNING DASHBOARD ".center(58) + "║")
        lines.append("║" + f" Last {days} Days ".center(58) + "║")
        lines.append("╠" + "═" * 58 + "╣")

        # Key Metrics
        stats = summary["epoch_stats"]
        lines.append("║  KEY METRICS" + " " * 45 + "║")
        lines.append("║" + "-" * 58 + "║")

        metrics = [
            ("Learning Epochs", f"{stats['total_epochs']}", ""),
            ("Total Trades", f"{stats['total_trades']:,}", ""),
            ("Win Rate", self.format_percentage(stats['avg_win_rate'] or 0),
             self.ascii_trend_indicator(trends['win_rate']['trend'])),
            ("Sharpe Ratio", f"{stats['avg_sharpe'] or 0:.2f}",
             self.ascii_trend_indicator(trends['sharpe_ratio']['trend'])),
            ("Total P&L", self.format_currency(stats['total_pnl'] or 0),
             self.ascii_trend_indicator(trends['pnl']['trend'])),
            ("Improvement Rate", self.format_percentage(summary['improvement_rate']), ""),
        ]

        for name, value, indicator in metrics:
            line = f"║  {name:<18} {value:>15} {indicator:>8}".ljust(58) + "  ║"
            lines.append(line)

        # Win Rate Trend
        lines.append("╠" + "═" * 58 + "╣")
        lines.append("║  WIN RATE TREND" + " " * 42 + "║")
        lines.append("║" + "-" * 58 + "║")

        sparkline = self.ascii_sparkline(trends['win_rate']['history'], width=40)
        lines.append(f"║  {sparkline}".ljust(58) + "  ║")

        min_wr = min(trends['win_rate']['history']) if trends['win_rate']['history'] else 0
        max_wr = max(trends['win_rate']['history']) if trends['win_rate']['history'] else 0
        lines.append(f"║  Min: {min_wr*100:.1f}%  Max: {max_wr*100:.1f}%  Avg: {trends['win_rate']['average']*100:.1f}%".ljust(58) + "  ║")

        # Sharpe Trend
        lines.append("╠" + "═" * 58 + "╣")
        lines.append("║  SHARPE RATIO TREND" + " " * 38 + "║")
        lines.append("║" + "-" * 58 + "║")

        sparkline = self.ascii_sparkline(trends['sharpe_ratio']['history'], width=40)
        lines.append(f"║  {sparkline}".ljust(58) + "  ║")

        min_sh = min(trends['sharpe_ratio']['history']) if trends['sharpe_ratio']['history'] else 0
        max_sh = max(trends['sharpe_ratio']['history']) if trends['sharpe_ratio']['history'] else 0
        lines.append(f"║  Min: {min_sh:.2f}  Max: {max_sh:.2f}  Avg: {trends['sharpe_ratio']['average']:.2f}".ljust(58) + "  ║")

        # Top Indicators
        lines.append("╠" + "═" * 58 + "╣")
        lines.append("║  TOP INDICATORS" + " " * 42 + "║")
        lines.append("║" + "-" * 58 + "║")

        for i, ind in enumerate(summary["top_indicators"][:5], 1):
            wr = ind['avg_win_rate'] or 0
            bar = self.ascii_progress_bar(wr, 1.0, width=15)
            line = f"║  {i}. {ind['indicator_name'][:15]:<15} {bar}".ljust(58) + "  ║"
            lines.append(line)

        # Recent Insights
        lines.append("╠" + "═" * 58 + "╣")
        lines.append("║  RECENT INSIGHTS" + " " * 41 + "║")
        lines.append("║" + "-" * 58 + "║")

        for insight in summary["recent_insights"][:3]:
            marker = "⚡" if insight["actionable"] else "📝"
            title = insight['title'][:50]
            lines.append(f"║  {marker} {title}".ljust(58) + "  ║")

        lines.append("╚" + "═" * 58 + "╝")

        return "\n".join(lines)

    def display_indicator_heatmap(self, regime: str = None) -> str:
        """Display indicator performance as an ASCII heatmap by regime."""
        analysis = self.log.get_indicator_analysis(market_regime=regime)

        if not analysis:
            return "No indicator data available."

        lines = []
        lines.append("")
        lines.append("╔" + "═" * 68 + "╗")
        lines.append("║" + " INDICATOR PERFORMANCE HEATMAP ".center(68) + "║")
        lines.append("╠" + "═" * 68 + "╣")

        # Header
        regimes = ["trending_up", "trending_down", "ranging", "volatile"]
        header = "║  Indicator        "
        for r in regimes:
            header += f" {r[:8]:>8}"
        header = header.ljust(68) + "  ║"
        lines.append(header)
        lines.append("║" + "-" * 68 + "║")

        # Performance blocks
        blocks = " ▁▂▃▄▅▆▇█"

        for ind_name, data in sorted(analysis.items())[:15]:
            row = f"║  {ind_name[:16]:<16}"

            for regime in regimes:
                regime_data = data.get("regimes", {}).get(regime, {})
                wr = regime_data.get("win_rate", 0)

                # Map win rate to block character
                idx = min(int(wr * (len(blocks) - 1)), len(blocks) - 1)
                block = blocks[idx]

                # Color indicator based on performance
                wr_pct = f"{wr*100:5.1f}%"
                row += f" {wr_pct:>8}"

            row = row.ljust(68) + "  ║"
            lines.append(row)

        lines.append("╚" + "═" * 68 + "╝")
        lines.append("  Legend: Win Rate % by Market Regime")

        return "\n".join(lines)

    def display_strategy_evolution(self, strategy_name: str = None) -> str:
        """Display strategy evolution over time."""
        trends = self.log.get_improvement_trends(strategy_name=strategy_name, periods=20)

        if "message" in trends:
            return trends["message"]

        lines = []
        lines.append("")
        lines.append("╔" + "═" * 58 + "╗")
        title = f" STRATEGY EVOLUTION: {strategy_name or 'ALL'} "
        lines.append("║" + title.center(58) + "║")
        lines.append("╠" + "═" * 58 + "╣")

        # Performance Evolution
        lines.append("║  P&L EVOLUTION" + " " * 43 + "║")
        lines.append("║" + "-" * 58 + "║")

        pnl_history = trends['pnl']['history']
        if pnl_history:
            # Create ASCII bar chart
            max_pnl = max(abs(p) for p in pnl_history) if pnl_history else 1
            for i, pnl in enumerate(pnl_history[-10:]):  # Last 10 epochs
                bar_width = int(abs(pnl) / max_pnl * 30) if max_pnl > 0 else 0

                if pnl >= 0:
                    bar = "▓" * bar_width
                    line = f"║  Epoch {i+1:2d} │{' '*15}{bar:<30} {self.format_currency(pnl):>10}"
                else:
                    bar = "▓" * bar_width
                    line = f"║  Epoch {i+1:2d} │{bar:>15}{' '*30} {self.format_currency(pnl):>10}"

                lines.append(line.ljust(58) + "  ║")

        # Summary stats
        lines.append("╠" + "═" * 58 + "╣")
        lines.append("║  SUMMARY" + " " * 49 + "║")
        lines.append("║" + "-" * 58 + "║")

        lines.append(f"║  Periods Analyzed: {trends['periods_analyzed']}".ljust(58) + "  ║")
        lines.append(f"║  Total P&L: {self.format_currency(trends['pnl']['total'])}".ljust(58) + "  ║")
        lines.append(f"║  Win Rate Trend: {trends['win_rate']['trend'].upper()} {self.ascii_trend_indicator(trends['win_rate']['trend'])}".ljust(58) + "  ║")
        lines.append(f"║  Sharpe Trend: {trends['sharpe_ratio']['trend'].upper()} {self.ascii_trend_indicator(trends['sharpe_ratio']['trend'])}".ljust(58) + "  ║")

        lines.append("╚" + "═" * 58 + "╝")

        return "\n".join(lines)

    def display_coach_analysis(self) -> str:
        """Display coach effectiveness analysis."""
        coach_eff = self.log.get_coach_effectiveness()

        lines = []
        lines.append("")
        lines.append("╔" + "═" * 58 + "╗")
        lines.append("║" + " COACH EFFECTIVENESS ANALYSIS ".center(58) + "║")
        lines.append("╠" + "═" * 58 + "╣")

        # With Coach
        with_coach = coach_eff.get("with_coach", {})
        lines.append("║  WITH COACH INTERVENTION" + " " * 33 + "║")
        lines.append("║" + "-" * 58 + "║")
        lines.append(f"║    Epochs: {with_coach.get('epochs', 0)}".ljust(58) + "  ║")
        lines.append(f"║    Avg Win Rate: {self.format_percentage(with_coach.get('avg_win_rate', 0))}".ljust(58) + "  ║")
        lines.append(f"║    Avg Sharpe: {with_coach.get('avg_sharpe', 0):.2f}".ljust(58) + "  ║")
        lines.append(f"║    Avg P&L: {self.format_currency(with_coach.get('avg_pnl', 0))}".ljust(58) + "  ║")

        # Without Coach
        without_coach = coach_eff.get("without_coach", {})
        lines.append("╠" + "═" * 58 + "╣")
        lines.append("║  WITHOUT COACH INTERVENTION" + " " * 30 + "║")
        lines.append("║" + "-" * 58 + "║")
        lines.append(f"║    Epochs: {without_coach.get('epochs', 0)}".ljust(58) + "  ║")
        lines.append(f"║    Avg Win Rate: {self.format_percentage(without_coach.get('avg_win_rate', 0))}".ljust(58) + "  ║")
        lines.append(f"║    Avg Sharpe: {without_coach.get('avg_sharpe', 0):.2f}".ljust(58) + "  ║")
        lines.append(f"║    Avg P&L: {self.format_currency(without_coach.get('avg_pnl', 0))}".ljust(58) + "  ║")

        # Impact
        impact = coach_eff.get("coach_impact", {})
        lines.append("╠" + "═" * 58 + "╣")
        lines.append("║  COACH IMPACT" + " " * 44 + "║")
        lines.append("║" + "-" * 58 + "║")

        wr_diff = impact.get("win_rate_diff", 0)
        sharpe_diff = impact.get("sharpe_diff", 0)
        pnl_diff = impact.get("pnl_diff", 0)

        wr_indicator = "✓" if wr_diff > 0 else "✗" if wr_diff < 0 else "○"
        sharpe_indicator = "✓" if sharpe_diff > 0 else "✗" if sharpe_diff < 0 else "○"
        pnl_indicator = "✓" if pnl_diff > 0 else "✗" if pnl_diff < 0 else "○"

        lines.append(f"║    Win Rate: {wr_diff*100:+.1f}% {wr_indicator}".ljust(58) + "  ║")
        lines.append(f"║    Sharpe: {sharpe_diff:+.2f} {sharpe_indicator}".ljust(58) + "  ║")
        lines.append(f"║    P&L: {self.format_currency(pnl_diff)} {pnl_indicator}".ljust(58) + "  ║")

        lines.append("╠" + "═" * 58 + "╣")
        conclusion = coach_eff.get("conclusion", "")
        # Word wrap conclusion
        words = conclusion.split()
        current_line = "║  "
        for word in words:
            if len(current_line) + len(word) + 1 < 58:
                current_line += word + " "
            else:
                lines.append(current_line.ljust(58) + "  ║")
                current_line = "║  " + word + " "
        if current_line.strip():
            lines.append(current_line.ljust(58) + "  ║")

        lines.append("╚" + "═" * 58 + "╝")

        return "\n".join(lines)

    def display_learning_timeline(self, days: int = 7) -> str:
        """Display a timeline of recent learning activity."""
        # Get epochs from log
        df = self.log.export_to_dataframe("learning_epochs", days=days)

        if df.empty:
            return "No learning data for the specified period."

        lines = []
        lines.append("")
        lines.append("╔" + "═" * 68 + "╗")
        lines.append("║" + f" LEARNING TIMELINE - Last {days} Days ".center(68) + "║")
        lines.append("╠" + "═" * 68 + "╣")

        # Group by date
        df['date'] = df['timestamp'].str[:10]

        for date in sorted(df['date'].unique())[-7:]:  # Last 7 days
            day_data = df[df['date'] == date]

            lines.append(f"║  📅 {date}".ljust(68) + "  ║")
            lines.append("║  " + "─" * 64 + "  ║")

            for _, row in day_data.iterrows():
                epoch_type = row['epoch_type']
                strategy = row['strategy_name'][:20]
                pnl = row['net_pnl']
                wr = row['win_rate']

                icon = "🔬" if epoch_type == "backtest" else "📄" if epoch_type == "paper_trade" else "💰"
                pnl_str = f"${pnl:+,.0f}" if pnl else "$0"
                wr_str = f"{wr*100:.0f}%" if wr else "N/A"

                line = f"║  {icon} {strategy:<20} P&L: {pnl_str:>10} WR: {wr_str:>5}"
                lines.append(line.ljust(68) + "  ║")

            lines.append("║" + " " * 68 + "║")

        lines.append("╚" + "═" * 68 + "╝")

        return "\n".join(lines)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def print_dashboard(db_path: str = "./learning_log.db", days: int = 30) -> None:
    """Print the learning dashboard to stdout."""
    log = LearningLog(db_path=db_path)
    viz = LearningVisualizer(log)
    print(viz.display_dashboard(days))


def print_report(db_path: str = "./learning_log.db", days: int = 30) -> None:
    """Print the full learning report to stdout."""
    log = LearningLog(db_path=db_path)
    print(log.generate_learning_report(days))
