"""
Paper Trading Performance Monitor
Tracks live positions, generates reports, and analyzes performance.
"""
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

STATE_FILE = "paper_trading_state.json"
PERFORMANCE_LOG = "paper_trading_performance.json"

def load_state():
    """Load current paper trading state."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return None

def load_performance_log():
    """Load historical performance data."""
    if os.path.exists(PERFORMANCE_LOG):
        with open(PERFORMANCE_LOG, 'r') as f:
            return json.load(f)
    return {"daily_snapshots": [], "trade_history": []}

def save_performance_log(data):
    """Save performance log."""
    with open(PERFORMANCE_LOG, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def get_current_prices(symbols):
    """Fetch current prices for all symbols."""
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                prices[symbol] = float(hist['Close'].iloc[-1])
        except:
            pass
    return prices

def calculate_portfolio_metrics(state, current_prices):
    """Calculate current portfolio metrics."""
    if not state or 'positions' not in state:
        return None

    initial_capital = 100000.0
    available = state.get('available_capital', initial_capital)

    positions_value = 0.0
    unrealized_pnl = 0.0
    position_details = []

    for symbol, pos in state['positions'].items():
        entry_price = pos['entry_price']
        quantity = pos['quantity']
        current_price = current_prices.get(symbol, entry_price)

        position_value = quantity * current_price
        positions_value += position_value

        pnl = (current_price - entry_price) * quantity
        pnl_pct = (current_price / entry_price - 1) * 100
        unrealized_pnl += pnl

        # Check stop loss
        stop_hit = current_price <= pos['stop_loss']

        position_details.append({
            'symbol': symbol,
            'direction': pos['direction'],
            'quantity': quantity,
            'entry_price': entry_price,
            'current_price': current_price,
            'stop_loss': pos['stop_loss'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'stop_hit': stop_hit,
            'entry_time': pos['entry_time']
        })

    # Calculate realized P&L from completed trades
    realized_pnl = sum(t.get('pnl', 0) for t in state.get('trades', []))

    total_equity = available + positions_value
    total_pnl = unrealized_pnl + realized_pnl
    total_return = (total_equity / initial_capital - 1) * 100

    return {
        'timestamp': datetime.now().isoformat(),
        'initial_capital': initial_capital,
        'available_capital': available,
        'positions_value': positions_value,
        'total_equity': total_equity,
        'unrealized_pnl': unrealized_pnl,
        'realized_pnl': realized_pnl,
        'total_pnl': total_pnl,
        'total_return_pct': total_return,
        'num_positions': len(state['positions']),
        'num_completed_trades': len(state.get('trades', [])),
        'position_details': position_details
    }

def generate_report(metrics):
    """Generate a formatted performance report."""
    if not metrics:
        return "No trading data available."

    report = []
    report.append("=" * 70)
    report.append("PAPER TRADING PERFORMANCE REPORT")
    report.append(f"Generated: {metrics['timestamp']}")
    report.append("=" * 70)

    report.append("\n📊 PORTFOLIO SUMMARY")
    report.append("-" * 40)
    report.append(f"Initial Capital:    ${metrics['initial_capital']:>12,.2f}")
    report.append(f"Available Capital:  ${metrics['available_capital']:>12,.2f}")
    report.append(f"Positions Value:    ${metrics['positions_value']:>12,.2f}")
    report.append(f"Total Equity:       ${metrics['total_equity']:>12,.2f}")
    report.append("-" * 40)
    report.append(f"Unrealized P&L:     ${metrics['unrealized_pnl']:>+12,.2f}")
    report.append(f"Realized P&L:       ${metrics['realized_pnl']:>+12,.2f}")
    report.append(f"Total P&L:          ${metrics['total_pnl']:>+12,.2f}")
    report.append(f"Total Return:       {metrics['total_return_pct']:>+12.2f}%")

    report.append(f"\n📈 OPEN POSITIONS ({metrics['num_positions']})")
    report.append("-" * 70)

    if metrics['position_details']:
        # Sort by P&L
        sorted_positions = sorted(metrics['position_details'], key=lambda x: x['pnl'], reverse=True)

        for pos in sorted_positions:
            status = "🔴 STOP HIT!" if pos['stop_hit'] else "🟢"
            pnl_str = f"${pos['pnl']:+,.0f}" if abs(pos['pnl']) >= 1 else f"${pos['pnl']:+.2f}"
            report.append(
                f"  {status} {pos['symbol']:<14} {pos['direction']:<5} "
                f"Qty: {pos['quantity']:>4} | Entry: ${pos['entry_price']:>8,.2f} | "
                f"Now: ${pos['current_price']:>8,.2f} | P&L: {pnl_str} ({pos['pnl_pct']:+.1f}%)"
            )
            report.append(f"       Stop Loss: ${pos['stop_loss']:,.2f} | Entry: {pos['entry_time'][:19]}")

    # Completed trades summary
    report.append(f"\n📜 COMPLETED TRADES: {metrics['num_completed_trades']}")

    report.append("\n" + "=" * 70)

    return "\n".join(report)

def analyze_performance_history(perf_log):
    """Analyze historical performance."""
    if not perf_log.get('daily_snapshots'):
        return "No historical data available yet."

    snapshots = perf_log['daily_snapshots']

    if len(snapshots) < 2:
        return "Need at least 2 snapshots for analysis."

    # Extract equity curve
    dates = [s['timestamp'][:10] for s in snapshots]
    equities = [s['total_equity'] for s in snapshots]
    returns = [s['total_return_pct'] for s in snapshots]

    # Calculate metrics
    peak = max(equities)
    current = equities[-1]
    drawdown = (peak - current) / peak * 100 if peak > 0 else 0

    report = []
    report.append("\n📈 PERFORMANCE HISTORY")
    report.append("-" * 40)
    report.append(f"First Snapshot: {dates[0]}")
    report.append(f"Latest Snapshot: {dates[-1]}")
    report.append(f"Total Snapshots: {len(snapshots)}")
    report.append(f"Peak Equity: ${peak:,.2f}")
    report.append(f"Current Drawdown: {drawdown:.2f}%")

    if len(returns) > 1:
        daily_returns = [returns[i] - returns[i-1] for i in range(1, len(returns))]
        avg_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns) if len(daily_returns) > 1 else 0
        sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

        report.append(f"Avg Daily Return: {avg_daily:.3f}%")
        report.append(f"Daily Volatility: {std_daily:.3f}%")
        report.append(f"Sharpe Ratio (est): {sharpe:.2f}")

    return "\n".join(report)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Paper Trading Monitor")
    parser.add_argument('--snapshot', action='store_true', help='Take daily snapshot')
    parser.add_argument('--history', action='store_true', help='Show performance history')
    args = parser.parse_args()

    state = load_state()
    if not state:
        print("No paper trading state found. Run live_paper_trader.py first.")
        return

    # Get current prices
    symbols = list(state.get('positions', {}).keys())
    print(f"Fetching prices for {len(symbols)} positions...")
    current_prices = get_current_prices(symbols)

    # Calculate metrics
    metrics = calculate_portfolio_metrics(state, current_prices)

    if args.snapshot:
        # Take snapshot and save to log
        perf_log = load_performance_log()
        snapshot = {
            'timestamp': metrics['timestamp'],
            'total_equity': metrics['total_equity'],
            'unrealized_pnl': metrics['unrealized_pnl'],
            'realized_pnl': metrics['realized_pnl'],
            'total_pnl': metrics['total_pnl'],
            'total_return_pct': metrics['total_return_pct'],
            'num_positions': metrics['num_positions']
        }
        perf_log['daily_snapshots'].append(snapshot)
        save_performance_log(perf_log)
        print("📸 Snapshot saved!")

    # Generate and print report
    report = generate_report(metrics)
    print(report)

    if args.history:
        perf_log = load_performance_log()
        history = analyze_performance_history(perf_log)
        print(history)

if __name__ == '__main__':
    main()
