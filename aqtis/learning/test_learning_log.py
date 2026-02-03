#!/usr/bin/env python3
"""
Test script for the AQTIS Learning Log system.

Demonstrates:
1. Recording learning epochs from backtests
2. Tracking indicator performance by regime
3. Monitoring strategy evolution
4. Generating learning reports and visualizations
5. Analyzing coach effectiveness
"""

import sys
import os
import tempfile
import shutil
import random
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def generate_mock_epoch_data(epoch_num: int, improving: bool = True) -> dict:
    """Generate mock epoch data with optional improvement trend."""
    base_win_rate = 0.45 + (epoch_num * 0.01 if improving else -epoch_num * 0.005)
    base_sharpe = 0.8 + (epoch_num * 0.05 if improving else -epoch_num * 0.03)
    base_pnl = 500 + (epoch_num * 100 if improving else -epoch_num * 50)

    # Add some noise
    win_rate = min(0.75, max(0.3, base_win_rate + random.uniform(-0.05, 0.05)))
    sharpe = max(0.1, base_sharpe + random.uniform(-0.2, 0.2))
    pnl = base_pnl + random.uniform(-200, 200)

    total_trades = random.randint(50, 150)
    winning_trades = int(total_trades * win_rate)

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
        "net_pnl": pnl,
        "sharpe_ratio": sharpe,
        "max_drawdown": random.uniform(0.05, 0.20),
        "profit_factor": 1.0 + (sharpe * 0.3),
    }


def generate_mock_indicator_performance() -> dict:
    """Generate mock indicator performance data."""
    indicators = [
        ("RSI_14", "momentum"),
        ("MACD_12_26_9", "momentum"),
        ("SMA_20", "trend"),
        ("EMA_50", "trend"),
        ("BB_20_2", "volatility"),
        ("ATR_14", "volatility"),
        ("OBV", "volume"),
        ("ADX_14", "trend"),
        ("STOCH_14_3_3", "momentum"),
        ("CCI_20", "momentum"),
    ]

    return {
        name: {
            "category": category,
            "win_rate": random.uniform(0.35, 0.70),
            "avg_pnl": random.uniform(-100, 300),
            "signal_accuracy": random.uniform(0.4, 0.8),
            "signal_count": random.randint(10, 100),
            "weight": random.uniform(0.05, 0.20),
        }
        for name, category in indicators
    }


def test_learning_log():
    """Test the core LearningLog functionality."""
    print("\n" + "=" * 60)
    print("  Testing LearningLog Core Functionality")
    print("=" * 60)

    from aqtis.learning import LearningLog

    # Use temp directory for test database
    test_dir = tempfile.mkdtemp(prefix="learning_test_")
    db_path = os.path.join(test_dir, "test_learning.db")

    try:
        log = LearningLog(db_path=db_path)
        print("✓ LearningLog initialized")

        # Record multiple epochs with improving trend
        print("\nRecording 15 learning epochs...")
        for i in range(15):
            data = generate_mock_epoch_data(i, improving=True)
            indicator_perf = generate_mock_indicator_performance()

            regimes = ["trending_up", "trending_down", "ranging", "volatile"]

            epoch_id = log.record_epoch(
                epoch_type="backtest",
                strategy_name="SuperIndicator_v1",
                strategy_version=1,
                total_trades=data["total_trades"],
                winning_trades=data["winning_trades"],
                net_pnl=data["net_pnl"],
                sharpe_ratio=data["sharpe_ratio"],
                max_drawdown=data["max_drawdown"],
                profit_factor=data["profit_factor"],
                market_regime=random.choice(regimes),
                coach_applied=(i > 5 and i % 3 == 0),  # Coach every 3rd epoch after 5
                top_indicators=["RSI_14", "MACD_12_26_9", "ADX_14"],
                worst_indicators=["OBV", "CCI_20"],
                indicator_performance=indicator_perf,
                symbols_traded=["AAPL", "MSFT", "GOOGL"],
            )
            print(f"  Epoch {epoch_id}: PnL=${data['net_pnl']:.0f}, Sharpe={data['sharpe_ratio']:.2f}")

        print("✓ Epochs recorded")

        # Record some strategy versions
        print("\nRecording strategy versions...")
        for v in range(1, 4):
            log.record_strategy_version(
                strategy_id="SI_001",
                strategy_name="SuperIndicator_v1",
                version=v,
                config={"entry_threshold": 0.6 + v * 0.05, "exit_threshold": 0.4 - v * 0.02},
                win_rate=0.5 + v * 0.05,
                sharpe_ratio=1.0 + v * 0.3,
                net_pnl=1000 + v * 500,
                max_drawdown=0.15 - v * 0.02,
                changes_made=f"Updated thresholds in version {v}",
            )
        print("✓ Strategy versions recorded")

        # Record some predictions
        print("\nRecording model predictions...")
        for _ in range(20):
            predicted = random.uniform(-0.02, 0.02)
            actual = predicted + random.uniform(-0.01, 0.01)
            log.record_prediction(
                model_name="SuperIndicator_v1",
                prediction_type="return",
                predicted_value=predicted,
                actual_value=actual,
                confidence=random.uniform(0.5, 0.9),
                market_regime=random.choice(regimes),
            )
        print("✓ Predictions recorded")

        # Test query methods
        print("\n--- Testing Query Methods ---")

        # Get learning summary
        summary = log.get_learning_summary(days=30)
        print(f"\nLearning Summary (30 days):")
        print(f"  Total epochs: {summary['epoch_stats']['total_epochs']}")
        print(f"  Total trades: {summary['epoch_stats']['total_trades']}")
        print(f"  Avg win rate: {summary['epoch_stats']['avg_win_rate']*100:.1f}%")
        print(f"  Total PnL: ${summary['epoch_stats']['total_pnl']:.2f}")
        print(f"  Improvement rate: {summary['improvement_rate']*100:.1f}%")

        # Get improvement trends
        trends = log.get_improvement_trends(periods=15)
        print(f"\nImprovement Trends:")
        print(f"  Win rate trend: {trends['win_rate']['trend']}")
        print(f"  Sharpe trend: {trends['sharpe_ratio']['trend']}")
        print(f"  PnL trend: {trends['pnl']['trend']}")

        # Get indicator analysis
        ind_analysis = log.get_indicator_analysis()
        print(f"\nIndicator Analysis:")
        print(f"  Indicators tracked: {len(ind_analysis)}")
        if ind_analysis:
            top_ind = list(ind_analysis.keys())[0]
            print(f"  Example ({top_ind}): {ind_analysis[top_ind]}")

        # Get coach effectiveness
        coach_eff = log.get_coach_effectiveness()
        print(f"\nCoach Effectiveness:")
        print(f"  Conclusion: {coach_eff['conclusion']}")

        # Get all insights
        insights = log.get_all_insights()
        print(f"\nInsights Generated: {len(insights)}")
        for insight in insights[:3]:
            print(f"  - {insight['title']}")

        print("\n✓ All query methods working")

        return log, db_path, test_dir

    except Exception as e:
        shutil.rmtree(test_dir, ignore_errors=True)
        raise e


def test_visualizer(log, db_path):
    """Test the LearningVisualizer."""
    print("\n" + "=" * 60)
    print("  Testing LearningVisualizer")
    print("=" * 60)

    from aqtis.learning import LearningVisualizer

    viz = LearningVisualizer(log)
    print("✓ LearningVisualizer initialized")

    # Test dashboard
    print("\n--- Dashboard ---")
    dashboard = viz.display_dashboard(days=30)
    print(dashboard)

    # Test strategy evolution
    print("\n--- Strategy Evolution ---")
    evolution = viz.display_strategy_evolution("SuperIndicator_v1")
    print(evolution)

    # Test coach analysis
    print("\n--- Coach Analysis ---")
    coach = viz.display_coach_analysis()
    print(coach)

    # Test learning timeline
    print("\n--- Learning Timeline ---")
    timeline = viz.display_learning_timeline(days=7)
    print(timeline)

    print("\n✓ All visualizations working")


def test_integration():
    """Test the LearningIntegration with mock data."""
    print("\n" + "=" * 60)
    print("  Testing LearningIntegration")
    print("=" * 60)

    from aqtis.learning import LearningIntegration

    test_dir = tempfile.mkdtemp(prefix="integration_test_")
    db_path = os.path.join(test_dir, "integration_test.db")

    try:
        integration = LearningIntegration(db_path=db_path)
        print("✓ LearningIntegration initialized")

        # Mock backtest result
        class MockBacktestResult:
            def __init__(self):
                self.strategy_name = "TestStrategy"
                self.total_trades = 100
                self.winning_trades = 55
                self.net_pnl = 2500.0
                self.sharpe_ratio = 1.5
                self.max_drawdown_pct = 0.12
                self.profit_factor = 1.8
                self.start_date = "2024-01-01"
                self.end_date = "2024-03-01"
                self.trades = []

            def to_dict(self):
                return vars(self)

        result = MockBacktestResult()
        epoch_id = integration.record_backtest_result(
            result,
            indicator_performance=generate_mock_indicator_performance(),
        )
        print(f"✓ Backtest result recorded as epoch {epoch_id}")

        # Record coach intervention
        epoch_id = integration.record_coach_intervention(
            strategy_name="TestStrategy",
            strategy_version=2,
            changes={"RSI_14_weight": "0.15 -> 0.20", "entry_threshold": "0.65 -> 0.70"},
            previous_metrics={"sharpe_ratio": 1.2, "win_rate": 0.50},
            new_metrics={"sharpe_ratio": 1.5, "win_rate": 0.55},
        )
        print(f"✓ Coach intervention recorded as epoch {epoch_id}")

        # Record paper trade session
        mock_trades = [
            {"symbol": "AAPL", "net_pnl": 150, "market_regime": "trending_up"},
            {"symbol": "MSFT", "net_pnl": -50, "market_regime": "trending_up"},
            {"symbol": "GOOGL", "net_pnl": 200, "market_regime": "ranging"},
        ]
        epoch_id = integration.record_paper_trade_session(
            trades=mock_trades,
            strategy_name="TestStrategy",
        )
        print(f"✓ Paper trade session recorded as epoch {epoch_id}")

        # Get learning status
        status = integration.get_learning_status()
        print(f"\nLearning Status:")
        print(f"  Is improving: {status['is_improving']}")
        print(f"  Recommendation: {status['recommendation']}")

        print("\n✓ Integration tests passed")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_report_generation(log):
    """Test report generation."""
    print("\n" + "=" * 60)
    print("  Testing Report Generation")
    print("=" * 60)

    report = log.generate_learning_report(days=30)
    print(report)

    print("\n✓ Report generation working")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  AQTIS LEARNING LOG - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    try:
        # Test core functionality
        log, db_path, test_dir = test_learning_log()

        # Test visualizer
        test_visualizer(log, db_path)

        # Test report generation
        test_report_generation(log)

        # Cleanup core test
        shutil.rmtree(test_dir, ignore_errors=True)

        # Test integration
        test_integration()

        print("\n" + "=" * 70)
        print("  ALL TESTS PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
