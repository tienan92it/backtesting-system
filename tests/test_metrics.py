"""
Tests for the performance metrics module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtesting.metrics import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_performance_metrics,
    calculate_trade_metrics
)


class TestPerformanceMetrics(unittest.TestCase):
    """Test case for performance metrics functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample equity curve
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        values = [100.0]
        
        # Generate a random walk with stronger positive drift
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.01, 99)  # Increased mean return
        for ret in returns:
            values.append(values[-1] * (1 + ret))
        
        self.equity_curve = pd.Series(values, index=dates)
        
        # Create sample trades with clearer win/loss pattern
        self.trades = pd.DataFrame({
            'timestamp': [
                datetime(2020, 1, 10),
                datetime(2020, 1, 20),
                datetime(2020, 1, 30),
                datetime(2020, 2, 10),
                datetime(2020, 2, 20)
            ],
            'symbol': ['BTCUSDT'] * 5,
            'direction': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY'],
            'price': [100.0, 110.0, 105.0, 95.0, 100.0],
            'quantity': [1.0, 1.0, 1.0, 1.0, 1.0],
            'commission': [0.1, 0.1, 0.1, 0.1, 0.1],
            'profit': [0.0, 9.9, 0.0, -10.1, 0.0]  # Profit after commission
        })
        
        # Filter to only include completed trades (with profit/loss)
        self.trades = self.trades[self.trades['profit'] != 0.0]
        self.trades.set_index('timestamp', inplace=True)
    
    def test_calculate_returns(self):
        """Test calculate_returns function."""
        returns = calculate_returns(self.equity_curve)
        
        # Check that returns have the correct length
        self.assertEqual(len(returns), len(self.equity_curve))
        
        # Check that the first return is 0
        self.assertEqual(returns.iloc[0], 0.0)
        
        # Check that returns are calculated correctly
        for i in range(1, len(self.equity_curve)):
            expected_return = self.equity_curve.iloc[i] / self.equity_curve.iloc[i-1] - 1
            self.assertAlmostEqual(returns.iloc[i], expected_return)
    
    def test_calculate_sharpe_ratio(self):
        """Test calculate_sharpe_ratio function."""
        returns = calculate_returns(self.equity_curve)
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
        
        # Check that Sharpe ratio is positive (since we have positive drift)
        self.assertGreater(sharpe, 0)
    
    def test_calculate_max_drawdown(self):
        """Test calculate_max_drawdown function."""
        max_dd, peak_date, valley_date = calculate_max_drawdown(self.equity_curve)
        
        # Check that max drawdown is negative
        self.assertLessEqual(max_dd, 0)
        
        # Check that peak date is before valley date
        if peak_date is not None and valley_date is not None:
            self.assertLess(peak_date, valley_date)
    
    def test_calculate_win_rate(self):
        """Test calculate_win_rate function."""
        win_rate = calculate_win_rate(self.trades)
        
        # Expected win rate: 1 win, 1 loss out of 2 completed trades
        expected_win_rate = 0.5
        self.assertAlmostEqual(win_rate, expected_win_rate, places=2)
    
    def test_calculate_profit_factor(self):
        """Test calculate_profit_factor function."""
        profit_factor = calculate_profit_factor(self.trades)
        
        # Expected profit factor: 9.9 / 10.1 ≈ 0.98
        expected_profit_factor = 9.9 / 10.1
        self.assertAlmostEqual(profit_factor, expected_profit_factor, places=2)
    
    def test_calculate_performance_metrics(self):
        """Test calculate_performance_metrics function."""
        metrics = calculate_performance_metrics(self.equity_curve, self.trades)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'total_return', 'annualized_return', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'win_rate', 'profit_factor'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check that total return is calculated correctly
        expected_total_return = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
        self.assertAlmostEqual(metrics['total_return'], expected_total_return)
    
    def test_calculate_trade_metrics(self):
        """Test calculate_trade_metrics function."""
        metrics = calculate_trade_metrics(self.trades)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'total_trades', 'win_rate', 'profit_factor', 'avg_trade',
            'avg_win', 'avg_loss', 'largest_win', 'largest_loss'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check that total trades is calculated correctly
        self.assertEqual(metrics['total_trades'], len(self.trades))
        
        # Check that win rate is calculated correctly
        self.assertAlmostEqual(metrics['win_rate'], 0.5, places=2)


if __name__ == '__main__':
    unittest.main() 