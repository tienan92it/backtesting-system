#!/usr/bin/env python
"""
Example script demonstrating how to use the performance metrics module
to analyze backtesting results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting.metrics import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_performance_metrics,
    calculate_trade_metrics
)


def create_sample_equity_curve(days=365, mean_return=0.0005, std_dev=0.01, initial_value=10000):
    """
    Create a sample equity curve for demonstration purposes.
    
    Parameters:
    -----------
    days : int
        Number of days in the equity curve.
    mean_return : float
        Mean daily return.
    std_dev : float
        Standard deviation of daily returns.
    initial_value : float
        Initial portfolio value.
    
    Returns:
    --------
    pd.Series
        Sample equity curve.
    """
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(days)]
    values = [initial_value]
    
    # Generate a random walk with specified drift
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(mean_return, std_dev, days - 1)
    
    for ret in returns:
        values.append(values[-1] * (1 + ret))
    
    return pd.Series(values, index=dates)


def create_sample_trades(equity_curve, n_trades=50, win_rate=0.55):
    """
    Create sample trades for demonstration purposes.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Equity curve to base trades on.
    n_trades : int
        Number of trades to generate.
    win_rate : float
        Target win rate.
    
    Returns:
    --------
    pd.DataFrame
        Sample trades.
    """
    # Generate random trade dates within the equity curve timeframe
    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]
    
    # Generate random timestamps for trades
    np.random.seed(42)  # For reproducibility
    trade_days = np.random.randint(0, len(equity_curve) - 1, n_trades)
    trade_days.sort()  # Sort to ensure chronological order
    
    timestamps = [equity_curve.index[i] for i in trade_days]
    
    # Generate random profits with specified win rate
    np.random.seed(42)  # For reproducibility
    is_win = np.random.random(n_trades) < win_rate
    
    # Generate profit values
    profits = []
    for win in is_win:
        if win:
            # Winning trade: 1-3% profit
            profit = np.random.uniform(0.01, 0.03) * 10000
        else:
            # Losing trade: 1-2% loss
            profit = -np.random.uniform(0.01, 0.02) * 10000
        profits.append(profit)
    
    # Create trades DataFrame
    trades = pd.DataFrame({
        'timestamp': timestamps,
        'symbol': ['BTCUSDT'] * n_trades,
        'direction': np.where(np.random.random(n_trades) > 0.5, 'BUY', 'SELL'),
        'price': np.random.uniform(9000, 11000, n_trades),
        'quantity': np.random.uniform(0.1, 1.0, n_trades),
        'commission': np.random.uniform(1, 5, n_trades),
        'profit': profits
    })
    
    trades.set_index('timestamp', inplace=True)
    return trades


def plot_equity_curve_with_drawdowns(equity_curve):
    """
    Plot equity curve with drawdowns highlighted.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Equity curve to plot.
    """
    # Calculate running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdowns
    drawdowns = (equity_curve - running_max) / running_max
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(equity_curve.index, equity_curve, label='Equity Curve')
    ax1.plot(running_max.index, running_max, 'r--', label='Running Maximum')
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Portfolio Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot drawdowns
    ax2.fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3)
    ax2.plot(drawdowns.index, drawdowns, 'r', label='Drawdown')
    ax2.set_title('Drawdowns')
    ax2.set_ylabel('Drawdown %')
    ax2.set_ylim(min(drawdowns) * 1.1, 0.01)  # Add some padding
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('equity_curve_with_drawdowns.png')
    plt.close()


def plot_trade_distribution(trades):
    """
    Plot trade profit distribution.
    
    Parameters:
    -----------
    trades : pd.DataFrame
        Trades to analyze.
    """
    profits = trades['profit']
    
    plt.figure(figsize=(12, 6))
    
    # Plot histogram
    plt.hist(profits, bins=20, alpha=0.7, color='blue')
    plt.axvline(x=0, color='r', linestyle='--', label='Breakeven')
    plt.axvline(x=profits.mean(), color='g', linestyle='-', label=f'Mean: {profits.mean():.2f}')
    
    plt.title('Trade Profit Distribution')
    plt.xlabel('Profit')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('trade_distribution.png')
    plt.close()


def print_performance_report(equity_curve, trades):
    """
    Print a comprehensive performance report.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Equity curve to analyze.
    trades : pd.DataFrame
        Trades to analyze.
    """
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(equity_curve, trades)
    trade_metrics = calculate_trade_metrics(trades)
    
    # Print report
    print("\n" + "="*50)
    print(" "*15 + "PERFORMANCE REPORT")
    print("="*50)
    
    print("\nPORTFOLIO METRICS:")
    print(f"Initial Capital: ${equity_curve.iloc[0]:,.2f}")
    print(f"Final Capital: ${equity_curve.iloc[-1]:,.2f}")
    print(f"Total Return: {performance_metrics['total_return']*100:.2f}%")
    print(f"Annualized Return: {performance_metrics['annualized_return']*100:.2f}%")
    print(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {performance_metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {performance_metrics['max_drawdown']*100:.2f}%")
    print(f"Calmar Ratio: {performance_metrics['calmar_ratio']:.2f}")
    print(f"Volatility (Annualized): {performance_metrics['volatility']*100:.2f}%")
    
    print("\nTRADE METRICS:")
    print(f"Total Trades: {trade_metrics['total_trades']}")
    print(f"Win Rate: {trade_metrics['win_rate']*100:.2f}%")
    print(f"Profit Factor: {trade_metrics['profit_factor']:.2f}")
    print(f"Average Trade: ${trade_metrics['avg_trade']:,.2f}")
    print(f"Average Win: ${trade_metrics['avg_win']:,.2f}")
    print(f"Average Loss: ${trade_metrics['avg_loss']:,.2f}")
    print(f"Largest Win: ${trade_metrics['largest_win']:,.2f}")
    print(f"Largest Loss: ${trade_metrics['largest_loss']:,.2f}")
    
    print("="*50)


def run_example():
    """Run the performance metrics example."""
    print("Generating sample data...")
    
    # Create sample data
    equity_curve = create_sample_equity_curve(days=365, mean_return=0.0008, std_dev=0.015)
    trades = create_sample_trades(equity_curve, n_trades=100, win_rate=0.6)
    
    # Plot equity curve with drawdowns
    print("Plotting equity curve with drawdowns...")
    plot_equity_curve_with_drawdowns(equity_curve)
    
    # Plot trade distribution
    print("Plotting trade distribution...")
    plot_trade_distribution(trades)
    
    # Print performance report
    print_performance_report(equity_curve, trades)
    
    print(f"\nPlots saved to:")
    print(f"  - {os.path.abspath('equity_curve_with_drawdowns.png')}")
    print(f"  - {os.path.abspath('trade_distribution.png')}")


if __name__ == "__main__":
    run_example() 