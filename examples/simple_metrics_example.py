#!/usr/bin/env python
"""
Simple example demonstrating how to use the performance metrics module
with a basic trading strategy.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting.metrics import (
    calculate_performance_metrics,
    calculate_trade_metrics,
    calculate_returns
)


def create_sample_data(periods=500, freq='D'):
    """
    Create sample price data for demonstration purposes.
    
    Parameters:
    -----------
    periods : int
        Number of periods to generate.
    freq : str
        Frequency of the data (e.g., 'D' for daily, 'H' for hourly).
    
    Returns:
    --------
    pd.DataFrame
        Sample price data.
    """
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=periods if freq == 'D' else periods//24)
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate price data with a sine wave component for cyclicality
    t = np.linspace(0, 10, len(dates))
    close_prices = 100 + 10 * np.sin(t) + np.cumsum(np.random.normal(0, 0.5, len(dates)))
    
    # Generate OHLCV data
    data = pd.DataFrame(index=dates)
    data['open'] = close_prices - np.random.normal(0, 1, len(dates))
    data['high'] = np.maximum(close_prices + np.random.normal(0, 1, len(dates)), data['open'])
    data['low'] = np.minimum(close_prices - np.random.normal(0, 1, len(dates)), data['open'])
    data['close'] = close_prices
    data['volume'] = np.random.lognormal(10, 1, len(dates))
    
    return data


def simple_moving_average_strategy(data, short_window=20, long_window=50):
    """
    Simple moving average crossover strategy.
    
    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data.
    short_window : int
        Short moving average window.
    long_window : int
        Long moving average window.
    
    Returns:
    --------
    pd.DataFrame
        Data with signals added.
    """
    # Make a copy of the data
    signals = data.copy()
    
    # Calculate moving averages
    signals['short_ma'] = signals['close'].rolling(window=short_window).mean()
    signals['long_ma'] = signals['close'].rolling(window=long_window).mean()
    
    # Calculate signals
    signals['signal'] = 0.0
    
    # Fill NaN values to avoid comparison issues
    signals['short_ma'] = signals['short_ma'].fillna(0)
    signals['long_ma'] = signals['long_ma'].fillna(0)
    
    # Calculate crossover as boolean
    signals['crossover'] = signals['short_ma'] > signals['long_ma']
    signals['crossover_prev'] = signals['crossover'].shift(1).fillna(False)
    
    # Generate buy/sell signals
    buy_signals = signals['crossover'] & ~signals['crossover_prev']
    sell_signals = ~signals['crossover'] & signals['crossover_prev']
    
    signals.loc[buy_signals, 'signal'] = 1.0  # Buy signal
    signals.loc[sell_signals, 'signal'] = -1.0  # Sell signal
    
    # Calculate positions (1 = long, 0 = cash, -1 = short)
    signals['position'] = 0.0
    signals['position'] = signals['signal'].replace(to_replace=0.0, method='ffill')
    signals['position'] = signals['position'].fillna(0.0)
    
    return signals


def backtest_strategy(signals, initial_capital=10000.0, position_size=1.0):
    """
    Backtest a strategy based on signals.
    
    Parameters:
    -----------
    signals : pd.DataFrame
        Data with signals added.
    initial_capital : float
        Initial capital.
    position_size : float
        Position size as a fraction of capital.
    
    Returns:
    --------
    tuple
        (portfolio, trades)
    """
    # Make a copy of the signals
    portfolio = signals.copy()
    
    # Calculate returns
    portfolio['returns'] = portfolio['close'].pct_change()
    
    # Calculate strategy returns
    portfolio['strategy_returns'] = portfolio['position'].shift(1) * portfolio['returns']
    
    # Calculate portfolio value
    portfolio['portfolio_value'] = initial_capital * (1 + portfolio['strategy_returns']).cumprod()
    
    # Generate trades
    trades = []
    
    # Find all buy and sell signals
    buy_signals = portfolio[portfolio['signal'] == 1.0]
    sell_signals = portfolio[portfolio['signal'] == -1.0]
    
    # Process buy signals
    for idx, row in buy_signals.iterrows():
        # Calculate position size
        price = row['close']
        quantity = (initial_capital * position_size) / price
        
        # Add trade
        trades.append({
            'timestamp': idx,
            'symbol': 'SAMPLE',
            'direction': 'BUY',
            'price': price,
            'quantity': quantity,
            'commission': price * quantity * 0.001,  # 0.1% commission
            'profit': 0.0  # Profit will be calculated later
        })
    
    # Process sell signals
    for i, (idx, row) in enumerate(sell_signals.iterrows()):
        if i >= len(buy_signals):
            break
        
        # Get the corresponding buy trade
        buy_trade = trades[i]
        
        # Calculate profit
        price = row['close']
        quantity = buy_trade['quantity']
        profit = (price - buy_trade['price']) * quantity - buy_trade['commission'] - (price * quantity * 0.001)
        
        # Add trade
        trades.append({
            'timestamp': idx,
            'symbol': 'SAMPLE',
            'direction': 'SELL',
            'price': price,
            'quantity': quantity,
            'commission': price * quantity * 0.001,  # 0.1% commission
            'profit': profit
        })
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df.set_index('timestamp', inplace=True)
    
    return portfolio, trades_df


def plot_strategy_results(data, signals, portfolio):
    """
    Plot strategy results.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Original OHLCV data.
    signals : pd.DataFrame
        Data with signals added.
    portfolio : pd.DataFrame
        Portfolio data.
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price and moving averages
    ax1.plot(data.index, data['close'], label='Price')
    ax1.plot(signals.index, signals['short_ma'], label='Short MA')
    ax1.plot(signals.index, signals['long_ma'], label='Long MA')
    
    # Plot buy and sell signals
    buy_signals = signals[signals['signal'] == 1.0]
    sell_signals = signals[signals['signal'] == -1.0]
    
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', s=100, label='Buy')
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', s=100, label='Sell')
    
    # Plot portfolio value
    ax2.plot(portfolio.index, portfolio['portfolio_value'], label='Portfolio Value')
    
    # Add labels and legend
    ax1.set_title('Moving Average Crossover Strategy')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Portfolio Value')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_strategy_results.png')
    plt.close()


def run_example():
    """Run the simple metrics example."""
    print("Creating sample data...")
    data = create_sample_data(periods=500)
    
    print("Running strategy...")
    signals = simple_moving_average_strategy(data, short_window=20, long_window=50)
    
    print("Backtesting strategy...")
    portfolio, trades = backtest_strategy(signals, initial_capital=10000.0, position_size=0.5)
    
    print("Plotting results...")
    plot_strategy_results(data, signals, portfolio)
    
    print("Calculating performance metrics...")
    if not portfolio.empty:
        equity_curve = portfolio['portfolio_value']
        
        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(equity_curve, trades)
        trade_metrics = calculate_trade_metrics(trades)
        
        # Print report
        print("\n" + "="*50)
        print(" "*15 + "PERFORMANCE REPORT")
        print("="*50)
        
        print("\nSTRATEGY PARAMETERS:")
        print(f"Short Window: 20")
        print(f"Long Window: 50")
        
        print("\nPORTFOLIO METRICS:")
        print(f"Initial Capital: $10,000.00")
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
    else:
        print("No portfolio values or trades generated.")
    
    print(f"\nPlot saved to: {os.path.abspath('simple_strategy_results.png')}")


if __name__ == "__main__":
    run_example() 