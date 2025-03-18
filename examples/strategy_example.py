import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

from backtesting.data import BinanceDataHandler
from backtesting.strategy import MovingAverageCrossStrategy, RSIStrategy, MLMeanReversionStrategy


def create_mock_data(periods=500):
    """
    Create mock price data for demonstration.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with mock OHLCV data.
    """
    dates = pd.date_range(start='2020-01-01', periods=periods, freq='D')
    
    # Create a price series with trend and noise
    trend = np.linspace(0, 20, periods)
    cycle = np.sin(np.linspace(0, 10, periods)) * 10
    noise = np.random.normal(0, 2, periods)
    
    close_prices = 100 + trend + cycle + noise
    
    # Create OHLCV data
    data = {
        'open': close_prices - np.random.uniform(0, 2, periods),
        'high': close_prices + np.random.uniform(1, 3, periods),
        'low': close_prices - np.random.uniform(1, 3, periods),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, periods)
    }
    
    return pd.DataFrame(data, index=dates)


def simulate_strategy(strategy, data):
    """
    Simulate a strategy on historical data.
    
    Parameters:
    -----------
    strategy : Strategy
        The strategy to test.
    data : pd.DataFrame
        Historical price data.
        
    Returns:
    --------
    tuple
        (signals, portfolio_values, trades)
    """
    # Set the data for the strategy
    strategy.set_data(data)
    
    # Variables to track portfolio
    initial_capital = 10000.0
    cash = initial_capital
    position = 0
    position_size = 0
    portfolio_values = []
    
    # Lists to store trades and signals
    trades = []
    
    # Mock buy and sell functions
    def buy_func(size=None, price=None, limit_price=None, stop_price=None, percent=None):
        nonlocal cash, position, position_size
        
        # Determine price
        current_price = price if price is not None else data['close'].iloc[strategy.current_index]
        
        # Determine size
        if size is not None:
            shares = size
        elif percent is not None:
            shares = (cash * percent / 100) / current_price
        else:
            shares = cash / current_price
        
        # Execute trade
        cost = shares * current_price
        if cost <= cash:
            cash -= cost
            position = 1
            position_size += shares
            
            trade = {
                'type': 'buy',
                'timestamp': data.index[strategy.current_index],
                'price': current_price,
                'shares': shares,
                'cost': cost,
                'portfolio_value': cash + position_size * current_price
            }
            trades.append(trade)
            
            return trade
        return None
    
    def sell_func(size=None, price=None, limit_price=None, stop_price=None, percent=None):
        nonlocal cash, position, position_size
        
        # Determine price
        current_price = price if price is not None else data['close'].iloc[strategy.current_index]
        
        # Determine size
        if size is not None:
            shares = min(size, position_size)
        elif percent is not None:
            shares = position_size * (percent / 100)
        else:
            shares = position_size
        
        # Execute trade
        if shares > 0:
            proceeds = shares * current_price
            cash += proceeds
            position_size -= shares
            if position_size <= 0:
                position = 0
                position_size = 0
            
            trade = {
                'type': 'sell',
                'timestamp': data.index[strategy.current_index],
                'price': current_price,
                'shares': shares,
                'proceeds': proceeds,
                'portfolio_value': cash + position_size * current_price
            }
            trades.append(trade)
            
            return trade
        return None
    
    # Set the buy and sell functions
    strategy.set_backtester_functions(buy_func, sell_func)
    
    # Initialize the strategy
    strategy.init()
    strategy.on_start()
    
    # Run the backtest
    for i in range(len(data)):
        # Update the current index
        strategy.current_index = i
        
        # Update portfolio value
        current_price = data['close'].iloc[i]
        portfolio_value = cash + position_size * current_price
        portfolio_values.append(portfolio_value)
        
        # Update strategy's portfolio info
        strategy.update_portfolio(position, position_size, cash, portfolio_value)
        
        # Call next() to generate signals for this bar
        strategy.next()
    
    # Call the on_finish hook
    strategy.on_finish()
    
    # Return results
    return strategy.signals, portfolio_values, trades


def plot_results(data, portfolio_values, trades, strategy_name):
    """
    Plot strategy results.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data.
    portfolio_values : list
        Portfolio values over time.
    trades : list
        List of trades.
    strategy_name : str
        Name of the strategy.
    """
    # Create a directory for the charts
    os.makedirs('strategy_charts', exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price and trades on top subplot
    ax1.plot(data.index, data['close'], label='Close Price')
    
    # Plot buy and sell points
    for trade in trades:
        if trade['type'] == 'buy':
            ax1.scatter(trade['timestamp'], trade['price'], marker='^', color='g', s=100)
        else:
            ax1.scatter(trade['timestamp'], trade['price'], marker='v', color='r', s=100)
    
    ax1.set_title(f'{strategy_name} - Price and Trades')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend()
    
    # Plot portfolio value on bottom subplot
    ax2.plot(data.index, portfolio_values, label='Portfolio Value', color='blue')
    ax2.set_title('Portfolio Value')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'strategy_charts/{strategy_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    print(f"Chart saved to strategy_charts/{strategy_name.replace(' ', '_').lower()}.png")


def run_strategy_example():
    """Run examples of each strategy type."""
    # Generate or load data
    try:
        # Try to fetch real data
        data_handler = BinanceDataHandler(use_cache=True, cache_dir='example_data')
        symbol = 'BTCUSDT'
        timeframe = '1d'
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365)  # Last year
        
        print(f"Fetching {symbol} {timeframe} data from {start_time.date()} to {end_time.date()}...")
        
        data = data_handler.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        data_source = f"{symbol} {timeframe}"
        
    except Exception as e:
        print(f"Error fetching real data: {e}")
        print("Using mock data instead.")
        data = create_mock_data(periods=365)
        data_source = "Mock Data"
    
    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Price range: ${data['low'].min():.2f} to ${data['high'].max():.2f}")
    
    # 1. Moving Average Crossover Strategy
    print("\n=== Testing Moving Average Crossover Strategy ===")
    ma_strategy = MovingAverageCrossStrategy(short_window=20, long_window=50)
    ma_signals, ma_portfolio, ma_trades = simulate_strategy(ma_strategy, data)
    
    print(f"Starting value: $10,000.00")
    print(f"Final value: ${ma_portfolio[-1]:.2f}")
    print(f"Return: {(ma_portfolio[-1] / 10000 - 1) * 100:.2f}%")
    print(f"Number of trades: {len(ma_trades)}")
    
    plot_results(data, ma_portfolio, ma_trades, f"MA Crossover Strategy - {data_source}")
    
    # 2. RSI Strategy
    print("\n=== Testing RSI Strategy ===")
    rsi_strategy = RSIStrategy(rsi_window=14, oversold=30, overbought=70)
    rsi_signals, rsi_portfolio, rsi_trades = simulate_strategy(rsi_strategy, data)
    
    print(f"Starting value: $10,000.00")
    print(f"Final value: ${rsi_portfolio[-1]:.2f}")
    print(f"Return: {(rsi_portfolio[-1] / 10000 - 1) * 100:.2f}%")
    print(f"Number of trades: {len(rsi_trades)}")
    
    plot_results(data, rsi_portfolio, rsi_trades, f"RSI Strategy - {data_source}")
    
    # 3. ML Strategy
    print("\n=== Testing ML Mean Reversion Strategy ===")
    ml_strategy = MLMeanReversionStrategy(train_size=0.7, prediction_horizon=5)
    ml_signals, ml_portfolio, ml_trades = simulate_strategy(ml_strategy, data)
    
    print(f"Starting value: $10,000.00")
    print(f"Final value: ${ml_portfolio[-1]:.2f}")
    print(f"Return: {(ml_portfolio[-1] / 10000 - 1) * 100:.2f}%")
    print(f"Number of trades: {len(ml_trades)}")
    
    plot_results(data, ml_portfolio, ml_trades, f"ML Strategy - {data_source}")
    
    # Compare the strategies
    print("\n=== Strategy Comparison ===")
    print(f"{'Strategy':<20} {'Final Value':<15} {'Return %':<10} {'# Trades':<10}")
    print("-" * 55)
    print(f"{'MA Crossover':<20} ${ma_portfolio[-1]:<14.2f} {(ma_portfolio[-1]/10000-1)*100:<9.2f}% {len(ma_trades):<10}")
    print(f"{'RSI':<20} ${rsi_portfolio[-1]:<14.2f} {(rsi_portfolio[-1]/10000-1)*100:<9.2f}% {len(rsi_trades):<10}")
    print(f"{'ML Mean Reversion':<20} ${ml_portfolio[-1]:<14.2f} {(ml_portfolio[-1]/10000-1)*100:<9.2f}% {len(ml_trades):<10}")


if __name__ == "__main__":
    run_strategy_example() 