#!/usr/bin/env python
# ema_cross_example.py - Complete EMA Cross Strategy Backtesting Example

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from pathlib import Path
import logging

# Import backtesting modules
from backtesting.data.ccxt_data import CCXTDataHandler
from backtesting.strategy.base import Strategy
from backtesting.portfolio.portfolio import Portfolio
from backtesting.execution.simulated_execution import SimulatedExecutionHandler
from backtesting.engine.backtester import Backtester
from backtesting.engine.event import SignalEvent
from backtesting.metrics.performance import calculate_performance_metrics, calculate_trade_metrics

# Create output directory for results
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)


class EMAcrossStrategy(Strategy):
    """
    EMA Cross Strategy - buys when fast EMA crosses above slow EMA
    and sells when fast EMA crosses below slow EMA.
    """
    
    def __init__(self):
        """Initialize strategy"""
        super().__init__()
        self.fast_window = 20  # Fast EMA window
        self.slow_window = 50  # Slow EMA window
        
    def init(self):
        """
        Initialize strategy with indicators.
        This is required by the Strategy base class.
        """
        self.initialized = True
        self.current_position = {}
        
        for symbol in self.symbols:
            self.current_position[symbol] = 0
        
        print(f"Initialized EMA Cross Strategy with fast={self.fast_window}, slow={self.slow_window}")
        
    def next(self):
        """
        Process the next bar.
        This is required by the Strategy base class.
        """
        # This will be called for each bar by the backtester
        # We'll use the on_data method from our implementation
        pass
        
    def initialize(self, data_handler, portfolio=None, event_loop=None):
        """
        Initialize the strategy with data handler, portfolio, and event loop.
        
        Parameters:
        -----------
        data_handler : DataHandler
            The data handler instance
        portfolio : Portfolio, optional
            The portfolio instance
        event_loop : EventLoop, optional
            The event loop for event dispatching
        """
        self.data_handler = data_handler
        self.portfolio = portfolio
        self.symbols = self.data_handler.symbols
        
        # Store the event loop if provided
        self.events_queue = event_loop
        
        # Call the required init method
        self.init()
        
    def calculate_ema_signals(self, symbol):
        """
        Calculate EMA signals for a symbol.
        
        Parameters:
        -----------
        symbol : str
            The symbol to calculate signals for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with price and EMA data
        """
        # Get the latest data
        data = self.data_handler.data[symbol].copy()
        
        # Calculate EMAs
        data['ema_fast'] = data['close'].ewm(span=self.fast_window, adjust=False).mean()
        data['ema_slow'] = data['close'].ewm(span=self.slow_window, adjust=False).mean()
        
        # Calculate crossover signals
        data['signal'] = 0.0
        data['position'] = 0.0
        
        # Crossover conditions
        data.loc[data['ema_fast'] > data['ema_slow'], 'signal'] = 1.0  # Buy signal
        data.loc[data['ema_fast'] < data['ema_slow'], 'signal'] = -1.0  # Sell signal
        
        # Generate position - this prevents multiple signals in the same direction
        data['position'] = data['signal'].diff()
        
        # Count how many crossover points exist
        buy_signals = (data['position'] > 0).sum()
        sell_signals = (data['position'] < 0).sum()
        
        # Debug print only once per symbol
        if not hasattr(self, 'debug_printed'):
            self.debug_printed = set()
        
        if symbol not in self.debug_printed:
            print(f"\nEMA Signal Analysis for {symbol}:")
            print(f"- Data points: {len(data)}")
            print(f"- Potential buy signals: {buy_signals}")
            print(f"- Potential sell signals: {sell_signals}")
            print(f"- First date: {data.index[0]}")
            print(f"- Last date: {data.index[-1]}")
            self.debug_printed.add(symbol)
        
        return data
    
    def on_data(self, event):
        """
        Process new market data and generate signals.
        
        Parameters:
        -----------
        event : MarketEvent
            The market event with new data
        """
        symbol = event.symbol
        timestamp = event.timestamp
        
        if symbol not in self.symbols:
            return
        
        # Calculate signals based on full dataset
        signal_data = self.calculate_ema_signals(symbol)
        
        # Only act on the current bar
        current_index = self.data_handler.current_index - 1
        
        # Debug - show current index periodically
        if current_index % 10 == 0 or current_index < 10:
            print(f"Processing bar at index {current_index}, timestamp: {timestamp}")
            if current_index < 10:
                # Print detailed debug for the first few bars
                print(f"Current bar data: {event.data}")
                if current_index > self.slow_window:
                    fast_ema = signal_data['ema_fast'].iloc[current_index]
                    slow_ema = signal_data['ema_slow'].iloc[current_index]
                    signal = signal_data['signal'].iloc[current_index]
                    position = signal_data['position'].iloc[current_index]
                    print(f"EMAs: fast={fast_ema:.2f}, slow={slow_ema:.2f}")
                    print(f"Signal: {signal}, Position Change: {position}")
        
        if current_index >= len(signal_data) or current_index < 0:
            if current_index % 10 == 0:
                print(f"⚠️ Index out of range: {current_index} (max: {len(signal_data)-1})")
            return
        
        # Only generate signals after we have enough data for EMA calculations
        if current_index <= self.slow_window:
            return
        
        current_position_signal = signal_data['position'].iloc[current_index]
        current_price = signal_data['close'].iloc[current_index]
        
        # Generate buy and sell signals
        if current_position_signal > 0 and self.current_position[symbol] <= 0:
            # Generate a buy signal
            self._generate_buy_signal(symbol, timestamp, current_price, 1.0)
            self.current_position[symbol] = 1
            
        elif current_position_signal < 0 and self.current_position[symbol] >= 0:
            # Generate a sell signal
            self._generate_sell_signal(symbol, timestamp, current_price, 1.0)
            self.current_position[symbol] = -1
    
    def _generate_buy_signal(self, symbol, timestamp, price, strength=1.0):
        """Generate buy signal"""
        print(f"🔵 BUY SIGNAL: {symbol} @ {price:.2f} [{timestamp}]")
        signal = SignalEvent(
            timestamp=timestamp,
            symbol=symbol,
            signal_type='BUY',
            strength=strength,
            metadata={'price': price}
        )
        self.events_queue.add_event(signal)
    
    def _generate_sell_signal(self, symbol, timestamp, price, strength=1.0):
        """Generate sell signal"""
        print(f"🔴 SELL SIGNAL: {symbol} @ {price:.2f} [{timestamp}]")
        signal = SignalEvent(
            timestamp=timestamp,
            symbol=symbol,
            signal_type='SELL',
            strength=strength,
            metadata={'price': price}
        )
        self.events_queue.add_event(signal)


def run_backtest(symbol, start_date, end_date, initial_capital=10000.0):
    """
    Run a backtest for the EMA cross strategy.
    
    Parameters:
    -----------
    symbol : str
        The symbol to backtest
    start_date : str
        The start date for the backtest (YYYY-MM-DD)
    end_date : str
        The end date for the backtest (YYYY-MM-DD)
    initial_capital : float, optional
        The initial capital for the backtest
        
    Returns:
    --------
    dict
        The backtest results
    """
    print(f"Starting backtest for {symbol} from {start_date} to {end_date}...")
    print(f"Initial capital: ${initial_capital:,.2f}")
    
    # Step 1: Initialize data handler
    print("\n1️⃣ Initializing data handler...")
    data_handler = CCXTDataHandler(exchange_id='binance')
    
    # Step 2: Fetch historical data
    print(f"\n2️⃣ Fetching historical data for {symbol}...")
    try:
        data = data_handler.get_historical_data(
            symbol=symbol,
            timeframe='5m',  # Daily timeframe
            start_time=start_date,
            end_time=end_date
        )
        
        # Set up data handler with the fetched data
        data_handler.data = {symbol: data}
        data_handler.symbols = [symbol]
        # Now call initialize to set up the other properties correctly
        data_handler.initialize(
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date)
        )
        
        print(f"Successfully loaded {len(data)} bars of data for {symbol}")
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Using sample data for demonstration...")
        
        # Generate sample data if fetching fails
        date_range = pd.date_range(start=start_date, end=end_date, freq='5m')
        data = pd.DataFrame({
            'open': np.random.normal(10000, 500, size=len(date_range)),
            'high': np.random.normal(10500, 500, size=len(date_range)),
            'low': np.random.normal(9500, 500, size=len(date_range)),
            'close': np.random.normal(10000, 500, size=len(date_range)),
            'volume': np.random.normal(1000, 100, size=len(date_range))
        }, index=date_range)
        
        # Ensure prices are realistic
        for i in range(1, len(data)):
            data.iloc[i, 0] = data.iloc[i-1, 3] * (1 + np.random.normal(0, 0.02))  # open based on prev close
            data.iloc[i, 3] = data.iloc[i, 0] * (1 + np.random.normal(0, 0.03))    # close based on open
            data.iloc[i, 1] = max(data.iloc[i, 0], data.iloc[i, 3]) * (1 + abs(np.random.normal(0, 0.01)))  # high
            data.iloc[i, 2] = min(data.iloc[i, 0], data.iloc[i, 3]) * (1 - abs(np.random.normal(0, 0.01)))  # low
        
        # Set up data handler with sample data
        data_handler.data = {symbol: data}
        data_handler.symbols = [symbol]
        # Now call initialize to set up the other properties correctly
        data_handler.initialize(
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date)
        )
    
    # Step 3: Initialize portfolio
    print("\n3️⃣ Initializing portfolio...")
    portfolio = Portfolio(initial_capital=initial_capital)
    portfolio.data_handler = data_handler
    
    # Step 4: Initialize strategy
    print("\n4️⃣ Initializing strategy...")
    strategy = EMAcrossStrategy()
    strategy.initialize(data_handler=data_handler, portfolio=portfolio)
    
    # Step 5: Initialize execution handler
    print("\n5️⃣ Initializing execution handler...")
    execution_handler = SimulatedExecutionHandler()
    execution_handler.initialize(
        data_handler=data_handler,
        commission=0.001,  # 0.1% commission
        slippage=0.0005    # 0.05% slippage
    )
    
    # Step 6: Initialize backtester
    print("\n6️⃣ Initializing backtester...")
    
    backtester = Backtester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        commission=0.001,  # 0.1% commission
        slippage=0.0005    # 0.05% slippage
    )
    
    # Step 7: Run the backtest
    print("\n7️⃣ Running backtest...")
    results = backtester.run()
    
    # Add debugging information
    print(f"\nDebug info:")
    print(f"- Equity curve entries: {len(results['equity_curve']) if 'equity_curve' in results and results['equity_curve'] else 0}")
    print(f"- Trades count: {len(results['trades']) if 'trades' in results and results['trades'] else 0}")
    
    # Step 8: Analyze results
    print("\n8️⃣ Analyzing results...")
    
    # Extract equity curve data
    if results['equity_curve']:
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        equity_df['returns'] = equity_df['portfolio_value'].pct_change().fillna(0)
        
        # Calculate performance metrics
        if 'trades' in results and results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            perf_metrics = calculate_performance_metrics(
                equity_curve=equity_df['portfolio_value'],
                trades=trades_df,
                risk_free_rate=0.0,
                periods_per_year=252
            )
            
            trade_metrics = calculate_trade_metrics(trades_df)
            
            # Print performance metrics
            print("\n=== PERFORMANCE METRICS ===")
            print(f"Total Return: {perf_metrics['total_return']*100:.2f}%")
            print(f"Annualized Return: {perf_metrics['annualized_return']*100:.2f}%")
            print(f"Sharpe Ratio: {perf_metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {perf_metrics['max_drawdown']*100:.2f}%")
            print(f"Win Rate: {trade_metrics['win_rate']*100:.2f}%")
            print(f"Profit Factor: {trade_metrics['profit_factor']:.2f}")
            print(f"Total Trades: {trade_metrics['total_trades']}")
            
            # Step 9: Visualize results
            print("\n9️⃣ Visualizing results...")
            plt.figure(figsize=(12, 8))
            
            # Plot equity curve
            plt.subplot(2, 1, 1)
            plt.plot(equity_df.index, equity_df['portfolio_value'], 'b-', label='Portfolio Value')
            plt.title(f'EMA Cross Strategy - {symbol} ({start_date} to {end_date})')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.legend()
            
            # Plot buy/sell signals
            plt.subplot(2, 1, 2)
            
            # Get the price and EMA data
            signal_data = strategy.calculate_ema_signals(symbol)
            
            # Plot price and EMAs
            plt.plot(signal_data.index, signal_data['close'], 'k-', label='Price')
            plt.plot(signal_data.index, signal_data['ema_fast'], 'r-', label=f'EMA{strategy.fast_window}')
            plt.plot(signal_data.index, signal_data['ema_slow'], 'g-', label=f'EMA{strategy.slow_window}')
            
            # Add buy/sell markers from trades
            for idx, trade in trades_df.iterrows():
                timestamp = pd.to_datetime(trade['timestamp'])
                price = trade['price']
                if trade['direction'] == 'BUY':
                    plt.plot(timestamp, price, 'go', markersize=8)
                else:  # SELL
                    plt.plot(timestamp, price, 'ro', markersize=8)
            
            plt.title(f'Price and EMAs with Trade Signals')
            plt.ylabel('Price ($)')
            plt.xlabel('Date')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            
            # Create a valid filename by replacing slashes in the symbol
            safe_symbol = symbol.replace('/', '_')
            plt.savefig(output_dir / f'ema_cross_{safe_symbol}_{start_date}_{end_date}.png')
            plt.show()
            
            # Save results to CSV
            equity_df.to_csv(output_dir / f'equity_curve_{safe_symbol}_{start_date}_{end_date}.csv')
            trades_df.to_csv(output_dir / f'trades_{safe_symbol}_{start_date}_{end_date}.csv')
            
            # Save performance metrics
            perf_df = pd.DataFrame([perf_metrics])
            perf_df.to_csv(output_dir / f'performance_{safe_symbol}_{start_date}_{end_date}.csv', index=False)
            
    return results


if __name__ == "__main__":
    # Run a backtest for Bitcoin from 2020-01-01 to 2025-03-19
    symbol = "BTC/USDT"
    start_date = "2025-01-01"
    end_date = "2025-03-19"  # Using a past date to ensure data is available
    initial_capital = 10000.0
    
    # Enable debug logging
    logging.basicConfig(level=logging.INFO)
    
    # Run backtest
    results = run_backtest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    # Additional debug info
    print("\nDetailed Results Summary:")
    for key in results:
        if isinstance(results[key], list):
            print(f"- {key}: {len(results[key])} items")
            if results[key] and len(results[key]) > 0:
                print(f"  First item: {results[key][0]}")
        else:
            print(f"- {key}: {results[key]}")
    
    print("\n✅ Backtest completed!")
