#!/usr/bin/env python
"""
Example script demonstrating how to run a backtest and analyze the results
using the performance metrics module.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting.data import DataHandler
from backtesting.strategy import Strategy
from backtesting.engine import Backtester
from backtesting.engine.event import (
    EventType, Event, MarketEvent, SignalEvent, 
    OrderEvent, FillEvent, PortfolioEvent
)
from backtesting.metrics import (
    calculate_performance_metrics,
    calculate_trade_metrics
)


class MovingAverageCrossStrategy(Strategy):
    """
    A simple moving average crossover strategy.
    
    This strategy generates a buy signal when the short moving average crosses above
    the long moving average, and a sell signal when the short moving average crosses
    below the long moving average.
    """
    
    def __init__(self, short_window=20, long_window=50):
        """
        Initialize the strategy with the specified parameters.
        
        Parameters:
        -----------
        short_window : int
            The window length for the short moving average.
        long_window : int
            The window length for the long moving average.
        """
        self.short_window = short_window
        self.long_window = long_window
        self.positions = {}
        self.bought = {}
        self.sold = {}
    
    def init(self, data_handler):
        """
        Initialize the strategy with the data handler.
        
        Parameters:
        -----------
        data_handler : DataHandler
            The data handler to use for data access.
        """
        self.data_handler = data_handler
        self.symbols = data_handler.symbols
        
        # Initialize positions for each symbol
        for symbol in self.symbols:
            self.positions[symbol] = 0
            self.bought[symbol] = False
            self.sold[symbol] = False
    
    def initialize(self, data_handler=None, portfolio=None, event_loop=None):
        """
        Initialize the strategy with the necessary components.
        
        Parameters:
        -----------
        data_handler : DataHandler, optional
            The data handler to use for data access.
        portfolio : Portfolio, optional
            The portfolio to use for position tracking.
        event_loop : EventLoop, optional
            The event loop to use for event handling.
        """
        if data_handler is not None:
            self.data_handler = data_handler
            self.symbols = data_handler.symbols
        
        self.portfolio = portfolio
        self.event_loop = event_loop
        
        # Initialize positions for each symbol
        for symbol in self.symbols:
            self.positions[symbol] = 0
            self.bought[symbol] = False
            self.sold[symbol] = False
    
    def on_data(self, event):
        """
        Process a market data event.
        
        Parameters:
        -----------
        event : MarketEvent
            The market event to process.
        """
        # Get the symbol and data from the event
        symbol = event.symbol
        data = event.data
        
        # Get the latest data for the symbol
        latest_data = self.data_handler.get_latest_data(symbol, N=self.long_window+1)
        
        if len(latest_data) < self.long_window:
            return
        
        # Calculate moving averages
        short_ma = latest_data['close'].rolling(window=self.short_window).mean()
        long_ma = latest_data['close'].rolling(window=self.long_window).mean()
        
        # Check for crossover
        if len(short_ma) >= 2 and len(long_ma) >= 2:
            if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
                # Buy signal
                if not self.bought[symbol]:
                    from backtesting.engine.event import SignalEvent
                    signal = SignalEvent(
                        timestamp=event.timestamp,
                        symbol=symbol,
                        signal_type='BUY',
                        strength=1.0
                    )
                    self.event_loop.add_event(signal)
                    self.bought[symbol] = True
                    self.sold[symbol] = False
            
            elif short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
                # Sell signal
                if not self.sold[symbol]:
                    from backtesting.engine.event import SignalEvent
                    signal = SignalEvent(
                        timestamp=event.timestamp,
                        symbol=symbol,
                        signal_type='SELL',
                        strength=1.0
                    )
                    self.event_loop.add_event(signal)
                    self.sold[symbol] = True
                    self.bought[symbol] = False
    
    def next(self, data):
        """
        Process the next data point and generate signals.
        
        Parameters:
        -----------
        data : dict
            Dictionary of data for each symbol.
        
        Returns:
        --------
        list
            List of Signal objects.
        """
        signals = []
        
        for symbol in self.symbols:
            # Get the latest data
            latest_data = self.data_handler.get_latest_data(symbol, N=self.long_window+1)
            
            if len(latest_data) < self.long_window:
                continue
            
            # Calculate moving averages
            short_ma = latest_data['close'].rolling(window=self.short_window).mean()
            long_ma = latest_data['close'].rolling(window=self.long_window).mean()
            
            # Check for crossover
            if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
                # Buy signal
                if not self.bought[symbol]:
                    from backtesting.engine.event import SignalEvent
                    signals.append(SignalEvent(
                        timestamp=latest_data.index[-1],
                        symbol=symbol,
                        signal_type='BUY',
                        strength=1.0
                    ))
                    self.bought[symbol] = True
                    self.sold[symbol] = False
            
            elif short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
                # Sell signal
                if not self.sold[symbol]:
                    from backtesting.engine.event import SignalEvent
                    signals.append(SignalEvent(
                        timestamp=latest_data.index[-1],
                        symbol=symbol,
                        signal_type='SELL',
                        strength=1.0
                    ))
                    self.sold[symbol] = True
                    self.bought[symbol] = False
        
        return signals
    
    def calculate_signals(self, data):
        """
        Calculate the trading signals for the given data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The OHLCV data to calculate signals for.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with signal columns added.
        """
        # Make a copy of the data
        signals = data.copy()
        
        # Calculate moving averages
        signals['short_ma'] = signals['close'].rolling(window=self.short_window).mean()
        signals['long_ma'] = signals['close'].rolling(window=self.long_window).mean()
        
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
        
        return signals


def create_mock_data(periods=500, freq='D'):
    """
    Create mock OHLCV data for demonstration purposes.
    
    Parameters:
    -----------
    periods : int
        Number of periods to generate.
    freq : str
        Frequency of the data (e.g., 'D' for daily, 'H' for hourly).
    
    Returns:
    --------
    pd.DataFrame
        Mock OHLCV data.
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


class MockDataHandler(DataHandler):
    """Mock data handler for backtesting."""
    
    def __init__(self, data):
        """
        Initialize with the provided data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data.
        """
        self.data = data
        self.latest_data = None
        self.current_idx = 0
        self.symbols = ['MOCK']
    
    def initialize(self, start_date=None, end_date=None):
        """
        Initialize the data handler with the specified date range.
        
        Parameters:
        -----------
        start_date : datetime, optional
            The start date for the data.
        end_date : datetime, optional
            The end date for the data.
        """
        # Filter data by date if provided
        if start_date is not None or end_date is not None:
            mask = pd.Series(True, index=self.data.index)
            if start_date is not None:
                mask = mask & (self.data.index >= start_date)
            if end_date is not None:
                mask = mask & (self.data.index <= end_date)
            self.data = self.data[mask]
        
        # Reset index
        self.current_idx = 0
        
        # Return the first market event
        if len(self.data) > 0:
            first_row = self.data.iloc[0]
            return MarketEvent(
                timestamp=self.data.index[0],
                symbol=self.symbols[0],
                data=first_row
            )
        else:
            return None
    
    def get_latest_data(self, symbol, N=1):
        """
        Get the latest N bars of data.
        
        Parameters:
        -----------
        symbol : str
            The symbol to get data for.
        N : int
            Number of bars to return.
        
        Returns:
        --------
        pd.DataFrame
            The latest N bars of data.
        """
        if self.current_idx < N:
            return self.data.iloc[:self.current_idx+1]
        else:
            return self.data.iloc[self.current_idx-N+1:self.current_idx+1]
    
    def update_bars(self):
        """
        Update the latest data.
        
        Returns:
        --------
        bool
            True if there is more data, False otherwise.
        """
        if self.current_idx >= len(self.data) - 1:
            return False
        
        self.current_idx += 1
        self.latest_data = self.data.iloc[self.current_idx]
        return True
    
    def get_historical_data(self, symbol, timeframe, start_date=None, end_date=None):
        """
        Get historical data for the specified symbol and timeframe.
        
        Parameters:
        -----------
        symbol : str
            The symbol to get data for.
        timeframe : str
            The timeframe to get data for.
        start_date : datetime, optional
            The start date for the data.
        end_date : datetime, optional
            The end date for the data.
        
        Returns:
        --------
        pd.DataFrame
            Historical OHLCV data.
        """
        # For mock data, we ignore the timeframe and just return the data
        if start_date is not None or end_date is not None:
            # Filter by date if provided
            mask = pd.Series(True, index=self.data.index)
            if start_date is not None:
                mask = mask & (self.data.index >= start_date)
            if end_date is not None:
                mask = mask & (self.data.index <= end_date)
            return self.data[mask]
        else:
            return self.data
    
    def get_symbols(self):
        """
        Get the list of available symbols.
        
        Returns:
        --------
        list
            List of available symbols.
        """
        return self.symbols
    
    def get_timeframes(self):
        """
        Get the list of available timeframes.
        
        Returns:
        --------
        list
            List of available timeframes.
        """
        return ['1d']  # Mock data is daily
    
    def get_all_bars(self):
        """
        Get all bars of data.
        
        Returns:
        --------
        dict
            Dictionary of DataFrames, keyed by symbol.
        """
        return {self.symbols[0]: self.data}


class MockPortfolio:
    """Mock portfolio for backtesting."""
    
    def __init__(self, data_handler, initial_capital=10000.0):
        """
        Initialize the portfolio.
        
        Parameters:
        -----------
        data_handler : DataHandler
            The data handler to use for data access.
        initial_capital : float
            The initial capital.
        """
        self.data_handler = data_handler
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.holdings = {}
        self.transactions = []
        
        # Initialize holdings for each symbol
        for symbol in data_handler.symbols:
            self.holdings[symbol] = 0
    
    def initialize(self, initial_capital=None, data_handler=None):
        """
        Initialize the portfolio.
        
        Parameters:
        -----------
        initial_capital : float, optional
            The initial capital. If None, use the value from __init__.
        data_handler : DataHandler, optional
            The data handler to use for data access. If None, use the one from __init__.
        
        Returns:
        --------
        PortfolioEvent
            The initial portfolio event.
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital
            self.current_capital = initial_capital
        
        if data_handler is not None:
            self.data_handler = data_handler
        
        # Reset holdings
        for symbol in self.data_handler.symbols:
            self.holdings[symbol] = 0
        
        # Reset transactions
        self.transactions = []
        
        # Create initial portfolio event
        return PortfolioEvent(
            timestamp=datetime.now(),
            cash=self.current_capital,
            positions=self.holdings.copy(),
            portfolio_value=self.current_capital
        )
    
    def update_signal(self, signal):
        """
        Update the portfolio based on a signal.
        
        Parameters:
        -----------
        signal : SignalEvent
            The signal event.
        
        Returns:
        --------
        OrderEvent
            The order event.
        """
        symbol = signal.symbol
        direction = signal.direction
        strength = signal.strength
        
        # Create an order
        order = OrderEvent(
            symbol=symbol,
            order_type='MARKET',
            quantity=1.0,  # Fixed quantity for simplicity
            direction=direction
        )
        
        return order
    
    def update_fill(self, fill):
        """
        Update the portfolio based on a fill.
        
        Parameters:
        -----------
        fill : FillEvent
            The fill event.
        """
        symbol = fill.symbol
        direction = fill.direction
        quantity = fill.quantity
        price = fill.price
        commission = fill.commission
        
        # Update holdings
        if direction == 'BUY':
            self.holdings[symbol] += quantity
            self.current_capital -= (price * quantity + commission)
        else:  # SELL
            self.holdings[symbol] -= quantity
            self.current_capital += (price * quantity - commission)
        
        # Record transaction
        self.transactions.append({
            'timestamp': fill.timestamp,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'commission': commission
        })
    
    def update_timeindex(self, market_event):
        """
        Update the portfolio based on a market event.
        
        Parameters:
        -----------
        market_event : MarketEvent
            The market event.
        """
        # Nothing to do for this simple portfolio
        pass


class MockExecutionHandler:
    """Mock execution handler for backtesting."""
    
    def __init__(self, data_handler, commission=0.001, slippage=0.0):
        """
        Initialize the execution handler.
        
        Parameters:
        -----------
        data_handler : DataHandler
            The data handler to use for data access.
        commission : float
            The commission rate.
        slippage : float
            The slippage rate.
        """
        self.data_handler = data_handler
        self.commission = commission
        self.slippage = slippage
    
    def initialize(self, data_handler=None, commission=None, slippage=None):
        """
        Initialize the execution handler.
        
        Parameters:
        -----------
        data_handler : DataHandler, optional
            The data handler to use for data access. If None, use the one from __init__.
        commission : float, optional
            The commission rate. If None, use the value from __init__.
        slippage : float, optional
            The slippage rate. If None, use the value from __init__.
        """
        if data_handler is not None:
            self.data_handler = data_handler
        
        if commission is not None:
            self.commission = commission
        
        if slippage is not None:
            self.slippage = slippage
    
    def execute_order(self, order):
        """
        Execute an order.
        
        Parameters:
        -----------
        order : OrderEvent
            The order event.
        
        Returns:
        --------
        FillEvent
            The fill event.
        """
        symbol = order.symbol
        direction = order.direction
        quantity = order.quantity
        
        # Get the latest price
        latest_data = self.data_handler.get_latest_data(symbol, N=1)
        price = latest_data['close'].iloc[-1]
        
        # Apply slippage
        if direction == 'BUY':
            price *= (1 + self.slippage)
        else:  # SELL
            price *= (1 - self.slippage)
        
        # Calculate commission
        commission = price * quantity * self.commission
        
        # Create a fill event
        fill = FillEvent(
            timestamp=latest_data.index[-1],
            symbol=symbol,
            exchange='MOCK',
            quantity=quantity,
            direction=direction,
            price=price,
            commission=commission
        )
        
        return fill


def plot_backtest_results(data, signals, portfolio_values, trades):
    """
    Plot the backtest results.
    
    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data.
    signals : pd.DataFrame
        Signal data.
    portfolio_values : pd.Series
        Portfolio values over time.
    trades : pd.DataFrame
        Trade data.
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price
    ax1.plot(data.index, data['close'], label='Price')
    
    # Calculate and plot moving averages if not in signals
    if 'short_ma' not in signals.columns or 'long_ma' not in signals.columns:
        # Calculate moving averages
        short_window = 20  # Default values
        long_window = 50
        short_ma = data['close'].rolling(window=short_window).mean()
        long_ma = data['close'].rolling(window=long_window).mean()
        
        ax1.plot(data.index, short_ma, label=f'Short MA ({short_window})')
        ax1.plot(data.index, long_ma, label=f'Long MA ({long_window})')
    else:
        # Use the moving averages from signals
        ax1.plot(signals.index, signals['short_ma'], label='Short MA')
        ax1.plot(signals.index, signals['long_ma'], label='Long MA')
    
    # Plot buy and sell signals if available
    if 'signal' in signals.columns:
        buy_signals = signals[signals['signal'] == 1.0]
        sell_signals = signals[signals['signal'] == -1.0]
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', s=100, label='Buy')
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', s=100, label='Sell')
    
    # Plot portfolio value
    if not portfolio_values.empty:
        ax2.plot(portfolio_values.index, portfolio_values, label='Portfolio Value')
    
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
    plt.savefig('backtest_results.png')
    plt.close()


def run_backtest_example():
    """Run a backtest example and analyze the results."""
    print("Creating mock data...")
    data = create_mock_data(periods=500)
    
    print("Initializing strategy and data handler...")
    strategy = MovingAverageCrossStrategy(short_window=20, long_window=50)
    data_handler = MockDataHandler(data)
    
    print("Initializing portfolio and execution handler...")
    portfolio = MockPortfolio(data_handler, initial_capital=10000.0)
    execution_handler = MockExecutionHandler(data_handler, commission=0.001)
    
    print("Running backtest...")
    backtester = Backtester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        initial_capital=10000.0
    )
    
    results = backtester.run()
    
    # Get results
    signals = results.get('signals', pd.DataFrame())
    portfolio_values = results.get('portfolio_values', pd.Series())
    trades = results.get('trades', pd.DataFrame())
    
    # If no signals were generated, calculate them directly
    if signals.empty:
        signals = strategy.calculate_signals(data)
    
    # Plot results
    print("Plotting backtest results...")
    plot_backtest_results(data, signals, portfolio_values, trades)
    
    # Calculate and print performance metrics
    print("Calculating performance metrics...")
    if len(portfolio_values) > 0:
        equity_curve = pd.Series(portfolio_values)
        
        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(equity_curve, trades)
        trade_metrics = calculate_trade_metrics(trades)
        
        # Print report
        print("\n" + "="*50)
        print(" "*15 + "BACKTEST RESULTS")
        print("="*50)
        
        print("\nSTRATEGY PARAMETERS:")
        print(f"Short Window: {strategy.short_window}")
        print(f"Long Window: {strategy.long_window}")
        
        print("\nPORTFOLIO METRICS:")
        print(f"Initial Capital: ${backtester.initial_capital:,.2f}")
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
        
        # Calculate and print signal metrics
        buy_signals = signals[signals['signal'] == 1.0]
        sell_signals = signals[signals['signal'] == -1.0]
        
        print("\nSIGNAL METRICS:")
        print(f"Total Buy Signals: {len(buy_signals)}")
        print(f"Total Sell Signals: {len(sell_signals)}")
    
    print(f"\nPlot saved to: {os.path.abspath('backtest_results.png')}")


if __name__ == "__main__":
    run_backtest_example() 