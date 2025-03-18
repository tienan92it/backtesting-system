"""
Example demonstrating portfolio tracking and simulated order execution.

This example shows how to use the Portfolio, SimulatedBroker, and 
SimulatedExecutionHandler classes for backtesting a simple trading strategy.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict

from backtesting.data.data_handler import DataHandler
from backtesting.strategy.examples.moving_average_cross import MovingAverageCrossStrategy
from backtesting.portfolio.portfolio import Portfolio
from backtesting.broker.simulated_broker import SimulatedBroker
from backtesting.execution.simulated_execution import SimulatedExecutionHandler
from backtesting.engine.backtester import Backtester
from backtesting.engine.event import EventType, Event, MarketEvent, SignalEvent, OrderEvent, FillEvent, PortfolioEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Add initialize method to Strategy class
class BacktestStrategy(MovingAverageCrossStrategy):
    """
    Extended MovingAverageCrossStrategy with initialize method for backtesting.
    """
    
    def initialize(self, data_handler=None, portfolio=None, event_loop=None):
        """
        Initialize the strategy for backtesting.
        
        Parameters:
        -----------
        data_handler : DataHandler, optional
            The data handler to use
        portfolio : Portfolio, optional
            The portfolio to use
        event_loop : EventLoop, optional
            The event loop to use
        """
        self.data_handler = data_handler
        self.portfolio = portfolio
        self.event_loop = event_loop
        
        # Set data from data handler
        if data_handler and data_handler.symbols:
            symbol = data_handler.symbols[0]  # Use first symbol
            data = data_handler.get_historical_data(symbol, '1d', None, None)
            self.set_data(data)
        
        # Initialize indicators
        self.init()
        
        # Connect buy and sell methods to signal generation
        self._buy_func = self._generate_buy_signal
        self._sell_func = self._generate_sell_signal
    
    def on_data(self, event):
        """
        Handle a market data event.
        
        Parameters:
        -----------
        event : MarketEvent
            The market event to handle
        """
        # Update current index
        self.current_index = len(self.data) - 1
        
        # Call next to generate signals
        self.next()
    
    def _generate_buy_signal(self, size=None, price=None, limit_price=None, stop_price=None, percent=None):
        """
        Generate a buy signal event.
        
        Parameters:
        -----------
        size : float, optional
            Size of the position to buy
        price : float, optional
            Price to buy at
        limit_price : float, optional
            Limit price for limit order
        stop_price : float, optional
            Stop price for stop order
        percent : float, optional
            Percentage of available cash to use
            
        Returns:
        --------
        None
        """
        if self.event_loop is None or self.data_handler is None:
            return
        
        # Get current timestamp and symbol
        timestamp = self.data.index[self.current_index]
        symbol = self.data_handler.symbols[0]  # Use first symbol
        
        # Create signal event
        signal_event = SignalEvent(
            timestamp=timestamp,
            symbol=symbol,
            signal_type='BUY',
            strength=1.0,
            metadata={
                'size': size,
                'price': price,
                'limit_price': limit_price,
                'stop_price': stop_price,
                'percent': percent
            }
        )
        
        # Add signal event to event loop
        self.event_loop.add_event(signal_event)
    
    def _generate_sell_signal(self, size=None, price=None, limit_price=None, stop_price=None, percent=None):
        """
        Generate a sell signal event.
        
        Parameters:
        -----------
        size : float, optional
            Size of the position to sell
        price : float, optional
            Price to sell at
        limit_price : float, optional
            Limit price for limit order
        stop_price : float, optional
            Stop price for stop order
        percent : float, optional
            Percentage of position to sell
            
        Returns:
        --------
        None
        """
        if self.event_loop is None or self.data_handler is None:
            return
        
        # Get current timestamp and symbol
        timestamp = self.data.index[self.current_index]
        symbol = self.data_handler.symbols[0]  # Use first symbol
        
        # Create signal event
        signal_event = SignalEvent(
            timestamp=timestamp,
            symbol=symbol,
            signal_type='SELL',
            strength=1.0,
            metadata={
                'size': size,
                'price': price,
                'limit_price': limit_price,
                'stop_price': stop_price,
                'percent': percent
            }
        )
        
        # Add signal event to event loop
        self.event_loop.add_event(signal_event)


# Add initialize method to Portfolio class
class BacktestPortfolio(Portfolio):
    """
    Extended Portfolio class with initialize method for backtesting.
    """
    
    def initialize(self, initial_capital=None, data_handler=None):
        """
        Initialize the portfolio for backtesting.
        
        Parameters:
        -----------
        initial_capital : float, optional
            The initial capital to use
        data_handler : DataHandler, optional
            The data handler to use
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital
            self.current_capital = initial_capital
        
        self.data_handler = data_handler
    
    def on_fill(self, fill_event):
        """
        Handle a fill event.
        
        Parameters:
        -----------
        fill_event : FillEvent
            The fill event to handle
            
        Returns:
        --------
        PortfolioEvent or None
            A portfolio event if the portfolio was updated
        """
        # Update the portfolio based on the fill
        self.update_fill(fill_event)
        
        # Create a portfolio event
        return PortfolioEvent(
            timestamp=fill_event.timestamp,
            cash=self.current_capital,
            positions=self.get_current_positions(),
            portfolio_value=self.get_portfolio_value()
        )
    
    def on_signal(self, signal_event):
        """
        Handle a signal event.
        
        Parameters:
        -----------
        signal_event : SignalEvent
            The signal event to handle
            
        Returns:
        --------
        OrderEvent or None
            An order event if an order should be placed
        """
        # Create an order based on the signal
        symbol = signal_event.symbol
        signal_type = signal_event.signal_type
        
        # Determine order direction
        if signal_type == 'BUY':
            direction = 'BUY'
        elif signal_type == 'SELL':
            direction = 'SELL'
        else:
            return None
        
        # Create order event
        order = OrderEvent(
            timestamp=signal_event.timestamp,
            symbol=symbol,
            order_type='MARKET',
            quantity=1.0,  # Fixed quantity for simplicity
            direction=direction,
            price=None  # Will be determined by the broker
        )
        
        return order
    
    def get_portfolio_value(self):
        """
        Get the current portfolio value.
        
        Returns:
        --------
        float
            The current portfolio value
        """
        # Calculate position values
        position_value = 0.0
        for symbol, quantity in self.positions.items():
            if quantity != 0 and self.data_handler and symbol in self.data_handler.current_bar:
                price = self.data_handler.current_bar[symbol]['close']
                position_value += price * quantity
        
        # Return total portfolio value
        return self.current_capital + position_value


# Create a simple implementation of DataHandler for our example
class SimpleDataHandler(DataHandler):
    """
    A simple implementation of DataHandler for backtesting with sample data.
    """
    
    def __init__(self):
        """Initialize the data handler."""
        self.data = {}
        self.current_bar = {}
        self.current_datetime = None
        self.symbols = []
        self.current_index = 0
        
    def add_data(self, symbol, data):
        """
        Add data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            The symbol to add data for
        data : pd.DataFrame
            The data to add
        """
        self.data[symbol] = data
        if symbol not in self.symbols:
            self.symbols.append(symbol)
    
    def get_historical_data(self, symbol, timeframe, start_time, end_time, limit=None):
        """
        Get historical data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            The symbol to get data for
        timeframe : str
            The timeframe to get data for
        start_time : datetime
            The start time
        end_time : datetime
            The end time
        limit : int, optional
            The maximum number of data points to return
            
        Returns:
        --------
        pd.DataFrame
            The historical data
        """
        if symbol not in self.data:
            raise ValueError(f"No data for symbol {symbol}")
        
        data = self.data[symbol]
        
        # Filter by date range
        if start_time is not None and end_time is not None:
            data = data.loc[start_time:end_time]
        
        # Limit the number of data points
        if limit is not None:
            data = data.iloc[-limit:]
        
        return data
    
    def get_symbols(self):
        """
        Get the list of available symbols.
        
        Returns:
        --------
        list
            The list of symbols
        """
        return self.symbols
    
    def get_timeframes(self):
        """
        Get the list of available timeframes.
        
        Returns:
        --------
        list
            The list of timeframes
        """
        return ['1d']  # We only support daily data in this example
    
    def initialize(self, start_date=None, end_date=None):
        """
        Initialize the data handler.
        
        Parameters:
        -----------
        start_date : datetime, optional
            The start date for the backtest
        end_date : datetime, optional
            The end date for the backtest
        """
        # Reset current index
        self.current_index = 0
        
        # Filter data by date range if provided
        if start_date is not None or end_date is not None:
            for symbol in self.symbols:
                self.data[symbol] = self.get_historical_data(symbol, '1d', start_date, end_date)
    
    def update_bars(self):
        """
        Update the current bar data.
        This would be called by the backtester to get the next bar.
        """
        # Check if we have data
        if not self.symbols or not self.data:
            return False
        
        # Check if we've reached the end of the data
        symbol = self.symbols[0]
        if self.current_index >= len(self.data[symbol]):
            return False
        
        # Update current bar for each symbol
        for symbol in self.symbols:
            if self.current_index < len(self.data[symbol]):
                self.current_bar[symbol] = self.data[symbol].iloc[self.current_index].to_dict()
                # Make sure all values are float for consistency
                for key, value in self.current_bar[symbol].items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        self.current_bar[symbol][key] = float(value)
        
        # Update current datetime
        self.current_datetime = self.data[symbol].index[self.current_index]
        
        # Increment index
        self.current_index += 1
        
        return True
    
    def get_latest_data(self, symbol, N=1):
        """
        Get the latest N bars of data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            The symbol to get data for
        N : int, optional
            The number of bars to get
            
        Returns:
        --------
        pd.DataFrame
            The latest N bars of data
        """
        if symbol not in self.data:
            raise ValueError(f"No data for symbol {symbol}")
        
        # Get the latest N bars
        end_idx = self.current_index
        start_idx = max(0, end_idx - N)
        
        return self.data[symbol].iloc[start_idx:end_idx]
    
    def get_all_bars(self):
        """
        Get all bars for all symbols, organized by timestamp.
        
        Returns:
        --------
        dict
            Dictionary of {timestamp: {symbol: bar_data}}
        """
        all_bars = defaultdict(dict)
        
        # For each symbol
        for symbol in self.symbols:
            # For each bar in the data
            for i, (timestamp, row) in enumerate(self.data[symbol].iterrows()):
                # Add the bar data to the dictionary
                all_bars[timestamp][symbol] = row.to_dict()
        
        return all_bars


def create_sample_data(periods=500):
    """
    Create sample OHLCV data for backtesting.
    
    Parameters:
    -----------
    periods : int
        Number of periods to generate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with sample OHLCV data
    """
    # Create date range
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=periods)
    dates = pd.date_range(start=start_date, end=end_date, periods=periods)
    
    # Generate random price data with trend
    np.random.seed(42)  # For reproducibility
    base_price = 100
    
    # Generate price with trend and noise
    trend = np.linspace(0, 30, periods)
    noise = np.random.normal(0, 1, periods)
    price = base_price + trend + noise * 3
    
    # Generate OHLCV data
    open_prices = price + np.random.normal(0, 0.5, periods)
    high_prices = price + np.random.normal(2, 0.5, periods)
    low_prices = price - np.random.normal(2, 0.5, periods)
    close_prices = price + np.random.normal(0, 0.5, periods)
    volume = np.random.normal(1000000, 200000, periods)
    
    # Ensure high is the highest and low is the lowest
    for i in range(periods):
        values = [open_prices[i], close_prices[i]]
        high_prices[i] = max(high_prices[i], max(values))
        low_prices[i] = min(low_prices[i], min(values))
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return data


def run_backtest_with_portfolio_broker():
    """
    Run a backtest with portfolio tracking and simulated order execution.
    """
    # Create sample data
    data = create_sample_data(periods=250)
    symbol = 'BTCUSDT'  # Changed from 'BTC/USDT' to match the strategy expectations
    
    # Create data handler with sample data
    data_handler = SimpleDataHandler()
    data_handler.add_data(symbol, data)
    
    # Create strategy (Moving Average Crossover)
    strategy = BacktestStrategy(
        short_window=5,
        long_window=15
    )
    
    # Create portfolio
    portfolio = BacktestPortfolio(
        initial_capital=100000.0,
        symbols=[symbol]
    )
    
    # Create simulated broker
    broker = SimulatedBroker(
        data_handler=data_handler,
        commission_model='percentage',
        commission_rate=0.001,  # 0.1% commission
        slippage_model='percentage',
        slippage_amount=0.001   # 0.1% slippage
    )
    broker.set_initial_balance(100000.0)
    
    # Create execution handler
    execution_handler = SimulatedExecutionHandler()
    execution_handler.initialize(
        data_handler=data_handler,
        commission=0.001,
        slippage=0.001
    )
    
    # Initialize components
    data_handler.initialize()
    portfolio.initialize(initial_capital=100000.0, data_handler=data_handler)
    
    # Create backtester
    backtester = Backtester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.001
    )
    
    # Initialize strategy with event loop from backtester
    strategy.initialize(data_handler=data_handler, portfolio=portfolio, event_loop=backtester.event_loop)
    
    # Run backtest
    logger.info("Starting backtest...")
    backtester.run()
    
    # Get results
    results = backtester.results
    
    # Print metrics
    logger.info("Backtest completed.")
    logger.info(f"Performance Metrics:")
    if 'performance_metrics' in results:
        for key, value in results['performance_metrics'].items():
            logger.info(f"  {key}: {value}")
    
    logger.info(f"Trade Metrics:")
    if 'trade_metrics' in results:
        for key, value in results['trade_metrics'].items():
            logger.info(f"  {key}: {value}")
    
    # Plot results
    plot_backtest_results(results, symbol)
    
    return results


def plot_backtest_results(results, symbol):
    """
    Plot backtest results.
    
    Parameters:
    -----------
    results : dict
        Backtest results
    symbol : str
        Trading symbol
    """
    if not results['equity_curve']:
        logger.warning("No equity curve data to plot")
        return
    
    # Convert results to DataFrame
    equity_df = pd.DataFrame(results['equity_curve'])
    equity_df.set_index('timestamp', inplace=True)
    equity_df.sort_index(inplace=True)
    
    # Get trades
    trades_df = pd.DataFrame(results['trades'])
    if not trades_df.empty:
        trades_df.set_index('timestamp', inplace=True)
        trades_df.sort_index(inplace=True)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(equity_df.index, equity_df['portfolio_value'], label='Portfolio Value')
    
    # Plot buy/sell signals
    if not trades_df.empty:
        # Buy signals
        buy_trades = trades_df[trades_df['direction'] == 'BUY']
        if not buy_trades.empty:
            ax1.scatter(buy_trades.index, buy_trades['price'], marker='^', color='green', label='Buy', s=100)
        
        # Sell signals
        sell_trades = trades_df[trades_df['direction'] == 'SELL']
        if not sell_trades.empty:
            ax1.scatter(sell_trades.index, sell_trades['price'], marker='v', color='red', label='Sell', s=100)
    
    # Plot cash and positions
    ax2.plot(equity_df.index, equity_df['cash'], label='Cash', color='orange')
    
    # Customize plots
    ax1.set_title(f'Backtest Results for {symbol}')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cash ($)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_portfolio_results.png')
    logger.info("Backtest results saved as 'backtest_portfolio_results.png'")
    plt.close(fig)


if __name__ == "__main__":
    run_backtest_with_portfolio_broker() 