"""
Example demonstrating how to use the backtester with mock components.

This example creates simple mock implementations of the required components
(data handler, strategy, portfolio, execution handler) and runs a backtest.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting.engine import (
    EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent, PortfolioEvent,
    Backtester
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backtester_example')


class MockDataHandler:
    """
    Mock data handler that generates synthetic price data.
    """
    
    def __init__(self, symbols: List[str], start_date: datetime, end_date: datetime, interval: str = '1d'):
        """
        Initialize the mock data handler.
        
        Parameters:
        -----------
        symbols : list
            List of symbols to generate data for.
        start_date : datetime
            Start date for the data.
        end_date : datetime
            End date for the data.
        interval : str, optional
            Interval for the data (e.g., '1d', '1h').
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data = {}
        
        # Generate synthetic data
        self._generate_data()
    
    def initialize(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        """
        Initialize the data handler.
        
        Parameters:
        -----------
        start_date : datetime, optional
            Start date for the data.
        end_date : datetime, optional
            End date for the data.
        """
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date
        
        # Regenerate data if dates changed
        if start_date or end_date:
            self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic price data."""
        # Calculate number of days
        days = (self.end_date - self.start_date).days + 1
        
        # Generate dates
        dates = [self.start_date + timedelta(days=i) for i in range(days)]
        
        # Generate data for each symbol
        for symbol in self.symbols:
            # Start with a base price
            base_price = 100.0 if symbol == 'BTCUSDT' else 50.0
            
            # Generate random walk prices
            np.random.seed(42)  # For reproducibility
            returns = np.random.normal(0.0005, 0.02, days)  # Mean daily return and volatility
            prices = base_price * np.cumprod(1 + returns)
            
            # Create OHLCV data
            data = []
            for i, date in enumerate(dates):
                price = prices[i]
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': price * (1 - 0.005 * np.random.random()),
                    'high': price * (1 + 0.01 * np.random.random()),
                    'low': price * (1 - 0.01 * np.random.random()),
                    'close': price,
                    'volume': 1000 * (1 + np.random.random())
                })
            
            # Store data
            self.data[symbol] = pd.DataFrame(data)
            self.data[symbol].set_index('timestamp', inplace=True)
    
    def get_all_bars(self) -> Dict[datetime, Dict[str, pd.Series]]:
        """
        Get all bars for all symbols.
        
        Returns:
        --------
        dict
            Dictionary of {timestamp: {symbol: bar_data}} for all timestamps.
        """
        # Create a dictionary of {timestamp: {symbol: bar_data}}
        all_bars = {}
        
        # Get all unique timestamps across all symbols
        all_timestamps = set()
        for symbol in self.symbols:
            all_timestamps.update(self.data[symbol].index)
        
        # Sort timestamps
        all_timestamps = sorted(all_timestamps)
        
        # Create the nested dictionary
        for timestamp in all_timestamps:
            all_bars[timestamp] = {}
            for symbol in self.symbols:
                if timestamp in self.data[symbol].index:
                    all_bars[timestamp][symbol] = self.data[symbol].loc[timestamp]
        
        return all_bars
    
    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """
        Get the latest n bars for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get bars for.
        n : int, optional
            Number of bars to get.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the latest n bars.
        """
        return self.data[symbol].iloc[-n:]


class MovingAverageCrossStrategy:
    """
    Simple moving average crossover strategy.
    
    Buys when the short moving average crosses above the long moving average,
    and sells when the short moving average crosses below the long moving average.
    """
    
    def __init__(self, symbols: List[str], short_window: int = 10, long_window: int = 30):
        """
        Initialize the strategy.
        
        Parameters:
        -----------
        symbols : list
            List of symbols to trade.
        short_window : int, optional
            Window for the short moving average.
        long_window : int, optional
            Window for the long moving average.
        """
        self.symbols = symbols
        self.short_window = short_window
        self.long_window = long_window
        
        # Data storage
        self.data = {symbol: pd.DataFrame() for symbol in symbols}
        
        # References to other components
        self.data_handler = None
        self.portfolio = None
        self.event_loop = None
        
        # Set up logger
        self.logger = logging.getLogger('strategy')
    
    def initialize(self, data_handler, portfolio, event_loop=None):
        """
        Initialize the strategy with references to other components.
        
        Parameters:
        -----------
        data_handler : DataHandler
            Data handler component.
        portfolio : Portfolio
            Portfolio component.
        event_loop : EventLoop, optional
            Event loop component.
        """
        self.data_handler = data_handler
        self.portfolio = portfolio
        self.event_loop = event_loop
    
    def on_data(self, event: MarketEvent):
        """
        Process new market data.
        
        Parameters:
        -----------
        event : MarketEvent
            Market event containing new data.
        """
        symbol = event.symbol
        
        # Update data storage
        new_data = pd.Series(event.data, name=event.timestamp)
        self.data[symbol] = pd.concat([self.data[symbol], pd.DataFrame([new_data])])
        
        # Log data size periodically
        if len(self.data[symbol]) % 50 == 0:
            self.logger.info(f"Processed {len(self.data[symbol])} bars for {symbol}")
        
        # Calculate signals if we have enough data
        if len(self.data[symbol]) >= self.long_window:
            # Calculate moving averages
            self.data[symbol]['short_ma'] = self.data[symbol]['close'].rolling(window=self.short_window).mean()
            self.data[symbol]['long_ma'] = self.data[symbol]['close'].rolling(window=self.long_window).mean()
            
            # Log MA values periodically
            if len(self.data[symbol]) % 50 == 0:
                self.logger.info(f"Short MA: {self.data[symbol]['short_ma'].iloc[-1]:.2f}, Long MA: {self.data[symbol]['long_ma'].iloc[-1]:.2f}")
            
            # Check for crossovers
            if len(self.data[symbol]) >= self.long_window + 1:
                # Get current and previous values
                curr_short = self.data[symbol]['short_ma'].iloc[-1]
                prev_short = self.data[symbol]['short_ma'].iloc[-2]
                curr_long = self.data[symbol]['long_ma'].iloc[-1]
                prev_long = self.data[symbol]['long_ma'].iloc[-2]
                
                # Log crossover check periodically
                if len(self.data[symbol]) % 50 == 0:
                    self.logger.info(f"Checking crossover - Prev: {prev_short:.2f} vs {prev_long:.2f}, Curr: {curr_short:.2f} vs {curr_long:.2f}")
                
                # Check for crossovers
                if prev_short <= prev_long and curr_short > curr_long:
                    # Golden cross: short MA crosses above long MA
                    self.logger.info(f"GOLDEN CROSS at {event.timestamp}: Short MA ({curr_short:.2f}) crossed above Long MA ({curr_long:.2f})")
                    self._generate_signal(event.timestamp, symbol, 'BUY')
                elif prev_short >= prev_long and curr_short < curr_long:
                    # Death cross: short MA crosses below long MA
                    self.logger.info(f"DEATH CROSS at {event.timestamp}: Short MA ({curr_short:.2f}) crossed below Long MA ({curr_long:.2f})")
                    self._generate_signal(event.timestamp, symbol, 'SELL')
    
    def _generate_signal(self, timestamp: datetime, symbol: str, signal_type: str):
        """
        Generate a trading signal.
        
        Parameters:
        -----------
        timestamp : datetime
            Timestamp for the signal.
        symbol : str
            Symbol to trade.
        signal_type : str
            Type of signal ('BUY' or 'SELL').
        """
        # Create a signal event
        signal = SignalEvent(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            strength=1.0,
            metadata={'strategy': 'MovingAverageCross'}
        )
        
        # Add the signal to the event queue if we have an event loop
        if self.event_loop:
            self.logger.info(f"Adding {signal_type} signal for {symbol} to event loop")
            self.event_loop.add_event(signal)
        else:
            # Fallback to direct portfolio call if no event loop
            self.logger.warning(f"No event loop available, calling portfolio directly for {signal_type} signal")
            self.portfolio.on_signal(signal)


class MockPortfolio:
    """
    Mock portfolio that tracks positions, cash, and performance.
    """
    
    def __init__(self):
        """Initialize the portfolio."""
        self.initial_capital = 0.0
        self.cash = 0.0
        self.positions = {}
        self.portfolio_value = 0.0
        self.data_handler = None
        self.trades = []
    
    def initialize(self, initial_capital: float, data_handler):
        """
        Initialize the portfolio.
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital for the portfolio.
        data_handler : DataHandler
            Data handler component.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.data_handler = data_handler
    
    def on_signal(self, event: SignalEvent) -> Optional[OrderEvent]:
        """
        Process a signal event.
        
        Parameters:
        -----------
        event : SignalEvent
            Signal event to process.
        
        Returns:
        --------
        OrderEvent or None
            Order event if a trade should be executed, None otherwise.
        """
        symbol = event.symbol
        signal_type = event.signal_type
        
        # Get the latest price
        latest_bar = self.data_handler.get_latest_bars(symbol, 1)
        if latest_bar.empty:
            return None
        
        price = latest_bar['close'].iloc[0]
        
        # Determine the order size
        if signal_type == 'BUY':
            # Use 50% of available cash
            cash_to_use = self.cash * 0.5
            quantity = cash_to_use / price
            
            # Create an order
            order = OrderEvent(
                timestamp=event.timestamp,
                symbol=symbol,
                order_type='MARKET',
                quantity=quantity,
                direction='BUY',
                price=price
            )
            return order
        
        elif signal_type == 'SELL':
            # Sell all holdings of this symbol
            quantity = self.positions.get(symbol, 0.0)
            
            if quantity > 0:
                # Create an order
                order = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    order_type='MARKET',
                    quantity=quantity,
                    direction='SELL',
                    price=price
                )
                return order
        
        return None
    
    def on_fill(self, event: FillEvent) -> Optional[PortfolioEvent]:
        """
        Process a fill event.
        
        Parameters:
        -----------
        event : FillEvent
            Fill event to process.
        
        Returns:
        --------
        PortfolioEvent or None
            Portfolio event with updated portfolio state.
        """
        symbol = event.symbol
        direction = event.direction
        quantity = event.quantity
        fill_price = event.fill_price
        commission = event.commission
        
        # Update positions and cash
        if direction == 'BUY':
            # Increase position
            self.positions[symbol] = self.positions.get(symbol, 0.0) + quantity
            
            # Decrease cash
            self.cash -= (quantity * fill_price + commission)
        
        elif direction == 'SELL':
            # Decrease position
            self.positions[symbol] = self.positions.get(symbol, 0.0) - quantity
            
            # Remove position if zero
            if self.positions[symbol] <= 0:
                self.positions[symbol] = 0.0
            
            # Increase cash
            self.cash += (quantity * fill_price - commission)
        
        # Record the trade
        self.trades.append({
            'timestamp': event.timestamp,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': fill_price,
            'commission': commission
        })
        
        # Calculate portfolio value
        self._update_portfolio_value()
        
        # Create a portfolio event
        portfolio_event = PortfolioEvent(
            timestamp=event.timestamp,
            cash=self.cash,
            positions=self.positions.copy(),
            portfolio_value=self.portfolio_value,
            trades={'latest': self.trades[-1] if self.trades else None}
        )
        
        return portfolio_event
    
    def _update_portfolio_value(self):
        """Update the portfolio value based on current positions and prices."""
        # Start with cash
        self.portfolio_value = self.cash
        
        # Add value of positions
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                # Get the latest price
                latest_bar = self.data_handler.get_latest_bars(symbol, 1)
                if not latest_bar.empty:
                    price = latest_bar['close'].iloc[0]
                    self.portfolio_value += quantity * price


class MockExecutionHandler:
    """
    Mock execution handler that simulates order execution.
    """
    
    def __init__(self):
        """Initialize the execution handler."""
        self.data_handler = None
        self.commission = 0.0
        self.slippage = 0.0
    
    def initialize(self, data_handler, commission: float = 0.001, slippage: float = 0.0):
        """
        Initialize the execution handler.
        
        Parameters:
        -----------
        data_handler : DataHandler
            Data handler component.
        commission : float, optional
            Commission rate as a decimal.
        slippage : float, optional
            Slippage model as a decimal.
        """
        self.data_handler = data_handler
        self.commission = commission
        self.slippage = slippage
    
    def execute_order(self, event: OrderEvent) -> Optional[FillEvent]:
        """
        Execute an order.
        
        Parameters:
        -----------
        event : OrderEvent
            Order event to execute.
        
        Returns:
        --------
        FillEvent or None
            Fill event if the order was executed, None otherwise.
        """
        symbol = event.symbol
        order_type = event.order_type
        quantity = event.quantity
        direction = event.direction
        price = event.price
        
        # Apply slippage
        if direction == 'BUY':
            # Buy at a slightly higher price
            fill_price = price * (1 + self.slippage)
        else:
            # Sell at a slightly lower price
            fill_price = price * (1 - self.slippage)
        
        # Calculate commission
        commission = fill_price * quantity * self.commission
        
        # Create a fill event
        fill = FillEvent(
            timestamp=event.timestamp,
            symbol=symbol,
            quantity=quantity,
            direction=direction,
            fill_price=fill_price,
            commission=commission,
            order_id=str(id(event))
        )
        
        return fill


def plot_results(backtester):
    """
    Plot the results of a backtest.
    
    Parameters:
    -----------
    backtester : Backtester
        Backtester instance with results.
    """
    # Get results
    equity_curve = backtester.get_equity_curve()
    trades = backtester.get_trades()
    
    if equity_curve is None or equity_curve.empty:
        logger.warning("No equity curve data available to plot")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot equity curve
    plt.plot(equity_curve.index, equity_curve['portfolio_value'], label='Portfolio Value')
    
    # Plot buy and sell points
    if trades is not None and not trades.empty:
        # Buy trades
        buy_trades = trades[trades['direction'] == 'BUY']
        if not buy_trades.empty:
            plt.scatter(buy_trades.index, buy_trades['price'] * buy_trades['quantity'], 
                       marker='^', color='green', s=100, label='Buy')
        
        # Sell trades
        sell_trades = trades[trades['direction'] == 'SELL']
        if not sell_trades.empty:
            plt.scatter(sell_trades.index, sell_trades['price'] * sell_trades['quantity'], 
                        marker='v', color='red', s=100, label='Sell')
    
    # Format the plot
    plt.title('Backtest Results: Portfolio Value and Trades')
    plt.ylabel('Portfolio Value')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig('backtest_results.png')
    logger.info("Saved backtest results plot to 'backtest_results.png'")
    
    # Show the plot
    plt.show()


def run_backtest():
    """Run a backtest with the mock components."""
    # Set up the components
    symbols = ['BTCUSDT']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    
    # Create the components
    data_handler = MockDataHandler(symbols, start_date, end_date)
    strategy = MovingAverageCrossStrategy(symbols, short_window=5, long_window=15)  # Smaller windows
    portfolio = MockPortfolio()
    execution_handler = MockExecutionHandler()
    
    # Create the backtester
    backtester = Backtester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        initial_capital=100000.0,
        start_date=start_date,
        end_date=end_date,
        commission=0.001,
        slippage=0.001
    )
    
    # Run the backtest
    logger.info("Running backtest...")
    results = backtester.run()
    
    # Print results
    logger.info("Backtest completed")
    
    # Get performance metrics
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        logger.info("Performance Metrics:")
        logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    
    # Get trade metrics
    if 'trade_metrics' in results:
        metrics = results['trade_metrics']
        logger.info("Trade Metrics:")
        logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    
    # Return the results for further analysis
    return backtester


if __name__ == "__main__":
    backtester = run_backtest()
    
    # Plot the results
    plot_results(backtester) 