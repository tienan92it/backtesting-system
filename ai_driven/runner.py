import importlib
import inspect
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Type, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

from backtesting.data.binance_data import BinanceDataHandler
from backtesting.data.ccxt_data import CCXTDataHandler
from backtesting.engine.backtester import Backtester
from backtesting.strategy.base import Strategy
from backtesting.portfolio.portfolio import Portfolio
from backtesting.execution.simulated_execution import SimulatedExecutionHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Data class to hold the results of a backtest."""
    equity_curve: pd.Series
    returns: pd.Series  
    trades: list
    metrics: Dict[str, Any]
    strategy_code: str
    strategy_params: Dict[str, Any]
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the results to a dictionary."""
        return {
            'equity_curve': self.equity_curve.to_dict() if isinstance(self.equity_curve, pd.Series) else self.equity_curve,
            'returns': self.returns.to_dict() if isinstance(self.returns, pd.Series) else self.returns,
            'trades': self.trades,
            'metrics': self.metrics,
            'strategy_code': self.strategy_code,
            'strategy_params': self.strategy_params,
            'config': self.config
        }

def load_strategy_from_code(strategy_code: str) -> Type[Strategy]:
    """
    Dynamically load a strategy class from a string of Python code.
    
    Parameters:
    -----------
    strategy_code : str
        String containing Python code that defines a strategy class.
        
    Returns:
    --------
    Type[Strategy]
        The strategy class that was defined in the code.
    """
    # Create a unique module name
    module_name = f"dynamic_strategy_{hash(strategy_code) % 10000}"
    
    try:
        # Create a new module
        module = type(sys)(module_name)
        sys.modules[module_name] = module
        
        # Execute the code in the context of the new module
        exec(strategy_code, module.__dict__)
        
        # Find all Strategy subclasses defined in the module
        strategies = []
        for name, obj in module.__dict__.items():
            if inspect.isclass(obj) and issubclass(obj, Strategy) and obj != Strategy:
                strategies.append(obj)
        
        if not strategies:
            raise ValueError("No strategy class found in the provided code")
        
        # Return the first strategy class found
        return strategies[0]
    
    except Exception as e:
        logger.error(f"Error loading strategy: {e}")
        raise ValueError(f"Failed to load strategy: {e}") from e

def generate_sample_data(symbol: str, start_date: str, end_date: str, timeframe: str = '1d') -> pd.DataFrame:
    """
    Generate sample price data for testing when real data cannot be fetched.
    
    Parameters:
    -----------
    symbol : str
        The trading symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    timeframe : str
        Candle timeframe
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with OHLCV data
    """
    # Convert dates to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate date range based on timeframe
    if timeframe == '1d':
        dates = pd.date_range(start=start, end=end, freq='D')
    elif timeframe == '1h':
        dates = pd.date_range(start=start, end=end, freq='H')
    else:
        # Default to daily
        dates = pd.date_range(start=start, end=end, freq='D')
    
    # Generate random price data with a slight upward trend
    n = len(dates)
    np.random.seed(42)  # For reproducibility
    
    # Start price around 20000 for BTC-like scale
    base_price = 20000
    
    # Generate returns with slight positive drift
    returns = np.random.normal(0.0005, 0.02, n)  # Mean daily return ~0.05%
    
    # Convert returns to price series
    price_series = base_price * (1 + np.cumsum(returns))
    
    # Generate OHLC data
    volatility = 0.01  # Intraday volatility factor
    df = pd.DataFrame({
        'open': price_series,
        'high': price_series * (1 + np.random.uniform(0, volatility, n)),
        'low': price_series * (1 - np.random.uniform(0, volatility, n)),
        'close': price_series * (1 + np.random.normal(0, volatility/2, n)),
        'volume': np.random.lognormal(10, 1, n) * 100
    }, index=dates)
    
    # Ensure high is always highest and low is always lowest
    for i in range(len(df)):
        high = max(df.iloc[i]['open'], df.iloc[i]['close'], df.iloc[i]['high'])
        low = min(df.iloc[i]['open'], df.iloc[i]['close'], df.iloc[i]['low'])
        df.iloc[i, df.columns.get_loc('high')] = high
        df.iloc[i, df.columns.get_loc('low')] = low
    
    return df

def run_backtest(
    strategy_code: str,
    config: Dict[str, Any]
) -> BacktestResult:
    """
    Run a backtest using the provided strategy code and configuration.
    
    Parameters:
    -----------
    strategy_code : str
        String containing Python code that defines a strategy class.
    config : Dict[str, Any]
        Configuration parameters for the backtest, including:
        - symbol: The trading symbol (e.g., 'BTC/USDT')
        - start_date: Start date for the backtest (e.g., '2022-01-01')
        - end_date: End date for the backtest (e.g., '2022-12-31')
        - initial_capital: Initial capital for the portfolio (e.g., 10000.0)
        - timeframe: Candle timeframe (e.g., '1d', '4h', '1h')
        - commission: Commission rate (e.g., 0.001 for 0.1%)
        - exchange: Exchange to use ('binance' or 'ccxt')
        - strategy_params: Parameters to pass to the strategy (optional)
        - use_sample_data: Whether to use generated sample data instead of real data (optional)
        
    Returns:
    --------
    BacktestResult
        Results of the backtest, including equity curve, returns, trades, and metrics.
    """
    logger.info("Setting up backtest configuration")
    
    # Extract configuration parameters with defaults
    symbol = config.get('symbol', 'BTC/USDT')
    start_date = config.get('start_date', '2022-01-01')
    end_date = config.get('end_date', '2022-12-31')
    initial_capital = float(config.get('initial_capital', 10000.0))
    timeframe = config.get('timeframe', '1d')
    commission = float(config.get('commission', 0.001))
    exchange = config.get('exchange', 'binance')
    strategy_params = config.get('strategy_params', {})
    use_sample_data = config.get('use_sample_data', False)
    
    try:
        # Load the strategy class from the code
        logger.info("Loading strategy from code")
        StrategyClass = load_strategy_from_code(strategy_code)
        
        # Create strategy instance with provided parameters
        strategy = StrategyClass(**strategy_params)
        
        # Convert symbol format for Binance
        binance_symbol = symbol.replace("/", "") if exchange.lower() == 'binance' else symbol
        
        # Get market data
        if use_sample_data:
            logger.info("Using generated sample data")
            df = generate_sample_data(symbol, start_date, end_date, timeframe)
            
            # Create data handler and manually set the data
            data_handler = BinanceDataHandler(use_cache=True) if exchange.lower() == 'binance' else CCXTDataHandler(use_cache=True)
            data_handler.symbols = [symbol]
            data_handler.data = {symbol: df}
            
        else:
            # Create data handler based on exchange
            logger.info(f"Creating data handler for {exchange}")
            if exchange.lower() == 'binance':
                data_handler = BinanceDataHandler(use_cache=True)
            elif exchange.lower() == 'ccxt':
                data_handler = CCXTDataHandler(use_cache=True)
            else:
                raise ValueError(f"Unsupported exchange: {exchange}")
            
            # Initialize data handler with date range
            data_handler.initialize(start_date=start_date, end_date=end_date)
            
            try:
                # Fetch historical data
                logger.info(f"Fetching historical data for {symbol}")
                if exchange.lower() == 'binance':
                    df = data_handler.get_historical_data(
                        symbol=binance_symbol,
                        timeframe=timeframe,
                        start_time=start_date,
                        end_time=end_date
                    )
                else:
                    df = data_handler.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_time=start_date,
                        end_time=end_date
                    )
                
                # Store symbol and data
                data_handler.symbols = [symbol]
                data_handler.data = {symbol: df}
                
            except Exception as e:
                logger.warning(f"Error fetching real data: {e}. Using generated sample data instead.")
                df = generate_sample_data(symbol, start_date, end_date, timeframe)
                data_handler.symbols = [symbol]
                data_handler.data = {symbol: df}
        
        # Explicitly set the strategy's data
        logger.info("Setting strategy data")
        strategy.set_data(df)
        
        # Create portfolio
        portfolio = Portfolio(initial_capital=initial_capital, symbols=[symbol])
        
        # Create execution handler and initialize it with commission
        execution_handler = SimulatedExecutionHandler()
        execution_handler.initialize(data_handler=data_handler, commission=commission)
        
        # Create and configure backtester
        logger.info("Creating backtester")
        backtester = Backtester(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution_handler,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            commission=commission
        )
        
        # Define the buy and sell functions that will generate signal events
        def buy_func(size=None, price=None, limit_price=None, stop_price=None, percent=None):
            logger.info(f"Buy signal received: symbol={symbol}, size={size}, price={price}, percent={percent}")
            from backtesting.engine.event import SignalEvent
            
            # Create signal event metadata
            metadata = {
                'size': size,
                'price': price,
                'limit_price': limit_price,
                'stop_price': stop_price,
                'percent': percent,
                'suggested_price': price
            }
            
            # Create a signal event
            signal = SignalEvent(
                timestamp=data_handler.current_datetime,
                symbol=symbol,
                signal_type='BUY',  # Use string instead of enum
                strength=1.0,
                metadata=metadata
            )
            
            # Add the signal to the event loop
            backtester.event_loop.add_event(signal)
            return signal
        
        def sell_func(size=None, price=None, limit_price=None, stop_price=None, percent=None):
            logger.info(f"Sell signal received: symbol={symbol}, size={size}, price={price}, percent={percent}")
            from backtesting.engine.event import SignalEvent
            
            # Create signal event metadata
            metadata = {
                'size': size,
                'price': price,
                'limit_price': limit_price,
                'stop_price': stop_price,
                'percent': percent,
                'suggested_price': price
            }
            
            # Create a signal event
            signal = SignalEvent(
                timestamp=data_handler.current_datetime,
                symbol=symbol,
                signal_type='SELL',  # Use string instead of enum
                strength=1.0,
                metadata=metadata
            )
            
            # Add the signal to the event loop
            backtester.event_loop.add_event(signal)
            return signal
        
        # Set the strategy's buy and sell functions after creating the backtester
        strategy.set_backtester_functions(buy_func, sell_func)
        
        # Run the backtest
        logger.info("Running backtest")
        results = backtester.run()
        
        # Process results into a standardized format
        logger.info("Processing backtest results")
        equity_curve = results.get('equity_curve', pd.Series())
        returns = results.get('returns', pd.Series())
        trades = results.get('trades', [])
        metrics = results.get('metrics', {})
        
        # Create BacktestResult object
        backtest_result = BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            trades=trades,
            metrics=metrics,
            strategy_code=strategy_code,
            strategy_params=strategy_params,
            config={
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'timeframe': timeframe,
                'commission': commission,
                'exchange': exchange,
                'used_sample_data': use_sample_data or 'sample_data' in locals()
            }
        )
        
        logger.info("Backtest completed successfully")
        return backtest_result
    
    except Exception as e:
        logger.error(f"Error during backtest: {e}", exc_info=True)
        raise RuntimeError(f"Backtest failed: {e}") from e

def test_runner():
    """
    Test the runner with a simple moving average crossover strategy.
    This is for development and testing purposes.
    """
    # Simple MA Crossover strategy code with debug logging
    strategy_code = """
import logging
from backtesting.strategy.base import Strategy
from backtesting.engine.event import MarketEvent

class DebugMovingAverageCrossStrategy(Strategy):
    def __init__(self, short_window=5, long_window=15):
        super().__init__()
        self.params = {
            'short_window': short_window,
            'long_window': long_window
        }
        self.ma_short = None
        self.ma_long = None
        self.logger = logging.getLogger(__name__)
        self.events_processed = 0
        self.signals_generated = 0
    
    def init(self):
        # Calculate moving averages
        short_window = self.params['short_window']
        long_window = self.params['long_window']
        
        print(f"Data columns: {self.data.columns}")
        print(f"Data shape: {self.data.shape}")
        print(f"First few rows: {self.data.head()}")
        
        self.ma_short = self.data['close'].rolling(window=short_window).mean()
        self.ma_long = self.data['close'].rolling(window=long_window).mean()
        
        # Print first few values of moving averages
        print(f"Short MA (first 20): {self.ma_short.head(20)}")
        print(f"Long MA (first 20): {self.ma_long.head(20)}")
    
    def next(self):
        # Skip if not enough data
        if self.current_index < self.params['long_window']:
            return
        
        # Print current index every 50 bars
        if self.current_index % 50 == 0:
            print(f"Processing bar {self.current_index}")
            
        # Current and previous values
        curr_short = self.ma_short.iloc[self.current_index]
        curr_long = self.ma_long.iloc[self.current_index]
        
        prev_short = self.ma_short.iloc[self.current_index - 1]
        prev_long = self.ma_long.iloc[self.current_index - 1]
        
        # Debug info every 50 bars
        if self.current_index % 50 == 0:
            print(f"Curr short: {curr_short}, Curr long: {curr_long}")
            print(f"Prev short: {prev_short}, Prev long: {prev_long}")
        
        # Golden cross (short MA crosses above long MA)
        if prev_short <= prev_long and curr_short > curr_long:
            if self.position <= 0:
                print(f"BUY SIGNAL at index {self.current_index}, date: {self.data.index[self.current_index]}")
                print(f"Short MA: {curr_short}, Long MA: {curr_long}")
                self.buy()
                self.signals_generated += 1
        
        # Death cross (short MA crosses below long MA)
        elif prev_short >= prev_long and curr_short < curr_long:
            if self.position > 0:
                print(f"SELL SIGNAL at index {self.current_index}, date: {self.data.index[self.current_index]}")
                print(f"Short MA: {curr_short}, Long MA: {curr_long}")
                self.sell()
                self.signals_generated += 1
    
    def on_data(self, event):
        # Count events processed
        self.events_processed += 1
        
        # Update current_index from data_handler
        if self.data_handler is not None:
            self.current_index = self.data_handler.current_index
            
        # Print debug info for every 50th event
        if self.events_processed % 50 == 0:
            print(f"Processed {self.events_processed} events, current index: {self.current_index}, signals: {self.signals_generated}")
        
        # Call the next method to process this event
        try:
            self.next()
        except Exception as e:
            print(f"Error in next(): {e}")
    
    def on_start(self):
        print("Strategy on_start called.")
        print(f"Data shape: {self.data.shape if self.data is not None else 'None'}")
        print(f"Strategy parameters: {self.params}")
        print(f"Current index at start: {self.current_index}")
    
    def on_finish(self):
        print("Strategy on_finish called.")
        print(f"Final index: {self.current_index}")
        print(f"Events processed: {self.events_processed}")
        print(f"Signals generated: {self.signals_generated}")
        print(f"Final position: {self.position}")
        
        # Close all positions at the end of the backtest
        if self.position > 0:
            print(f"Closing final position: {self.position_size} @ {self.data['close'].iloc[-1]}")
            self.sell()  # Sell all
"""
    
    # Configuration
    config = {
        'symbol': 'BTC/USDT',
        'start_date': '2022-01-01',
        'end_date': '2022-12-31',
        'initial_capital': 10000.0,
        'timeframe': '1d',
        'commission': 0.001,
        'exchange': 'binance',
        'strategy_params': {
            'short_window': 5,
            'long_window': 15
        },
        'use_sample_data': True  # Use generated data for testing
    }
    
    # Run backtest
    result = run_backtest(strategy_code, config)
    
    # Print results
    print(f"Total Return: {result.metrics.get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
    print(f"Total Trades: {len(result.trades)}")
    
    # Print equity curve for debugging
    if 'equity_curve' in result.__dict__ and result.equity_curve is not None:
        if len(result.equity_curve) > 0:
            print("\nEquity Curve (first 5 entries):")
            if isinstance(result.equity_curve, pd.Series):
                print(result.equity_curve.head())
            else:
                print(result.equity_curve[:5])
            
            print("\nEquity Curve (last 5 entries):")
            if isinstance(result.equity_curve, pd.Series):
                print(result.equity_curve.tail())
            else:
                print(result.equity_curve[-5:])
            
            # Calculate and print total return manually from the equity curve
            if isinstance(result.equity_curve, list) and len(result.equity_curve) > 1:
                first_value = result.equity_curve[0]['portfolio_value']
                last_value = result.equity_curve[-1]['portfolio_value']
                total_return = (last_value / first_value) - 1
                print(f"\nManually calculated total return: {total_return:.2%}")
                print(f"Starting portfolio value: ${first_value:.2f}")
                print(f"Ending portfolio value: ${last_value:.2f}")
        else:
            print("Equity curve is empty")
    else:
        print("No equity curve available")
    
    return result

if __name__ == "__main__":
    test_runner() 