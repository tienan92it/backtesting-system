import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtesting.strategy import Strategy, StrategyBase
from backtesting.strategy import MovingAverageCrossStrategy, RSIStrategy
from backtesting.strategy import MLMeanReversionStrategy


class MockStrategy(Strategy):
    """A simple mock strategy for testing."""
    
    def init(self):
        """Initialize the strategy."""
        self.initialized = True
    
    def next(self):
        """Process the next candle."""
        # Simple logic - buy when price increases, sell when price decreases
        if self.current_index > 0:
            prev_close = self.data['close'].iloc[self.current_index - 1]
            curr_close = self.data['close'].iloc[self.current_index]
            
            if curr_close > prev_close and self.position <= 0:
                self.buy()
            elif curr_close < prev_close and self.position > 0:
                self.sell()


class TestStrategyBase:
    """Test the Strategy base class functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        data = {
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            'low': [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
            'close': [105, 102, 103, 101, 99, 100, 103, 105, 106, 107],
            'volume': [1000, 1200, 1300, 1100, 1000, 900, 1100, 1200, 1300, 1400]
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def mock_strategy(self, sample_data):
        """Create a mock strategy with sample data."""
        strategy = MockStrategy()
        strategy.set_data(sample_data)
        
        # Set up mock buy/sell functions
        buy_orders = []
        sell_orders = []
        
        def mock_buy(size=None, price=None, limit_price=None, stop_price=None, percent=None):
            buy_orders.append({
                'timestamp': strategy.data.index[strategy.current_index],
                'price': price or strategy.data['close'].iloc[strategy.current_index],
                'size': size,
                'percent': percent
            })
        
        def mock_sell(size=None, price=None, limit_price=None, stop_price=None, percent=None):
            sell_orders.append({
                'timestamp': strategy.data.index[strategy.current_index],
                'price': price or strategy.data['close'].iloc[strategy.current_index],
                'size': size,
                'percent': percent
            })
        
        strategy.set_backtester_functions(mock_buy, mock_sell)
        return strategy, buy_orders, sell_orders
    
    def test_strategy_initialization(self, sample_data):
        """Test that a strategy initializes correctly."""
        strategy = MockStrategy()
        assert isinstance(strategy, Strategy)
        assert strategy.data is None
        assert strategy.current_index == 0
        
        strategy.set_data(sample_data)
        assert strategy.data is not None
        assert len(strategy.data) == len(sample_data)
    
    def test_strategy_buy_sell(self, mock_strategy):
        """Test that buy and sell methods work."""
        strategy, buy_orders, sell_orders = mock_strategy
        
        # Initialize the strategy
        strategy.init()
        assert strategy.initialized
        
        # Run next() for each bar
        for i in range(len(strategy.data)):
            strategy.current_index = i
            strategy.next()
        
        # Check that orders were placed
        assert len(buy_orders) > 0
        assert len(sell_orders) > 0
        
        # Check that we have the correct number of signals recorded
        assert len(strategy.signals) == len(buy_orders) + len(sell_orders)
    
    def test_crossover_detection(self, sample_data):
        """Test the crossover and crossunder detection."""
        # Create two series that cross
        series1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        series2 = pd.Series([5, 4, 3, 2, 1, 2, 3, 4, 5, 6])
        
        strategy = MockStrategy()
        strategy.set_data(sample_data)
        
        # We expect a crossover at index 4->5 (series1 crosses above series2)
        for i in range(len(series1)):
            strategy.current_index = i
            if i >= 5:
                assert strategy.crossover(series1, series2) == (i == 5)
            else:
                assert not strategy.crossover(series1, series2)
        
        # Test crossunder (series2 crosses below series1)
        series3 = pd.Series([5, 6, 7, 8, 9, 8, 7, 6, 5, 4])
        for i in range(len(series1)):
            strategy.current_index = i
            if i >= 5:
                assert strategy.crossunder(series3, series1) == (i == 5)
            else:
                assert not strategy.crossunder(series3, series1)
    
    def test_moving_average_cross_strategy(self, sample_data):
        """Test the MovingAverageCrossStrategy."""
        # This is a more integration-level test
        # For proper testing, we need more data points
        dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
        close_prices = np.sin(np.linspace(0, 10, 300)) * 10 + 100  # Sine wave + offset
        data = {
            'open': close_prices - 1,
            'high': close_prices + 2,
            'low': close_prices - 2,
            'close': close_prices,
            'volume': np.random.randint(1000, 2000, 300)
        }
        df = pd.DataFrame(data, index=dates)
        
        # Create the strategy
        strategy = MovingAverageCrossStrategy(short_window=20, long_window=50)
        strategy.set_data(df)
        
        # Mock buy/sell functions
        buy_orders = []
        sell_orders = []
        
        def mock_buy(size=None, price=None, limit_price=None, stop_price=None, percent=None):
            buy_orders.append({
                'timestamp': strategy.data.index[strategy.current_index],
                'price': price or strategy.data['close'].iloc[strategy.current_index],
                'size': size,
                'percent': percent
            })
        
        def mock_sell(size=None, price=None, limit_price=None, stop_price=None, percent=None):
            sell_orders.append({
                'timestamp': strategy.data.index[strategy.current_index],
                'price': price or strategy.data['close'].iloc[strategy.current_index],
                'size': size,
                'percent': percent
            })
        
        strategy.set_backtester_functions(mock_buy, mock_sell)
        
        # Initialize and run
        strategy.init()
        for i in range(len(df)):
            strategy.current_index = i
            strategy.update_portfolio(position=1 if buy_orders and not sell_orders else 0,
                                     position_size=1 if buy_orders and not sell_orders else 0,
                                     cash=10000,
                                     portfolio_value=10000)
            strategy.next()
        
        # Check that we have crosses and trades
        assert len(buy_orders) > 0
        assert len(sell_orders) > 0
        
        # Simple check - we should have approximately similar number of buys and sells
        # for a mean-reverting price series like a sine wave
        assert abs(len(buy_orders) - len(sell_orders)) <= 1 