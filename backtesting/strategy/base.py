from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union, List, Callable


class Strategy(ABC):
    """
    Base class for all trading strategies.
    
    Strategy classes should implement:
    - init(): Initialize indicators or models
    - next(): Generate trading signals for each new candle
    
    The strategy has access to:
    - self.data: DataFrame of price data
    - self.position: Current position information
    - self.buy() and self.sell() methods to place orders
    """
    
    def __init__(self):
        """
        Initialize the strategy object.
        This is called once when the strategy is first created.
        """
        self.data = None
        self.current_index = 0
        self.position = 0
        self.position_size = 0
        self.cash = 0
        self.portfolio_value = 0
        self.signals = []
        
        # References to the backtester's order methods (will be set by backtester)
        self._buy_func = None
        self._sell_func = None
        
        # User-defined parameters
        self.params = {}
    
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Set the data for the strategy.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with OHLCV data.
        """
        self.data = data
    
    def set_backtester_functions(self, buy_func: Callable, sell_func: Callable) -> None:
        """
        Set the backtester's buy and sell functions.
        These will be called by the strategy's buy() and sell() methods.
        
        Parameters:
        -----------
        buy_func : callable
            Function to execute a buy order.
        sell_func : callable
            Function to execute a sell order.
        """
        self._buy_func = buy_func
        self._sell_func = sell_func
    
    def update_portfolio(self, position: float, position_size: float, 
                         cash: float, portfolio_value: float) -> None:
        """
        Update the strategy's portfolio information.
        Called by the backtester after each step.
        
        Parameters:
        -----------
        position : float
            Current position direction (1 for long, -1 for short, 0 for flat).
        position_size : float
            Current position size in units of the asset.
        cash : float
            Available cash.
        portfolio_value : float
            Total portfolio value.
        """
        self.position = position
        self.position_size = position_size
        self.cash = cash
        self.portfolio_value = portfolio_value
    
    def buy(self, size: Optional[float] = None, 
            price: Optional[float] = None, 
            limit_price: Optional[float] = None,
            stop_price: Optional[float] = None,
            percent: Optional[float] = None) -> None:
        """
        Place a buy order.
        
        Parameters:
        -----------
        size : float, optional
            Size of the position to buy in units of the asset.
            If None, will use all available cash.
        price : float, optional
            Price to buy at. If None, will use market order.
        limit_price : float, optional
            Limit price for limit order.
        stop_price : float, optional
            Stop price for stop order.
        percent : float, optional
            Percentage of available cash to use (0-100).
        """
        if self._buy_func is None:
            raise RuntimeError("Strategy not properly initialized with backtester functions")
        
        # Record signal
        self.signals.append({
            'timestamp': self.data.index[self.current_index],
            'type': 'buy',
            'size': size,
            'price': price,
            'limit_price': limit_price,
            'stop_price': stop_price,
            'percent': percent
        })
        
        # Execute the order
        return self._buy_func(size, price, limit_price, stop_price, percent)
    
    def sell(self, size: Optional[float] = None, 
             price: Optional[float] = None, 
             limit_price: Optional[float] = None,
             stop_price: Optional[float] = None,
             percent: Optional[float] = None) -> None:
        """
        Place a sell order.
        
        Parameters:
        -----------
        size : float, optional
            Size of the position to sell in units of the asset.
            If None, will sell entire position.
        price : float, optional
            Price to sell at. If None, will use market order.
        limit_price : float, optional
            Limit price for limit order.
        stop_price : float, optional
            Stop price for stop order.
        percent : float, optional
            Percentage of position to sell (0-100).
        """
        if self._sell_func is None:
            raise RuntimeError("Strategy not properly initialized with backtester functions")
        
        # Record signal
        self.signals.append({
            'timestamp': self.data.index[self.current_index],
            'type': 'sell',
            'size': size,
            'price': price,
            'limit_price': limit_price,
            'stop_price': stop_price,
            'percent': percent
        })
        
        # Execute the order
        return self._sell_func(size, price, limit_price, stop_price, percent)
    
    @abstractmethod
    def init(self) -> None:
        """
        Initialize the strategy.
        This is called once with full data before backtesting.
        It's the place to calculate indicators or train ML models.
        """
        pass
    
    @abstractmethod
    def next(self) -> None:
        """
        Process the next candle.
        This is called for each new candle during backtesting.
        It's where trading decisions should be made.
        """
        pass
    
    def on_start(self) -> None:
        """
        Called when the backtesting starts.
        Optional hook that can be overridden by child classes.
        """
        pass
    
    def on_finish(self) -> None:
        """
        Called when the backtesting finishes.
        Optional hook that can be overridden by child classes.
        """
        pass
    
    def on_trade(self, trade: Dict[str, Any]) -> None:
        """
        Called after a trade is executed.
        Optional hook that can be overridden by child classes.
        
        Parameters:
        -----------
        trade : dict
            Dictionary with trade information.
        """
        pass
    
    def get_current_price(self) -> float:
        """
        Get the current close price.
        
        Returns:
        --------
        float
            Current close price.
        """
        return self.data.iloc[self.current_index]['close']
    
    def get_current_bar(self) -> pd.Series:
        """
        Get the current price bar.
        
        Returns:
        --------
        pd.Series
            Current price bar (open, high, low, close, volume)
        """
        return self.data.iloc[self.current_index]
    
    def crossover(self, series1: Union[pd.Series, np.ndarray, List[float]],
                 series2: Union[pd.Series, np.ndarray, List[float]]) -> bool:
        """
        Check if series1 crosses above series2 at the current index.
        
        Parameters:
        -----------
        series1 : array-like
            First series.
        series2 : array-like
            Second series.
            
        Returns:
        --------
        bool
            True if series1 crosses above series2, False otherwise.
        """
        # Need at least 2 values to check for crossover
        if self.current_index < 1:
            return False
        
        # Get current and previous values
        if isinstance(series1, (pd.Series, np.ndarray)):
            s1_curr = series1[self.current_index]
            s1_prev = series1[self.current_index - 1]
        else:
            s1_curr = series1[self.current_index]
            s1_prev = series1[self.current_index - 1]
            
        if isinstance(series2, (pd.Series, np.ndarray)):
            s2_curr = series2[self.current_index]
            s2_prev = series2[self.current_index - 1]
        else:
            s2_curr = series2[self.current_index]
            s2_prev = series2[self.current_index - 1]
        
        # Check for crossover
        return s1_prev <= s2_prev and s1_curr > s2_curr
    
    def crossunder(self, series1: Union[pd.Series, np.ndarray, List[float]],
                  series2: Union[pd.Series, np.ndarray, List[float]]) -> bool:
        """
        Check if series1 crosses below series2 at the current index.
        
        Parameters:
        -----------
        series1 : array-like
            First series.
        series2 : array-like
            Second series.
            
        Returns:
        --------
        bool
            True if series1 crosses below series2, False otherwise.
        """
        # Need at least 2 values to check for crossunder
        if self.current_index < 1:
            return False
        
        # Get current and previous values
        if isinstance(series1, (pd.Series, np.ndarray)):
            s1_curr = series1[self.current_index]
            s1_prev = series1[self.current_index - 1]
        else:
            s1_curr = series1[self.current_index]
            s1_prev = series1[self.current_index - 1]
            
        if isinstance(series2, (pd.Series, np.ndarray)):
            s2_curr = series2[self.current_index]
            s2_prev = series2[self.current_index - 1]
        else:
            s2_curr = series2[self.current_index]
            s2_prev = series2[self.current_index - 1]
        
        # Check for crossunder
        return s1_prev >= s2_prev and s1_curr < s2_curr


class StrategyBase(Strategy):
    """
    Convenience class that implements empty init() and next() methods.
    Users can inherit from this class instead of Strategy if they only
    want to implement one of the methods.
    """
    
    def init(self) -> None:
        """Empty initialization."""
        pass
    
    def next(self) -> None:
        """Empty next method."""
        pass
