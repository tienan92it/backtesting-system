import pandas as pd
import numpy as np
from typing import Dict, Any

from backtesting.strategy.base import Strategy


class MovingAverageCrossStrategy(Strategy):
    """
    A simple moving average crossover strategy.
    Buys when short MA crosses above long MA (golden cross).
    Sells when short MA crosses below long MA (death cross).
    """
    
    def __init__(self, short_window: int = 50, long_window: int = 200):
        """
        Initialize the strategy with MA periods.
        
        Parameters:
        -----------
        short_window : int
            Period for the short moving average.
        long_window : int
            Period for the long moving average.
        """
        super().__init__()
        self.params = {
            'short_window': short_window,
            'long_window': long_window
        }
        self.ma_short = None
        self.ma_long = None
    
    def init(self) -> None:
        """
        Calculate moving averages.
        Called once at the start of the backtest.
        """
        # Get parameters
        short_window = self.params['short_window']
        long_window = self.params['long_window']
        
        # Calculate moving averages
        self.ma_short = self.data['close'].rolling(window=short_window).mean()
        self.ma_long = self.data['close'].rolling(window=long_window).mean()
        
        # Give the moving averages descriptive names for the results
        self.ma_short.name = f'MA({short_window})'
        self.ma_long.name = f'MA({long_window})'
    
    def next(self) -> None:
        """
        Check for crossover events and place trades.
        Called for each new candle.
        """
        # Skip if we don't have enough data yet
        if self.current_index < self.params['long_window']:
            return
        
        # Check for golden cross (short MA crosses above long MA)
        if self.crossover(self.ma_short, self.ma_long):
            # If we're not already in a long position
            if self.position <= 0:
                self.buy()  # Buy with all available cash
        
        # Check for death cross (short MA crosses below long MA)
        elif self.crossunder(self.ma_short, self.ma_long):
            # If we're in a long position
            if self.position > 0:
                self.sell()  # Sell entire position
    
    def on_start(self) -> None:
        """
        Called when the backtest starts.
        """
        print(f"Starting backtest with short MA: {self.params['short_window']}, long MA: {self.params['long_window']}")
    
    def on_finish(self) -> None:
        """
        Called when the backtest finishes.
        """
        print(f"Finished backtest with {len(self.signals)} signals generated")
    
    def on_trade(self, trade: Dict[str, Any]) -> None:
        """
        Called after a trade is executed.
        
        Parameters:
        -----------
        trade : dict
            Dictionary with trade information.
        """
        action = "BUY" if trade['type'] == 'buy' else "SELL"
        price = trade.get('price', 'market')
        size = trade.get('size', 'all')
        print(f"Trade executed: {action} {size} at {price}")


class RSIStrategy(Strategy):
    """
    A simple RSI-based strategy.
    Buys when RSI is below oversold level (e.g., 30).
    Sells when RSI is above overbought level (e.g., 70).
    """
    
    def __init__(self, rsi_window: int = 14, oversold: int = 30, overbought: int = 70):
        """
        Initialize the strategy with RSI parameters.
        
        Parameters:
        -----------
        rsi_window : int
            Period for the RSI calculation.
        oversold : int
            Level below which the asset is considered oversold.
        overbought : int
            Level above which the asset is considered overbought.
        """
        super().__init__()
        self.params = {
            'rsi_window': rsi_window,
            'oversold': oversold,
            'overbought': overbought
        }
        self.rsi = None
    
    def init(self) -> None:
        """
        Calculate RSI.
        Called once at the start of the backtest.
        """
        # Calculate price changes
        delta = self.data['close'].diff()
        
        # Calculate gains and losses
        gains = delta.copy()
        gains[gains < 0] = 0
        losses = -delta.copy()
        losses[losses < 0] = 0
        
        # Calculate average gains and losses
        window = self.params['rsi_window']
        avg_gain = gains.rolling(window=window).mean()
        avg_loss = losses.rolling(window=window).mean()
        
        # Calculate relative strength and RSI
        rs = avg_gain / avg_loss
        self.rsi = 100 - (100 / (1 + rs))
        self.rsi.name = f'RSI({window})'
    
    def next(self) -> None:
        """
        Check RSI levels and place trades.
        Called for each new candle.
        """
        # Skip if we don't have enough data yet
        if self.current_index < self.params['rsi_window']:
            return
        
        current_rsi = self.rsi[self.current_index]
        
        # If RSI is below oversold level and we're not already in a position
        if current_rsi < self.params['oversold'] and self.position <= 0:
            self.buy()
        
        # If RSI is above overbought level and we're in a position
        elif current_rsi > self.params['overbought'] and self.position > 0:
            self.sell() 