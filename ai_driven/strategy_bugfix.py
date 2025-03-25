"""
Strategy bugfix implementation for the backtesting system.
"""

import logging
import pandas as pd
import numpy as np
from backtesting.strategy.base import Strategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedMAStrategy(Strategy):
    """
    Fixed MA Crossover strategy with proper position tracking.
    Buys when short MA crosses above long MA and sells when it crosses below.
    """
    
    def __init__(self, short_window=5, long_window=20, position_size_pct=10):
        super().__init__()
        self.params = {
            'short_window': short_window,
            'long_window': long_window,
            'position_size_pct': position_size_pct
        }
        self.ma_short = None
        self.ma_long = None
        self.events_processed = 0
        self.signals_generated = 0
        self.crosses_detected = 0
        self.portfolio_updates = 0
    
    def init(self):
        """Calculate the moving averages."""
        # Get parameters
        short_window = self.params['short_window']
        long_window = self.params['long_window']
        
        # Log data properties
        logger.info(f"Data shape: {self.data.shape}")
        logger.info(f"Data columns: {self.data.columns}")
        logger.info(f"First few rows: {self.data.head()}")
        
        # Calculate moving averages
        self.ma_short = self.data['close'].rolling(window=short_window).mean()
        self.ma_long = self.data['close'].rolling(window=long_window).mean()
        
        # Log first few values
        logger.info(f"Short MA (first 5): {self.ma_short.head()}")
        logger.info(f"Long MA (first 5): {self.ma_long.head()}")
    
    def update_portfolio(self, position, position_size, cash, portfolio_value):
        """
        Track portfolio updates from the backtester.
        This method is called by the backtester after each step.
        """
        self.portfolio_updates += 1
        
        # Log portfolio updates periodically
        if self.portfolio_updates % 20 == 0 or self.portfolio_updates < 5:
            logger.info(f"Portfolio update #{self.portfolio_updates}: "
                     f"Position={position}, Size={position_size:.6f}, "
                     f"Cash=${cash:.2f}, Value=${portfolio_value:.2f}")
        
        # Update strategy's position tracking
        super().update_portfolio(position, position_size, cash, portfolio_value)
    
    def next(self):
        """Generate trading signals for the current bar."""
        # Skip if not enough data
        if self.current_index < self.params['long_window']:
            return
            
        # Log status every 50 bars or on specific events
        log_detailed = (self.current_index % 50 == 0) or (self.signals_generated > 0 and self.signals_generated < 5)
        
        # Get current values
        try:
            curr_short = self.ma_short.iloc[self.current_index]
            curr_long = self.ma_long.iloc[self.current_index]
            
            prev_short = self.ma_short.iloc[self.current_index - 1]
            prev_long = self.ma_long.iloc[self.current_index - 1]
        except IndexError:
            logger.warning(f"Index out of bounds at {self.current_index}, data length: {len(self.data)}")
            return
        
        # Check for NaN values
        if pd.isna(curr_short) or pd.isna(curr_long) or pd.isna(prev_short) or pd.isna(prev_long):
            if log_detailed:
                logger.warning(f"NaN values detected at index {self.current_index}")
            return
        
        if log_detailed:
            logger.info(f"Bar {self.current_index}: Date={self.data.index[self.current_index]}, "
                     f"Close={self.data['close'].iloc[self.current_index]:.2f}, "
                     f"Short MA={curr_short:.2f}, Long MA={curr_long:.2f}, "
                     f"Position={self.position}, Size={self.position_size:.6f}")
        
        # Check for golden cross (short MA crosses above long MA)
        if prev_short <= prev_long and curr_short > curr_long:
            self.crosses_detected += 1
            logger.info(f"GOLDEN CROSS at bar {self.current_index}: "
                     f"Short MA={curr_short:.2f}, Long MA={curr_long:.2f}, "
                     f"Position={self.position}, Size={self.position_size:.6f}")
            
            # Buy only if not already long
            if self.position <= 0:
                # Calculate position size as percentage of portfolio
                pct = self.params['position_size_pct']
                current_price = self.data['close'].iloc[self.current_index]
                
                logger.info(f"BUY SIGNAL at {self.data.index[self.current_index]}: "
                         f"Price=${current_price:.2f}, Using {pct}% of portfolio (${self.portfolio_value:.2f})")
                
                # Execute buy order
                self.buy(percent=pct)
                self.signals_generated += 1
            else:
                logger.info(f"Already long, not buying more. Position={self.position}, Size={self.position_size:.6f}")
        
        # Check for death cross (short MA crosses below long MA)
        elif prev_short >= prev_long and curr_short < curr_long:
            self.crosses_detected += 1
            logger.info(f"DEATH CROSS at bar {self.current_index}: "
                     f"Short MA={curr_short:.2f}, Long MA={curr_long:.2f}, "
                     f"Position={self.position}, Size={self.position_size:.6f}")
            
            # Sell only if we have a position
            if self.position > 0:
                current_price = self.data['close'].iloc[self.current_index]
                logger.info(f"SELL SIGNAL at {self.data.index[self.current_index]}: "
                         f"Price=${current_price:.2f}, Position={self.position}, Size={self.position_size:.6f}")
                
                # Execute sell order - sell entire position
                self.sell()
                self.signals_generated += 1
            else:
                logger.info(f"No position to sell. Position={self.position}")
    
    def on_start(self):
        """Called when the backtesting starts."""
        logger.info("Strategy starting")
        logger.info(f"Parameters: {self.params}")
        logger.info(f"Initial position: {self.position}")
        logger.info(f"Initial portfolio value: ${self.portfolio_value:.2f}")
    
    def on_finish(self):
        """Called when the backtesting finishes."""
        logger.info("Strategy finished")
        logger.info(f"Final position: {self.position}")
        logger.info(f"Final portfolio value: ${self.portfolio_value:.2f}")
        logger.info(f"Events processed: {self.events_processed}")
        logger.info(f"Crosses detected: {self.crosses_detected}")
        logger.info(f"Signals generated: {self.signals_generated}")
        logger.info(f"Portfolio updates: {self.portfolio_updates}")
        
        # Log warning if no trades executed
        if self.signals_generated == 0:
            logger.warning("No signals generated during the backtest")
    
    def on_data(self, event):
        """Process market data events."""
        self.events_processed += 1
        
        # Update current index
        if hasattr(self, 'data_handler') and self.data_handler is not None:
            self.current_index = self.data_handler.current_index
        
        # Log processing occasionally
        if self.events_processed % 100 == 0:
            logger.info(f"Processed {self.events_processed} events, "
                     f"current index: {self.current_index}, "
                     f"signals: {self.signals_generated}")
        
        # Call next method
        try:
            self.next()
        except Exception as e:
            logger.error(f"Error in next(): {str(e)}", exc_info=True)

# Convenience function to get strategy code as string
def get_strategy_code():
    """Return the strategy code as a string for use with run_backtest."""
    code = """
import logging
import pandas as pd
import numpy as np
from backtesting.strategy.base import Strategy

class FixedMAStrategy(Strategy):
    \"\"\"
    Fixed MA Crossover strategy with proper position tracking.
    Buys when short MA crosses above long MA and sells when it crosses below.
    \"\"\"
    
    def __init__(self, short_window=5, long_window=20, position_size_pct=10):
        super().__init__()
        self.params = {
            'short_window': short_window,
            'long_window': long_window,
            'position_size_pct': position_size_pct
        }
        self.ma_short = None
        self.ma_long = None
        self.events_processed = 0
        self.signals_generated = 0
        self.crosses_detected = 0
        self.portfolio_updates = 0
    
    def init(self):
        \"\"\"Calculate the moving averages.\"\"\"
        # Get parameters
        short_window = self.params['short_window']
        long_window = self.params['long_window']
        
        # Calculate moving averages
        self.ma_short = self.data['close'].rolling(window=short_window).mean()
        self.ma_long = self.data['close'].rolling(window=long_window).mean()
    
    def update_portfolio(self, position, position_size, cash, portfolio_value):
        \"\"\"
        Track portfolio updates from the backtester.
        This method is called by the backtester after each step.
        \"\"\"
        self.portfolio_updates += 1
        
        # Update strategy's position tracking
        super().update_portfolio(position, position_size, cash, portfolio_value)
    
    def next(self):
        \"\"\"Generate trading signals for the current bar.\"\"\"
        # Skip if not enough data
        if self.current_index < self.params['long_window']:
            return
        
        # Get current values
        try:
            curr_short = self.ma_short.iloc[self.current_index]
            curr_long = self.ma_long.iloc[self.current_index]
            
            prev_short = self.ma_short.iloc[self.current_index - 1]
            prev_long = self.ma_long.iloc[self.current_index - 1]
        except IndexError:
            return
        
        # Check for NaN values
        if pd.isna(curr_short) or pd.isna(curr_long) or pd.isna(prev_short) or pd.isna(prev_long):
            return
        
        # Check for golden cross (short MA crosses above long MA)
        if prev_short <= prev_long and curr_short > curr_long:
            self.crosses_detected += 1
            
            # Buy only if not already long
            if self.position <= 0:
                # Calculate position size as percentage of portfolio
                pct = self.params['position_size_pct']
                # Execute buy order
                self.buy(percent=pct)
                self.signals_generated += 1
        
        # Check for death cross (short MA crosses below long MA)
        elif prev_short >= prev_long and curr_short < curr_long:
            self.crosses_detected += 1
            
            # Sell only if we have a position
            if self.position > 0:
                # Execute sell order - sell entire position
                self.sell()
                self.signals_generated += 1
    
    def on_data(self, event):
        \"\"\"Process market data events.\"\"\"
        self.events_processed += 1
        
        # Update current index
        if hasattr(self, 'data_handler') and self.data_handler is not None:
            self.current_index = self.data_handler.current_index
        
        # Call next method
        try:
            self.next()
        except Exception as e:
            pass
"""
    return code 