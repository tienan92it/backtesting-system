"""
Portfolio tracking for backtesting.

This module provides a Portfolio class that tracks positions, cash balances,
and equity curves throughout a backtest.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime

from backtesting.engine.event import FillEvent, OrderEvent

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Portfolio class for tracking positions, cash balances, and equity curves.
    
    The Portfolio class is responsible for:
    - Tracking current positions for all symbols
    - Managing cash balances
    - Calculating equity values and returns
    - Recording historical positions and values
    - Generating orders based on signals
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        symbols: Optional[List[str]] = None
    ):
        """
        Initialize the portfolio with initial capital and symbols.
        
        Parameters:
        -----------
        initial_capital : float
            The initial capital in the base currency (e.g., USD, USDT)
        symbols : list of str, optional
            The list of symbols to track in the portfolio
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.symbols = symbols or []
        
        # Current positions for each symbol (quantity)
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        
        # Current position values for each symbol
        self.position_values = {symbol: 0.0 for symbol in self.symbols}
        
        # Current prices for each symbol
        self.current_prices = {symbol: 0.0 for symbol in self.symbols}
        
        # Historical data tracking
        self.equity_curve = []
        self.position_history = []
        self.trades = []
        
        # Current portfolio state
        self.portfolio_value = initial_capital
        self.returns = []
        
        # Trade statistics
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.trade_count = 0
        
        logger.info(f"Portfolio initialized with ${initial_capital:.2f}")
    
    def update_fill(self, fill_event: FillEvent) -> None:
        """
        Updates the portfolio based on a fill event.
        
        Parameters:
        -----------
        fill_event : FillEvent
            The fill event containing fill information
        """
        # Extract relevant information from fill
        symbol = fill_event.symbol
        fill_price = fill_event.fill_price
        quantity = fill_event.quantity
        direction = fill_event.direction
        commission = fill_event.commission
        
        # Ensure symbol is in the portfolio
        if symbol not in self.positions:
            self.positions[symbol] = 0.0
            self.position_values[symbol] = 0.0
            self.current_prices[symbol] = 0.0
            if symbol not in self.symbols:
                self.symbols.append(symbol)
        
        # Update position based on direction
        position_change = quantity
        if direction == 'SELL':
            position_change = -quantity
        
        # Calculate cost of trade
        cost = fill_price * quantity
        if direction == 'BUY':
            # Buy: reduce capital by cost + commission
            self.current_capital -= (cost + commission)
        else:
            # Sell: increase capital by cost - commission
            self.current_capital += (cost - commission)
        
        # Update position
        self.positions[symbol] += position_change
        self.current_prices[symbol] = fill_price
        self.position_values[symbol] = self.positions[symbol] * fill_price
        
        # Update statistics
        self.total_commission += commission
        self.trade_count += 1
        
        # Record trade
        trade = {
            'timestamp': fill_event.timestamp,
            'symbol': symbol,
            'direction': direction,
            'price': fill_price,
            'quantity': quantity,
            'commission': commission,
            'profit': 0.0  # Will be calculated on exit trades
        }
        
        # Calculate profit for sell trades (exits)
        if direction == 'SELL':
            # Look for corresponding buy trade for accurate P&L
            # This is a simple approach; in practice, need a more sophisticated matching
            # (e.g., FIFO, LIFO, or average cost)
            for i in range(len(self.trades) - 1, -1, -1):
                prev_trade = self.trades[i]
                if (prev_trade['symbol'] == symbol and 
                    prev_trade['direction'] == 'BUY' and
                    prev_trade['profit'] == 0.0):  # Hasn't been matched yet
                    
                    # Calculate profit for this exit on the matched entry
                    entry_price = prev_trade['price']
                    entry_quantity = prev_trade['quantity']
                    match_quantity = min(quantity, entry_quantity)
                    
                    # P&L = (exit price - entry price) * quantity - commissions
                    profit = (fill_price - entry_price) * match_quantity - (
                        prev_trade['commission'] * (match_quantity / entry_quantity) +
                        commission * (match_quantity / quantity)
                    )
                    
                    # Update the previous trade's profit (partially if needed)
                    if match_quantity == entry_quantity:
                        prev_trade['profit'] = profit
                    else:
                        # Handle partial fills (simplified approach)
                        prev_trade['profit'] = profit * (match_quantity / entry_quantity)
                        prev_trade['quantity'] = entry_quantity - match_quantity
                    
                    # Update current trade's profit
                    trade['profit'] = profit
                    
                    # If we've matched all the quantity, break
                    quantity -= match_quantity
                    if quantity <= 0:
                        break
        
        self.trades.append(trade)
        
        # Log the fill
        logger.info(
            f"Fill: {direction} {quantity} {symbol} @ {fill_price:.2f}, "
            f"Commission: ${commission:.2f}, Position: {self.positions[symbol]}, "
            f"Capital: ${self.current_capital:.2f}"
        )
    
    def update_market(self, timestamp: datetime, market_data: Dict[str, Dict[str, float]]) -> None:
        """
        Updates the portfolio based on new market data.
        
        Parameters:
        -----------
        timestamp : datetime
            The timestamp of the market data
        market_data : dict
            Dictionary of market data for each symbol
            {symbol: {'open': x, 'high': y, 'low': z, 'close': w, 'volume': v}}
        """
        # Update current prices for each symbol
        for symbol, data in market_data.items():
            if symbol in self.positions:
                self.current_prices[symbol] = data['close']
                self.position_values[symbol] = self.positions[symbol] * data['close']
        
        # Calculate total portfolio value
        total_position_value = sum(self.position_values.values())
        self.portfolio_value = self.current_capital + total_position_value
        
        # Record equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': self.portfolio_value,
            'cash': self.current_capital,
            'positions_value': total_position_value
        })
        
        # Record position history
        position_record = {
            'timestamp': timestamp,
            'cash': self.current_capital
        }
        # Add position for each symbol
        for symbol in self.symbols:
            position_record[f'{symbol}_position'] = self.positions.get(symbol, 0.0)
            position_record[f'{symbol}_value'] = self.position_values.get(symbol, 0.0)
        
        self.position_history.append(position_record)
        
        # Calculate return (if we have at least 2 data points)
        if len(self.equity_curve) >= 2:
            prev_value = self.equity_curve[-2]['portfolio_value']
            current_value = self.portfolio_value
            period_return = (current_value / prev_value) - 1.0
            self.returns.append(period_return)
    
    def generate_signals(self, timestamp: datetime, signals: Dict[str, Any]) -> List[OrderEvent]:
        """
        Generate orders based on signals.
        
        Parameters:
        -----------
        timestamp : datetime
            The timestamp of the signals
        signals : dict
            Dictionary of signals for each symbol
            
        Returns:
        --------
        list
            List of OrderEvent objects
        """
        # This would be implemented based on the signal format and order generation logic
        # For now, this is a placeholder that would be customized based on the strategy signals
        pass
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get the equity curve as a pandas DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the equity curve
        """
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        return df
    
    def get_position_history(self) -> pd.DataFrame:
        """
        Get the position history as a pandas DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the position history
        """
        if not self.position_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.position_history)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        return df
    
    def get_trades(self) -> pd.DataFrame:
        """
        Get the trades as a pandas DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the trades
        """
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        return df
    
    def get_current_positions(self) -> Dict[str, float]:
        """
        Get the current positions.
        
        Returns:
        --------
        dict
            Dictionary of current positions
        """
        return self.positions.copy()
    
    def get_portfolio_value(self) -> float:
        """
        Get the current portfolio value.
        
        Returns:
        --------
        float
            The current portfolio value
        """
        return self.portfolio_value
    
    def get_current_capital(self) -> float:
        """
        Get the current capital.
        
        Returns:
        --------
        float
            The current capital
        """
        return self.current_capital
    
    def get_returns(self) -> List[float]:
        """
        Get the returns.
        
        Returns:
        --------
        list
            List of returns
        """
        return self.returns.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get portfolio statistics.
        
        Returns:
        --------
        dict
            Dictionary of portfolio statistics
        """
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'final_portfolio_value': self.portfolio_value,
            'total_return': (self.portfolio_value / self.initial_capital) - 1.0,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'trade_count': self.trade_count
        }
