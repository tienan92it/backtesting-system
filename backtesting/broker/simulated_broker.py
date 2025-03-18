"""
Simulated broker for backtesting.

This module provides a simulated broker implementation
for backtesting trading strategies.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np

from backtesting.broker.broker import Broker
from backtesting.engine.event import OrderEvent, FillEvent


class SimulatedBroker(Broker):
    """
    Simulated broker for backtesting.
    
    This broker simulates order execution with configurable
    commission and slippage models.
    """
    
    def __init__(self, data_handler, commission_model='percentage', 
                 commission_rate=0.001, slippage_model='fixed',
                 slippage_amount=0.0):
        """
        Initialize the simulated broker.
        
        Parameters:
        -----------
        data_handler : DataHandler
            The data handler providing market data
        commission_model : str, optional
            The commission model to use ('fixed', 'percentage', 'none')
        commission_rate : float, optional
            The commission rate to apply
        slippage_model : str, optional
            The slippage model to use ('fixed', 'percentage', 'none')
        slippage_amount : float, optional
            The slippage amount to apply
        """
        self.logger = logging.getLogger(__name__)
        self.data_handler = data_handler
        self.commission_model = commission_model
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.slippage_amount = slippage_amount
        
        self.current_positions = {}  # Symbol -> quantity
        self.current_balances = {"cash": 0.0}  # Currency -> amount
        self.fill_history = []
        
        self.logger.info(
            f"SimulatedBroker initialized with commission model: {commission_model}, "
            f"rate: {commission_rate}, slippage model: {slippage_model}, "
            f"amount: {slippage_amount}"
        )
    
    def execute_order(self, order_event: OrderEvent) -> Optional[FillEvent]:
        """
        Execute an order and generate a fill event.
        
        Parameters:
        -----------
        order_event : OrderEvent
            The order event to execute
            
        Returns:
        --------
        FillEvent or None
            The fill event if the order was executed, None otherwise
        """
        self.logger.debug(f"Executing order: {order_event}")
        
        # Get current market data
        if not self.data_handler.current_bar:
            self.logger.warning("No current bar data available to execute order")
            return None
        
        current_data = self.data_handler.current_bar
        if order_event.symbol not in current_data:
            self.logger.warning(f"No data for symbol {order_event.symbol} in current bar")
            return None
        
        # Determine execution price with slippage
        execution_price = self._calculate_execution_price(
            order_event.symbol, 
            order_event.direction,
            order_event.order_type,
            order_event.price
        )
        
        # Calculate commission
        commission = self._calculate_commission(execution_price, order_event.quantity)
        
        # Create fill event
        fill_event = FillEvent(
            timestamp=self.data_handler.current_datetime,
            symbol=order_event.symbol,
            exchange="SIMULATED",
            quantity=order_event.quantity,
            direction=order_event.direction,
            fill_price=execution_price,
            commission=commission
        )
        
        # Update positions
        self._update_positions(fill_event)
        
        # Log the fill
        self.fill_history.append(fill_event)
        self.logger.info(f"Order filled: {fill_event}")
        
        return fill_event
    
    def _calculate_execution_price(self, symbol: str, direction: str, 
                                  order_type: str, limit_price: Optional[float] = None) -> float:
        """
        Calculate the execution price with slippage.
        
        Parameters:
        -----------
        symbol : str
            The symbol to execute
        direction : str
            The direction of the order (BUY or SELL)
        order_type : str
            The type of order (MARKET, LIMIT, etc.)
        limit_price : float, optional
            The limit price for limit orders
            
        Returns:
        --------
        float
            The execution price with slippage
        """
        current_data = self.data_handler.current_bar[symbol]
        
        # Use the close price as the base price for simplicity
        # In a more advanced implementation, you might use bid/ask prices
        base_price = current_data['close']
        
        if order_type == 'LIMIT' and limit_price is not None:
            # For limit orders, check if the price is favorable
            if (direction == 'BUY' and base_price > limit_price) or \
               (direction == 'SELL' and base_price < limit_price):
                self.logger.debug(f"Limit order not executed: {direction} at {limit_price}, market price: {base_price}")
                return None
            
            # Use the limit price if it's more favorable than the market price
            if direction == 'BUY':
                base_price = min(base_price, limit_price)
            else:
                base_price = max(base_price, limit_price)
        
        # Apply slippage
        if self.slippage_model == 'fixed':
            # Add/subtract fixed amount
            slippage = self.slippage_amount
            execution_price = base_price + slippage if direction == 'BUY' else base_price - slippage
        elif self.slippage_model == 'percentage':
            # Apply percentage slippage
            slippage = base_price * self.slippage_amount
            execution_price = base_price + slippage if direction == 'BUY' else base_price - slippage
        else:  # No slippage
            execution_price = base_price
        
        return execution_price
    
    def _calculate_commission(self, price: float, quantity: float) -> float:
        """
        Calculate the commission for a trade.
        
        Parameters:
        -----------
        price : float
            The execution price
        quantity : float
            The quantity traded
            
        Returns:
        --------
        float
            The commission amount
        """
        if self.commission_model == 'fixed':
            return self.commission_rate
        elif self.commission_model == 'percentage':
            return price * quantity * self.commission_rate
        else:  # No commission
            return 0.0
    
    def _update_positions(self, fill_event: FillEvent) -> None:
        """
        Update positions based on a fill event.
        
        Parameters:
        -----------
        fill_event : FillEvent
            The fill event to apply
        """
        symbol = fill_event.symbol
        
        # Update position
        if symbol not in self.current_positions:
            self.current_positions[symbol] = 0.0
        
        # Adjust position based on direction
        position_change = fill_event.quantity
        if fill_event.direction == 'SELL':
            position_change = -position_change
        
        self.current_positions[symbol] += position_change
        
        # Update cash balance
        trade_value = fill_event.fill_price * fill_event.quantity
        if fill_event.direction == 'BUY':
            self.current_balances["cash"] -= trade_value
        else:
            self.current_balances["cash"] += trade_value
        
        # Subtract commission
        self.current_balances["cash"] -= fill_event.commission
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information such as balance, margin, etc.
        
        Returns:
        --------
        dict
            Dictionary containing account information
        """
        total_position_value = 0.0
        for symbol, quantity in self.current_positions.items():
            if quantity != 0 and symbol in self.data_handler.current_bar:
                price = self.data_handler.current_bar[symbol]['close']
                total_position_value += price * quantity
        
        total_equity = self.current_balances["cash"] + total_position_value
        
        return {
            "cash": self.current_balances["cash"],
            "position_value": total_position_value,
            "total_equity": total_equity
        }
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
        --------
        dict
            Dictionary of current positions, keyed by symbol
        """
        positions = {}
        for symbol, quantity in self.current_positions.items():
            if quantity != 0:
                current_price = self.data_handler.current_bar.get(symbol, {}).get('close', 0)
                position_value = quantity * current_price
                positions[symbol] = {
                    "quantity": quantity,
                    "current_price": current_price,
                    "position_value": position_value
                }
        
        return positions
    
    def get_market_data(self, symbol: str, timeframe: str, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get market data for a symbol (delegates to data handler).
        
        Parameters:
        -----------
        symbol : str
            The symbol to get market data for
        timeframe : str
            The timeframe to get market data for (e.g., '1m', '1h', '1d')
        start_date : datetime, optional
            The start date for the market data
        end_date : datetime, optional
            The end date for the market data
            
        Returns:
        --------
        dict
            Dictionary containing market data
        """
        return self.data_handler.get_data(symbol, timeframe, start_date, end_date)
    
    def set_initial_balance(self, amount: float, currency: str = "cash") -> None:
        """
        Set the initial balance for the broker account.
        
        Parameters:
        -----------
        amount : float
            The initial balance amount
        currency : str, optional
            The currency of the balance
        """
        self.current_balances[currency] = amount
        self.logger.info(f"Initial balance set to {amount} {currency}")
    
    def get_fill_history(self) -> List[FillEvent]:
        """
        Get the history of all fills.
        
        Returns:
        --------
        list
            List of FillEvent objects
        """
        return self.fill_history
