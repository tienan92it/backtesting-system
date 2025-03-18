"""
Simulated execution handler for backtesting.

This module provides a simulated execution handler implementation
for backtesting trading strategies.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from backtesting.execution.execution_handler import ExecutionHandler
from backtesting.engine.event import OrderEvent, FillEvent


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulated execution handler for backtesting.
    
    This handler simulates order execution with configurable
    commission and slippage models.
    """
    
    def __init__(self):
        """Initialize the simulated execution handler."""
        self.logger = logging.getLogger(__name__)
        self.data_handler = None
        self.commission = 0.001  # Default 0.1% commission
        self.slippage = 0.0      # Default no slippage
    
    def initialize(self, data_handler, commission: float = 0.001, slippage: float = 0.0) -> None:
        """
        Initialize the execution handler.
        
        Parameters:
        -----------
        data_handler : DataHandler
            The data handler to use for data access
        commission : float, optional
            The commission rate to apply (e.g., 0.001 for 0.1%)
        slippage : float, optional
            The slippage rate to apply (e.g., 0.001 for 0.1%)
        """
        self.data_handler = data_handler
        self.commission = commission
        self.slippage = slippage
        
        self.logger.info(
            f"SimulatedExecutionHandler initialized with commission: {commission}, "
            f"slippage: {slippage}"
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
        
        # Check if data handler is initialized
        if not self.data_handler:
            self.logger.warning("Data handler not initialized, cannot execute order")
            return None
            
        # Check if current_bar exists and is not None
        if not hasattr(self.data_handler, 'current_bar') or self.data_handler.current_bar is None:
            self.logger.warning("No current bar available in data handler, cannot execute order")
            return None
        
        # Check if current_datetime exists and is not None
        if not hasattr(self.data_handler, 'current_datetime') or self.data_handler.current_datetime is None:
            self.logger.warning("No current datetime available in data handler, cannot execute order")
            return None
        
        # Get latest data for the symbol
        symbol = order_event.symbol
        if symbol not in self.data_handler.current_bar:
            self.logger.warning(f"No data available for symbol: {symbol} in current bar")
            return None
        
        # Get current bar data
        bar_data = self.data_handler.current_bar[symbol]
        
        # Determine execution price
        try:
            execution_price = self._calculate_execution_price(
                bar_data, 
                order_event.direction, 
                order_event.order_type,
                order_event.price
            )
            
            if execution_price is None:
                self.logger.debug(f"Order not executed (limit price not reached): {order_event}")
                return None
            
            # Calculate commission
            commission = self._calculate_commission(execution_price, order_event.quantity)
            
            # Create fill event
            fill_event = FillEvent(
                timestamp=self.data_handler.current_datetime,
                symbol=symbol,
                quantity=order_event.quantity,
                direction=order_event.direction,
                fill_price=execution_price,
                commission=commission,
                order_id=order_event.order_id
            )
            
            self.logger.info(
                f"Order filled: {fill_event.direction} {fill_event.quantity} {fill_event.symbol} @ "
                f"{fill_event.fill_price:.6f}, Commission: {fill_event.commission:.6f}"
            )
            
            return fill_event
            
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return None
    
    def _calculate_execution_price(self, bar_data: Dict[str, float], direction: str, 
                                  order_type: str, limit_price: Optional[float] = None) -> Optional[float]:
        """
        Calculate the execution price with slippage.
        
        Parameters:
        -----------
        bar_data : dict
            The current bar data for the symbol
        direction : str
            The direction of the order (BUY or SELL)
        order_type : str
            The type of order (MARKET or LIMIT)
        limit_price : float, optional
            The limit price for limit orders
            
        Returns:
        --------
        float or None
            The execution price with slippage, or None if the order cannot be executed
        """
        # Use the close price as the base price
        base_price = bar_data['close']
        
        # Handle limit orders
        if order_type == 'LIMIT' and limit_price is not None:
            # Check if limit price is reached
            if (direction == 'BUY' and base_price > limit_price) or \
               (direction == 'SELL' and base_price < limit_price):
                return None  # Limit price not reached
            
            # Use limit price if it's more favorable than market price
            if direction == 'BUY':
                base_price = min(base_price, limit_price)
            else:  # SELL
                base_price = max(base_price, limit_price)
        
        # Apply slippage
        if direction == 'BUY':
            execution_price = base_price * (1 + self.slippage)
        else:  # SELL
            execution_price = base_price * (1 - self.slippage)
        
        return execution_price
    
    def _calculate_commission(self, price: float, quantity: float) -> float:
        """
        Calculate commission for a trade.
        
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
        # Simple percentage commission model
        return price * quantity * self.commission 