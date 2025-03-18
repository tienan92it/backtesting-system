"""
Execution handler interface for order execution.

This module provides an abstract ExecutionHandler interface that defines
the methods for order execution in a trading system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

from backtesting.engine.event import OrderEvent, FillEvent


class ExecutionHandler(ABC):
    """
    Abstract base class for execution handling.
    
    The ExecutionHandler class is responsible for:
    - Executing orders
    - Simulating market fills
    - Handling commission and slippage
    """
    
    @abstractmethod
    def initialize(self, data_handler, commission: float = 0.001, slippage: float = 0.0) -> None:
        """
        Initialize the execution handler.
        
        Parameters:
        -----------
        data_handler : DataHandler
            The data handler to use for data access
        commission : float, optional
            The commission rate to apply
        slippage : float, optional
            The slippage rate to apply
        """
        pass
    
    @abstractmethod
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
        pass 