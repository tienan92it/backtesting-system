"""
Broker interface for order execution.

This module provides an abstract Broker interface that defines
the methods for order execution in a trading system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

from backtesting.engine.event import OrderEvent, FillEvent


class Broker(ABC):
    """
    Abstract base class for a broker.
    
    The Broker class is responsible for:
    - Executing orders
    - Tracking fills
    - Handling commission and slippage
    - Maintaining account information
    """
    
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
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information such as balance, margin, etc.
        
        Returns:
        --------
        dict
            Dictionary containing account information
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
        --------
        dict
            Dictionary of current positions, keyed by symbol
        """
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str, timeframe: str, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get market data for a symbol.
        
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
        pass
