from enum import Enum
from typing import Dict, Any, Optional, Union
import pandas as pd
from datetime import datetime


class EventType(Enum):
    """Enum of different event types in the system."""
    MARKET = 'MARKET'           # New market data (tick/bar)
    SIGNAL = 'SIGNAL'           # Strategy signal (buy/sell)
    ORDER = 'ORDER'             # Order to be executed
    FILL = 'FILL'               # Order has been filled
    PORTFOLIO = 'PORTFOLIO'     # Portfolio has been updated
    CUSTOM = 'CUSTOM'           # Custom event type


class Event:
    """Base class for all events."""
    
    def __init__(self, event_type: EventType, timestamp: Optional[datetime] = None):
        """
        Initialize a new event.
        
        Parameters:
        -----------
        event_type : EventType
            Type of the event.
        timestamp : datetime, optional
            Timestamp when the event was created. If None, current time is used.
        """
        self.event_type = event_type
        self.timestamp = timestamp or datetime.now()
        
    def __lt__(self, other):
        """Compare events by timestamp for sorting in priority queue."""
        if not isinstance(other, Event):
            return NotImplemented
        return self.timestamp < other.timestamp


class MarketEvent(Event):
    """
    Event that occurs when new market data is available.
    Triggers strategy recalculation.
    """
    
    def __init__(self, timestamp: datetime, symbol: str, 
                 data: Union[pd.Series, Dict[str, Any]]):
        """
        Initialize a new market event.
        
        Parameters:
        -----------
        timestamp : datetime
            Timestamp of the market data.
        symbol : str
            Symbol of the instrument.
        data : pd.Series or dict
            Market data (OHLCV or tick).
        """
        super().__init__(EventType.MARKET, timestamp)
        self.symbol = symbol
        self.data = data
    
    def __str__(self):
        """String representation of the event."""
        return f"MarketEvent(timestamp={self.timestamp}, symbol={self.symbol})"


class SignalEvent(Event):
    """
    Event generated when a strategy generates a signal.
    Contains the action to take (buy/sell) and optional metadata.
    """
    
    def __init__(self, timestamp: datetime, symbol: str, 
                 signal_type: str, strength: float = 1.0, 
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new signal event.
        
        Parameters:
        -----------
        timestamp : datetime
            Timestamp when the signal was generated.
        symbol : str
            Symbol of the instrument.
        signal_type : str
            Type of signal ('BUY', 'SELL', etc.).
        strength : float
            Strength of the signal (0.0 to 1.0).
        metadata : dict, optional
            Additional information about the signal.
        """
        super().__init__(EventType.SIGNAL, timestamp)
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength
        self.metadata = metadata or {}
    
    def __str__(self):
        """String representation of the event."""
        return f"SignalEvent(timestamp={self.timestamp}, symbol={self.symbol}, type={self.signal_type})"


class OrderEvent(Event):
    """
    Event that represents an order to be executed.
    Generated in response to SignalEvent.
    """
    
    def __init__(self, timestamp: datetime, symbol: str, order_type: str,
                 quantity: float, direction: str, price: Optional[float] = None,
                 limit_price: Optional[float] = None, stop_price: Optional[float] = None):
        """
        Initialize a new order event.
        
        Parameters:
        -----------
        timestamp : datetime
            Timestamp when the order was created.
        symbol : str
            Symbol of the instrument.
        order_type : str
            Type of order ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT').
        quantity : float
            Quantity to trade.
        direction : str
            Direction of the order ('BUY', 'SELL').
        price : float, optional
            Price for market orders (used for backtesting).
        limit_price : float, optional
            Limit price for limit and stop-limit orders.
        stop_price : float, optional
            Stop price for stop and stop-limit orders.
        """
        super().__init__(EventType.ORDER, timestamp)
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.price = price
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.status = 'CREATED'
        self.filled_quantity = 0.0
        self.filled_price = None
        self.filled_time = None
        self.order_id = None  # Will be set when order is submitted
    
    def __str__(self):
        """String representation of the event."""
        return (f"OrderEvent(timestamp={self.timestamp}, symbol={self.symbol}, "
                f"type={self.order_type}, direction={self.direction}, "
                f"quantity={self.quantity}, status={self.status})")


class FillEvent(Event):
    """
    Event that represents a filled order.
    Generated when an OrderEvent is filled.
    """
    
    def __init__(self, timestamp: datetime, symbol: str, quantity: float,
                 direction: str, fill_price: float, commission: float = 0.0,
                 order_id: Optional[str] = None):
        """
        Initialize a new fill event.
        
        Parameters:
        -----------
        timestamp : datetime
            Timestamp when the order was filled.
        symbol : str
            Symbol of the instrument.
        quantity : float
            Quantity that was traded.
        direction : str
            Direction of the fill ('BUY', 'SELL').
        fill_price : float
            Price at which the order was filled.
        commission : float
            Commission or fees paid.
        order_id : str, optional
            ID of the original order.
        """
        super().__init__(EventType.FILL, timestamp)
        self.symbol = symbol
        self.quantity = quantity
        self.direction = direction
        self.fill_price = fill_price
        self.commission = commission
        self.order_id = order_id
    
    def __str__(self):
        """String representation of the event."""
        return (f"FillEvent(timestamp={self.timestamp}, symbol={self.symbol}, "
                f"direction={self.direction}, quantity={self.quantity}, "
                f"price={self.fill_price})")


class PortfolioEvent(Event):
    """
    Event that signals a portfolio update.
    Generated after a FillEvent is processed.
    """
    
    def __init__(self, timestamp: datetime, cash: float, positions: Dict[str, float],
                 portfolio_value: float, trades: Optional[Dict[str, Any]] = None):
        """
        Initialize a new portfolio event.
        
        Parameters:
        -----------
        timestamp : datetime
            Timestamp of the portfolio update.
        cash : float
            Current available cash.
        positions : dict
            Dictionary of current positions (symbol -> quantity).
        portfolio_value : float
            Total portfolio value.
        trades : dict, optional
            Dictionary of recent trades.
        """
        super().__init__(EventType.PORTFOLIO, timestamp)
        self.cash = cash
        self.positions = positions
        self.portfolio_value = portfolio_value
        self.trades = trades or {}
    
    def __str__(self):
        """String representation of the event."""
        return (f"PortfolioEvent(timestamp={self.timestamp}, "
                f"portfolio_value={self.portfolio_value}, "
                f"positions={len(self.positions)})")


class CustomEvent(Event):
    """
    Custom event for user-defined events.
    """
    
    def __init__(self, timestamp: datetime, name: str, data: Dict[str, Any]):
        """
        Initialize a new custom event.
        
        Parameters:
        -----------
        timestamp : datetime
            Timestamp of the event.
        name : str
            Name of the custom event.
        data : dict
            Dictionary of event data.
        """
        super().__init__(EventType.CUSTOM, timestamp)
        self.name = name
        self.data = data
    
    def __str__(self):
        """String representation of the event."""
        return f"CustomEvent(timestamp={self.timestamp}, name={self.name})"
