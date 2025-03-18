"""
Example demonstrating the event system for the backtesting engine.

This example shows how to create different types of events, register handlers,
and process events through the event loop.
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting.engine import (
    EventType, Event, MarketEvent, SignalEvent, 
    OrderEvent, FillEvent, PortfolioEvent, CustomEvent,
    EventLoop
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('event_example')


def create_sample_events():
    """Create a sample sequence of events for demonstration."""
    
    # Create a series of timestamps
    now = datetime.now()
    timestamps = [now + timedelta(seconds=i) for i in range(10)]
    
    # Create sample events
    events = []
    
    # 1. Market data events
    symbol = 'BTCUSDT'
    for i, ts in enumerate(timestamps[:5]):
        price = 50000 + i * 100
        data = pd.Series({
            'open': price - 50,
            'high': price + 100,
            'low': price - 100,
            'close': price,
            'volume': 10.5 + i
        })
        events.append(MarketEvent(ts, symbol, data))
    
    # 2. Signal events
    events.append(SignalEvent(
        timestamps[5], 
        symbol, 
        signal_type='BUY', 
        strength=0.8,
        metadata={'reason': 'Price above moving average'}
    ))
    
    # 3. Order events
    events.append(OrderEvent(
        timestamps[6],
        symbol,
        order_type='MARKET',
        quantity=1.0,
        direction='BUY',
        price=50300
    ))
    
    # 4. Fill events
    events.append(FillEvent(
        timestamps[7],
        symbol,
        quantity=1.0,
        direction='BUY',
        fill_price=50305,
        commission=5.03,
        order_id='order123'
    ))
    
    # 5. Portfolio events
    events.append(PortfolioEvent(
        timestamps[8],
        cash=49690.0,
        positions={symbol: 1.0},
        portfolio_value=50000.0,
        trades={'latest': {
            'symbol': symbol,
            'direction': 'BUY',
            'quantity': 1.0,
            'price': 50305,
            'timestamp': timestamps[7]
        }}
    ))
    
    # 6. Custom events
    events.append(CustomEvent(
        timestamps[9],
        name='STRATEGY_UPDATE',
        data={'strategy': 'MovingAverageCross', 'updated_params': {'short_window': 10}}
    ))
    
    return events


def market_data_handler(event):
    """Handler for market data events."""
    assert isinstance(event, MarketEvent)
    logger.info(f"Market data received for {event.symbol}: Close = {event.data['close']:.2f}")


def signal_handler(event):
    """Handler for signal events."""
    assert isinstance(event, SignalEvent)
    logger.info(f"Signal generated: {event.signal_type} {event.symbol} (strength: {event.strength:.2f})")
    logger.info(f"Signal reason: {event.metadata.get('reason', 'Unknown')}")


def order_handler(event):
    """Handler for order events."""
    assert isinstance(event, OrderEvent)
    logger.info(f"Order placed: {event.direction} {event.quantity} {event.symbol} at {event.price}")


def fill_handler(event):
    """Handler for fill events."""
    assert isinstance(event, FillEvent)
    logger.info(f"Order filled: {event.direction} {event.quantity} {event.symbol} at {event.fill_price}")
    logger.info(f"Commission paid: {event.commission:.2f}")


def portfolio_handler(event):
    """Handler for portfolio events."""
    assert isinstance(event, PortfolioEvent)
    logger.info(f"Portfolio updated: Cash={event.cash:.2f}, Value={event.portfolio_value:.2f}")
    positions_str = ", ".join([f"{symbol}: {qty}" for symbol, qty in event.positions.items()])
    logger.info(f"Current positions: {positions_str}")


def custom_handler(event):
    """Handler for custom events."""
    assert isinstance(event, CustomEvent)
    logger.info(f"Custom event '{event.name}' received with data: {event.data}")


def generic_handler(event):
    """Generic handler called for all events."""
    logger.debug(f"Generic handler called for {event.event_type} event at {event.timestamp}")


def run_example():
    """Run the event system example."""
    # Create the event loop
    event_loop = EventLoop()
    
    # Register handlers
    event_loop.register_handler(EventType.MARKET, market_data_handler)
    event_loop.register_handler(EventType.SIGNAL, signal_handler)
    event_loop.register_handler(EventType.ORDER, order_handler)
    event_loop.register_handler(EventType.FILL, fill_handler)
    event_loop.register_handler(EventType.PORTFOLIO, portfolio_handler)
    event_loop.register_handler(EventType.CUSTOM, custom_handler)
    
    # Register a generic handler for all events
    event_loop.register_handler_all_events(generic_handler)
    
    # Create sample events
    events = create_sample_events()
    
    # Add events to the loop
    event_loop.add_events(events)
    
    # Process events
    logger.info(f"Starting event loop with {len(events)} events")
    event_loop.run()
    
    # Print statistics
    stats = event_loop.get_stats()
    logger.info("Event loop statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    run_example() 