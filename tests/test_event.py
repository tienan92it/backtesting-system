import sys
import os
import pytest
from datetime import datetime
import pandas as pd

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting.engine import (
    EventType, Event, MarketEvent, SignalEvent, 
    OrderEvent, FillEvent, PortfolioEvent, CustomEvent,
    EventLoop
)


class TestEventSystem:
    """Test suite for the event system."""
    
    def test_event_base_class(self):
        """Test the base Event class."""
        now = datetime.now()
        event = Event(EventType.MARKET, now)
        
        assert event.event_type == EventType.MARKET
        assert event.timestamp == now
    
    def test_market_event(self):
        """Test MarketEvent creation and properties."""
        now = datetime.now()
        symbol = 'BTCUSDT'
        data = pd.Series({'close': 50000, 'open': 49900, 'high': 50100, 'low': 49800, 'volume': 10})
        
        event = MarketEvent(now, symbol, data)
        
        assert event.event_type == EventType.MARKET
        assert event.timestamp == now
        assert event.symbol == symbol
        assert event.data['close'] == 50000
        assert str(event) == f"MarketEvent(timestamp={now}, symbol={symbol})"
    
    def test_signal_event(self):
        """Test SignalEvent creation and properties."""
        now = datetime.now()
        symbol = 'BTCUSDT'
        signal_type = 'BUY'
        strength = 0.8
        metadata = {'reason': 'Price above MA'}
        
        event = SignalEvent(now, symbol, signal_type, strength, metadata)
        
        assert event.event_type == EventType.SIGNAL
        assert event.timestamp == now
        assert event.symbol == symbol
        assert event.signal_type == signal_type
        assert event.strength == strength
        assert event.metadata == metadata
    
    def test_order_event(self):
        """Test OrderEvent creation and properties."""
        now = datetime.now()
        symbol = 'BTCUSDT'
        order_type = 'MARKET'
        quantity = 1.5
        direction = 'BUY'
        price = 50000
        
        event = OrderEvent(now, symbol, order_type, quantity, direction, price)
        
        assert event.event_type == EventType.ORDER
        assert event.timestamp == now
        assert event.symbol == symbol
        assert event.order_type == order_type
        assert event.quantity == quantity
        assert event.direction == direction
        assert event.price == price
        assert event.status == 'CREATED'
    
    def test_fill_event(self):
        """Test FillEvent creation and properties."""
        now = datetime.now()
        symbol = 'BTCUSDT'
        quantity = 1.5
        direction = 'BUY'
        fill_price = 50050
        commission = 50.05
        order_id = 'order123'
        
        event = FillEvent(now, symbol, quantity, direction, fill_price, commission, order_id)
        
        assert event.event_type == EventType.FILL
        assert event.timestamp == now
        assert event.symbol == symbol
        assert event.quantity == quantity
        assert event.direction == direction
        assert event.fill_price == fill_price
        assert event.commission == commission
        assert event.order_id == order_id
    
    def test_portfolio_event(self):
        """Test PortfolioEvent creation and properties."""
        now = datetime.now()
        cash = 50000
        positions = {'BTCUSDT': 1.5, 'ETHUSDT': 10.0}
        portfolio_value = 100000
        trades = {'latest': {'symbol': 'BTCUSDT', 'quantity': 0.5}}
        
        event = PortfolioEvent(now, cash, positions, portfolio_value, trades)
        
        assert event.event_type == EventType.PORTFOLIO
        assert event.timestamp == now
        assert event.cash == cash
        assert event.positions == positions
        assert event.portfolio_value == portfolio_value
        assert event.trades == trades
    
    def test_custom_event(self):
        """Test CustomEvent creation and properties."""
        now = datetime.now()
        name = 'TEST_EVENT'
        data = {'param1': 'value1', 'param2': 'value2'}
        
        event = CustomEvent(now, name, data)
        
        assert event.event_type == EventType.CUSTOM
        assert event.timestamp == now
        assert event.name == name
        assert event.data == data
    
    def test_event_sorting(self):
        """Test that events are properly sorted by timestamp."""
        event1 = Event(EventType.MARKET, datetime(2023, 1, 1, 12, 0, 0))
        event2 = Event(EventType.MARKET, datetime(2023, 1, 1, 12, 0, 1))
        event3 = Event(EventType.MARKET, datetime(2023, 1, 1, 12, 0, 2))
        
        # Events should be sorted by timestamp
        assert event1 < event2 < event3
        
        # Sorting a list of events
        events = [event3, event1, event2]
        events.sort()
        assert events == [event1, event2, event3]


class TestEventLoop:
    """Test suite for the EventLoop class."""
    
    @pytest.fixture
    def bt_event_loop(self):
        """Create a new EventLoop instance for testing."""
        return EventLoop()
    
    @pytest.fixture
    def sample_events(self):
        """Create a set of sample events for testing."""
        events = [
            MarketEvent(
                datetime(2023, 1, 1, 12, 0, 0),
                'BTCUSDT',
                pd.Series({'close': 50000})
            ),
            SignalEvent(
                datetime(2023, 1, 1, 12, 0, 1),
                'BTCUSDT',
                'BUY'
            ),
            OrderEvent(
                datetime(2023, 1, 1, 12, 0, 2),
                'BTCUSDT',
                'MARKET',
                1.0,
                'BUY',
                50000
            )
        ]
        return events
    
    def test_add_event(self, bt_event_loop, sample_events):
        """Test adding events to the EventLoop."""
        event = sample_events[0]
        bt_event_loop.add_event(event)
        
        assert bt_event_loop.events.qsize() == 1
    
    def test_add_events(self, bt_event_loop, sample_events):
        """Test adding multiple events at once."""
        bt_event_loop.add_events(sample_events)
        
        assert bt_event_loop.events.qsize() == 3
    
    def test_register_handler(self, bt_event_loop):
        """Test registering event handlers."""
        handler_called = {'count': 0}
        
        def test_handler(event):
            handler_called['count'] += 1
        
        bt_event_loop.register_handler(EventType.MARKET, test_handler)
        
        # Add a market event
        event = MarketEvent(
            datetime.now(),
            'BTCUSDT',
            pd.Series({'close': 50000})
        )
        bt_event_loop.add_event(event)
        
        # Process the event
        bt_event_loop.process_next_event()
        
        assert handler_called['count'] == 1
    
    def test_register_handler_all_events(self, bt_event_loop, sample_events):
        """Test registering a handler for all event types."""
        handler_called = {'count': 0}
        
        def test_handler(event):
            handler_called['count'] += 1
        
        bt_event_loop.register_handler_all_events(test_handler)
        bt_event_loop.add_events(sample_events)
        
        # Process all events
        bt_event_loop.run()
        
        assert handler_called['count'] == 3
    
    def test_event_order_processing(self, bt_event_loop):
        """Test that events are processed in timestamp order."""
        processed_events = []
        
        def record_event(event):
            processed_events.append(event)
        
        bt_event_loop.register_handler_all_events(record_event)
        
        # Add events out of order
        event3 = MarketEvent(datetime(2023, 1, 1, 12, 0, 2), 'BTCUSDT', pd.Series({'close': 50000}))
        event1 = MarketEvent(datetime(2023, 1, 1, 12, 0, 0), 'BTCUSDT', pd.Series({'close': 49800}))
        event2 = MarketEvent(datetime(2023, 1, 1, 12, 0, 1), 'BTCUSDT', pd.Series({'close': 49900}))
        
        bt_event_loop.add_events([event3, event1, event2])
        
        # Process events
        bt_event_loop.run()
        
        # Check that events were processed in timestamp order
        assert processed_events[0].timestamp < processed_events[1].timestamp
        assert processed_events[1].timestamp < processed_events[2].timestamp
    
    def test_stop_and_clear(self, bt_event_loop, sample_events):
        """Test stopping the event loop and clearing events."""
        bt_event_loop.add_events(sample_events)
        
        # Clear the queue
        bt_event_loop.clear()
        assert bt_event_loop.events.qsize() == 0
        
        # Add events and start running
        bt_event_loop.add_events(sample_events)
        
        # Set up a handler that stops after first event
        def stop_after_first(event):
            bt_event_loop.stop()
        
        bt_event_loop.register_handler(EventType.MARKET, stop_after_first)
        
        # Run the loop
        bt_event_loop.run()
        
        # It should have stopped after processing one event
        assert bt_event_loop.stats['processed_events'] == 1
        
    def test_handler_exception_handling(self, bt_event_loop):
        """Test that exceptions in handlers are caught and don't crash the loop."""
        def bad_handler(event):
            raise ValueError("Test exception")
        
        bt_event_loop.register_handler(EventType.MARKET, bad_handler)
        
        # Add a market event
        event = MarketEvent(
            datetime.now(),
            'BTCUSDT',
            pd.Series({'close': 50000})
        )
        bt_event_loop.add_event(event)
        
        # This should not raise an exception
        bt_event_loop.process_next_event()
        
        # The event should still be processed
        assert bt_event_loop.stats['processed_events'] == 1 