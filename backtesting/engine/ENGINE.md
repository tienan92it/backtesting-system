# Engine Module Knowledge Base

## 1. Module Overview
- **Module Name & Version:**  
  `Engine Module v1.0`
  
- **Purpose & Scope:**  
  The Engine module is the core of the backtesting system, managing the event-driven architecture and flow of information between different components. It defines the event types, event loop, and backtester components that coordinate the entire simulation process.

- **Key Responsibilities:**  
  - Define event types and structures for system communication
  - Implement the event loop for managing event flow
  - Orchestrate backtesting through the backtester class
  - Maintain chronological ordering of events
  - Provide event handlers for various components
  
- **Dependencies & Interfaces:**  
  - Dependencies: Python standard libraries (queue, datetime, enum), pandas
  - Interfaces with all other modules through event registration and dispatch

## 2. Detailed Functionality & Responsibilities

### Event System
- **Event Class Hierarchy:**
  - Base `Event` class with timestamp and event type
  - Specialized event types: MarketEvent, SignalEvent, OrderEvent, FillEvent, PortfolioEvent, CustomEvent
  - Events are comparable by timestamp for priority queue ordering

- **Event Flow:**
  1. Data handlers generate MarketEvents from price data
  2. Strategies process MarketEvents and generate SignalEvents
  3. Portfolio processes SignalEvents and generates OrderEvents
  4. Execution handler processes OrderEvents and generates FillEvents
  5. Portfolio processes FillEvents and generates PortfolioEvents
  6. Backtester updates performance metrics based on PortfolioEvents

- **Event Loop:**
  - Priority queue for chronological event processing
  - Handler registration for different event types
  - Statistics tracking for processed events
  - Control flow for running, stopping, and clearing the simulation

- **Backtester:**
  - Central coordinator that connects all components
  - Initializes and configures components
  - Runs the event loop
  - Calculates and reports performance metrics
  - Manages simulation state and parameters

## 3. API / Interface Description

### Event Types
```python
class EventType(Enum):
    MARKET = 'MARKET'           # New market data (tick/bar)
    SIGNAL = 'SIGNAL'           # Strategy signal (buy/sell)
    ORDER = 'ORDER'             # Order to be executed
    FILL = 'FILL'               # Order has been filled
    PORTFOLIO = 'PORTFOLIO'     # Portfolio has been updated
    CUSTOM = 'CUSTOM'           # Custom event type
```

### Base Event Class
```python
class Event:
    def __init__(self, event_type: EventType, timestamp: Optional[datetime] = None):
        # Initializes event with type and timestamp
    
    def __lt__(self, other):
        # Compares events by timestamp for priority queue
```

### Event-Specific Classes
- **MarketEvent**: Represents new market data
- **SignalEvent**: Strategy-generated trading signals
- **OrderEvent**: Orders to be executed
- **FillEvent**: Executed orders with fill information
- **PortfolioEvent**: Updated portfolio states
- **CustomEvent**: User-defined events

### EventLoop Class
```python
class EventLoop:
    def __init__(self):
        # Initialize event queue and handlers
    
    def register_handler(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        # Register handler for specific event type
    
    def register_handler_all_events(self, handler: Callable[[Event], None]) -> None:
        # Register handler for all event types
    
    def add_event(self, event: Event) -> None:
        # Add event to queue
    
    def add_events(self, events: List[Event]) -> None:
        # Add multiple events to queue
    
    def process_next_event(self) -> Optional[Event]:
        # Process next event in queue
    
    def run(self, max_events: Optional[int] = None) -> None:
        # Run event loop until queue is empty or max_events is reached
    
    def stop(self) -> None:
        # Stop event loop
    
    def clear(self) -> None:
        # Clear all events from queue
    
    def get_stats(self) -> Dict[str, Any]:
        # Get statistics about event processing
```

## 4. Usage Examples

### Creating and Processing Events
```python
from backtesting.engine.event import EventType, MarketEvent
from backtesting.engine.event_loop import EventLoop
from datetime import datetime
import pandas as pd

# Create an event loop
event_loop = EventLoop()

# Define a handler for market events
def handle_market_event(event):
    if event.event_type == EventType.MARKET:
        print(f"Processing market data for {event.symbol} at {event.timestamp}")

# Register the handler
event_loop.register_handler(EventType.MARKET, handle_market_event)

# Create a market event
data = pd.Series({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.5, 'volume': 1000})
market_event = MarketEvent(datetime.now(), 'BTC/USD', data)

# Add event to the loop
event_loop.add_event(market_event)

# Run the event loop
event_loop.run()
```

### Complete Backtest Setup
```python
from backtesting.engine.backtester import Backtester
from backtesting.data.data_handler import DataHandler
from backtesting.strategy.base import Strategy
from backtesting.portfolio.portfolio import Portfolio
from backtesting.execution.simulated_execution import SimulatedExecutionHandler

# Create components
data_handler = DataHandler(...)
strategy = Strategy(...)
portfolio = Portfolio(...)
execution = SimulatedExecutionHandler(...)

# Create and run backtester
backtester = Backtester(
    data_handler=data_handler,
    strategy=strategy,
    portfolio=portfolio,
    execution_handler=execution,
    initial_capital=10000.0,
    start_date='2022-01-01',
    end_date='2022-12-31'
)

# Run the backtest
results = backtester.run()
```

## 5. Configuration & Environment Setup
- **Required Python Version:** 3.7+
- **Dependencies:**
  - Python standard libraries (queue, enum, datetime)
  - pandas for data handling
  - logging for debugging

## 6. Testing & Validation
- **Event Flow Testing:**
  Test that events flow correctly through the system and are processed in the expected order.

- **Component Interaction Testing:**
  Verify that all components interact correctly through the event loop.

- **Event Ordering Testing:**
  Ensure events are processed in chronological order by timestamp.

## 7. LLM-Specific Instructions
- **Code Regeneration Hints:**
  - Follow PEP 8 style guidelines
  - Maintain type annotations for all methods
  - Use consistent logging throughout
  - Preserve the event-driven architecture pattern

- **Contextual Guidance:**
  - The event system is the backbone of the entire backtesting framework
  - All components communicate exclusively through events
  - Event ordering is critical for accurate simulation results

- **Examples of Extension:**
  - Adding new event types for specific use cases
  - Implementing parallel event processing for performance
  - Adding event persistence for resumable backtests

## 8. Versioning & Change Log
- **Version 1.0:**
  - Initial implementation of event system and event loop
  - Basic backtester implementation

- **Future Roadmap:**
  - Optimize event processing for large datasets
  - Add support for live trading through the same interface
  - Implement event persistence and serialization 