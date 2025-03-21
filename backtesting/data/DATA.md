# Data Module Knowledge Base

## 1. Module Overview
- **Module Name & Version:**  
  `Data Module v1.0`
  
- **Purpose & Scope:**  
  The Data module is responsible for retrieving, processing, and providing market data to the backtesting system. It acts as the source of price information that drives the entire backtesting process through market events.

- **Key Responsibilities:**  
  - Retrieve historical OHLCV (Open, High, Low, Close, Volume) data from various sources
  - Clean and prepare data for backtesting
  - Provide a consistent interface for accessing market data
  - Generate market events as part of the event-driven architecture
  - Support data caching for improved performance
  
- **Dependencies & Interfaces:**  
  - External dependencies: pandas, numpy, requests, tqdm
  - Interfaces with the Engine module to generate MarketEvents

## 2. Detailed Functionality & Responsibilities

### Data Retrieval
- **Data Sources:**
  - Binance exchange API (BinanceDataHandler)
  - CCXT library for multiple exchanges (CCXTDataHandler)
  - Local file caching for repeated runs

- **Data Types:**
  - OHLCV candlestick data at various timeframes
  - Multiple symbols/trading pairs
  - Additional metadata like trading volume

- **Caching Mechanism:**
  - Save data to disk for faster reuse
  - Optimization for repeated backtests
  - Configurable cache directory

### Data Processing
- **Data Cleaning:**
  - Handle missing values
  - Convert string values to appropriate numeric types
  - Ensure chronological ordering
  - Remove or fill outliers

- **Transformation:**
  - Convert timestamps to datetime objects
  - Normalize data formats from different sources
  - Apply any pre-processing required by strategies

### Backtesting Interface
- **Bar Iteration:**
  - Maintain current bar position
  - Update to next bar on demand
  - Keep track of available symbols and timeframes

- **Event Generation:**
  - Create MarketEvents from bar data
  - Ensure events are properly timestamped
  - Propagate events to the EventLoop

## 3. API / Interface Description

### DataHandler Abstract Base Class
```python
class DataHandler(ABC):
    @abstractmethod
    def initialize(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> None:
        # Initialize data within date range
        
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        # Fetch historical OHLCV data
        
    @abstractmethod
    def get_symbols(self) -> list:
        # Get list of available trading symbols
        
    @abstractmethod
    def get_timeframes(self) -> list:
        # Get list of supported timeframes
        
    @abstractmethod
    def update_bars(self) -> bool:
        # Update current bar data, return False when no more data
        
    @abstractmethod
    def get_all_bars(self) -> Dict[datetime, Dict[str, Dict[str, float]]]:
        # Get all bars for all symbols by timestamp
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clean and prepare data for backtesting
        
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str, directory: str) -> str:
        # Save data to disk for future use
        
    def load_data(self, symbol: str, timeframe: str, start_time, end_time, directory: str) -> pd.DataFrame:
        # Load data from disk if available
```

### BinanceDataHandler Implementation
```python
class BinanceDataHandler(DataHandler):
    BASE_URL = "https://api.binance.com/api/v3"
    MAX_LIMIT = 1000  # Maximum number of candles per request
    
    # Mapping of timeframe strings to milliseconds
    TIMEFRAME_MAPPING = {
        "1m": 60 * 1000,
        "1h": 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        # ...
    }
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 use_cache: bool = True, cache_dir: str = 'data'):
        # Initialize with optional API credentials and caching settings
```

### CCXTDataHandler Implementation
```python
class CCXTDataHandler(DataHandler):
    def __init__(self, exchange_id: str, api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None, password: Optional[str] = None,
                 use_cache: bool = True, cache_dir: str = 'data'):
        # Initialize with CCXT exchange and credentials
```

## 4. Usage Examples

### Basic Data Retrieval
```python
from backtesting.data.binance_data import BinanceDataHandler

# Create data handler
data_handler = BinanceDataHandler(use_cache=True)

# Fetch historical data
df = data_handler.get_historical_data(
    symbol='BTCUSDT',
    timeframe='1h',
    start_time='2022-01-01',
    end_time='2022-12-31'
)

print(f"Retrieved {len(df)} bars of data")
print(df.head())
```

### Integration with Backtesting System
```python
from backtesting.data.binance_data import BinanceDataHandler
from backtesting.engine.event_loop import EventLoop

# Create components
data_handler = BinanceDataHandler()
event_loop = EventLoop()

# Initialize with date range
data_handler.initialize(start_date='2022-01-01', end_date='2022-12-31')

# Create market event handler
def handle_market_event(event):
    print(f"Processing market data: {event.symbol} at {event.timestamp}")

# Register handler
event_loop.register_handler(EventType.MARKET, handle_market_event)

# Simulate the event loop
while data_handler.update_bars():
    # This would create market events in a real backtest
    current_data = data_handler.current_bar
    # Process the current bar...
```

## 5. Configuration & Environment Setup
- **Required Python Version:** 3.7+
- **Dependencies:**
  - pandas: For data manipulation
  - numpy: For numerical operations
  - requests: For API calls
  - tqdm: For progress bars
  - ccxt (optional): For multi-exchange support
- **Configuration Options:**
  - API credentials for exchange access
  - Cache directory for data storage
  - Start and end dates for backtesting period

## 6. Testing & Validation
- **Data Integrity Testing:**
  Test that retrieved data is complete, properly formatted, and chronologically ordered.

- **API Connection Testing:**
  Verify that connections to exchange APIs function correctly with proper error handling.

- **Caching Testing:**
  Ensure data is correctly saved and loaded from the cache.

## 7. LLM-Specific Instructions
- **Code Regeneration Hints:**
  - Follow the abstract base class interface when implementing new data handlers
  - Maintain error handling for API calls
  - Implement caching for performance optimization
  - Ensure proper data cleaning and transformation

- **Contextual Guidance:**
  - The data module is the starting point of the event-driven backtesting system
  - All other components depend on accurate and reliable market data
  - Performance optimization is critical for handling large datasets

- **Examples of Extension:**
  - Adding support for new exchanges through additional handler implementations
  - Implementing real-time data feeds for live trading
  - Adding support for alternative data sources (sentiment, news, etc.)
  - Implementing more sophisticated data cleaning and normalization

## 8. Versioning & Change Log
- **Version 1.0:**
  - Initial implementation with Binance and CCXT data handlers
  - Basic data caching functionality
  - OHLCV data support

- **Future Roadmap:**
  - Add support for order book data (L2/L3)
  - Implement streaming data for live trading
  - Optimize data storage for large datasets
  - Add support for tick-level data 