# Strategy Module Knowledge Base

## 1. Module Overview
- **Module Name & Version:**  
  `Strategy Module v1.0`
  
- **Purpose & Scope:**  
  The Strategy module is responsible for implementing trading strategies that analyze market data and generate trading signals. It provides a framework for creating custom strategies with consistent interfaces to the backtesting system.

- **Key Responsibilities:**  
  - Process market data to identify trading opportunities
  - Generate buy/sell signals based on technical indicators, patterns, or models
  - Manage strategy-specific parameters and state
  - Interface with the portfolio and execution components
  - Define entry and exit conditions for trades
  
- **Dependencies & Interfaces:**  
  - Dependencies: pandas, numpy
  - Receives MarketEvents from the Engine module
  - Generates SignalEvents to be processed by the Portfolio module
  - Interfaces with DataHandler for market data access

## 2. Detailed Functionality & Responsibilities

### Strategy Framework
- **Base Strategy Class:**
  - Abstract base class that defines the interface for all strategies
  - Provides common functionality for data access and signal generation
  - Implements lifecycle methods: init, next, on_start, on_finish, on_trade

- **Strategy Lifecycle:**
  1. Initialization (init): Set up indicators, models, and parameters
  2. Data processing (next): Analyze new data and generate signals
  3. Event handling: Respond to market events and portfolio updates
  4. Trade execution: Submit buy/sell orders with appropriate parameters

- **Data Access:**
  - Strategies have access to historical and current market data
  - Indicators can be calculated from raw OHLCV data
  - Current position and portfolio information is available

- **Signal Generation:**
  - Strategies trigger buy/sell signals based on market conditions
  - Signals can specify quantity, price, order type, and execution details
  - Multiple signal types are supported: market, limit, stop, stop-limit

### Example Strategies
- **Moving Average Cross:**
  - Classic trading strategy using two moving averages
  - Buys when short MA crosses above long MA (golden cross)
  - Sells when short MA crosses below long MA (death cross)

- **Machine Learning Strategy:**
  - Uses trained ML models to predict price movements
  - Features can include technical indicators, price patterns, and external data
  - Makes trading decisions based on prediction confidence

## 3. API / Interface Description

### Strategy Abstract Base Class
```python
class Strategy(ABC):
    def __init__(self):
        # Initialize strategy state and parameters
    
    def initialize(self, data_handler=None, portfolio=None, event_loop=None):
        # Connect strategy to backtesting components
    
    def set_data(self, data: pd.DataFrame) -> None:
        # Set the market data for the strategy
    
    def set_backtester_functions(self, buy_func: Callable, sell_func: Callable) -> None:
        # Set callbacks for order execution
    
    def update_portfolio(self, position: float, position_size: float, 
                        cash: float, portfolio_value: float) -> None:
        # Update the strategy's knowledge of portfolio state
    
    def buy(self, size: Optional[float] = None, 
           price: Optional[float] = None, 
           limit_price: Optional[float] = None,
           stop_price: Optional[float] = None,
           percent: Optional[float] = None) -> None:
        # Place a buy order
    
    def sell(self, size: Optional[float] = None, 
            price: Optional[float] = None, 
            limit_price: Optional[float] = None,
            stop_price: Optional[float] = None,
            percent: Optional[float] = None) -> None:
        # Place a sell order
    
    @abstractmethod
    def init(self) -> None:
        # Initialize strategy-specific items (indicators, etc.)
    
    @abstractmethod
    def next(self) -> None:
        # Process the current bar and generate signals
    
    def on_start(self) -> None:
        # Called when the backtest starts
    
    def on_finish(self) -> None:
        # Called when the backtest finishes
    
    def on_trade(self, trade: Dict[str, Any]) -> None:
        # Called after a trade is executed
    
    def get_current_price(self) -> float:
        # Helper to get current price
    
    def get_current_bar(self) -> pd.Series:
        # Helper to get current bar data
    
    def crossover(self, series1, series2) -> bool:
        # Helper to detect when series1 crosses above series2
    
    def crossunder(self, series1, series2) -> bool:
        # Helper to detect when series1 crosses below series2
```

### Strategy Example Implementation
```python
class MovingAverageCrossStrategy(Strategy):
    def __init__(self, short_window: int = 50, long_window: int = 200):
        super().__init__()
        self.params = {
            'short_window': short_window,
            'long_window': long_window
        }
    
    def init(self) -> None:
        # Calculate moving averages
        short_window = self.params['short_window']
        long_window = self.params['long_window']
        self.ma_short = self.data['close'].rolling(window=short_window).mean()
        self.ma_long = self.data['close'].rolling(window=long_window).mean()
    
    def next(self) -> None:
        # Skip if not enough data
        if self.current_index < self.params['long_window']:
            return
        
        # Check for golden cross (buy signal)
        if self.crossover(self.ma_short, self.ma_long):
            if self.position <= 0:
                self.buy()
        
        # Check for death cross (sell signal)
        elif self.crossunder(self.ma_short, self.ma_long):
            if self.position > 0:
                self.sell()
```

## 4. Usage Examples

### Creating a Simple Strategy
```python
from backtesting.strategy.base import Strategy

class SimpleBreakoutStrategy(Strategy):
    def __init__(self, window: int = 20):
        super().__init__()
        self.params = {'window': window}
        self.highest = None
        self.lowest = None
    
    def init(self) -> None:
        # Calculate rolling high and low
        window = self.params['window']
        self.highest = self.data['high'].rolling(window=window).max()
        self.lowest = self.data['low'].rolling(window=window).min()
    
    def next(self) -> None:
        # Skip if not enough data
        if self.current_index < self.params['window']:
            return
        
        current_price = self.get_current_price()
        
        # Buy if price breaks above highest high
        if current_price > self.highest[self.current_index - 1]:
            if self.position <= 0:
                self.buy()
        
        # Sell if price breaks below lowest low
        elif current_price < self.lowest[self.current_index - 1]:
            if self.position > 0:
                self.sell()
```

### Using a Strategy in a Backtest
```python
from backtesting.strategy.examples.moving_average_cross import MovingAverageCrossStrategy
from backtesting.engine.backtester import Backtester
from backtesting.data.binance_data import BinanceDataHandler

# Create data handler
data_handler = BinanceDataHandler(use_cache=True)

# Create strategy with custom parameters
strategy = MovingAverageCrossStrategy(short_window=20, long_window=50)

# Create and run backtester
backtester = Backtester(
    data_handler=data_handler,
    strategy=strategy,
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
  - pandas: For data manipulation and indicators
  - numpy: For numerical operations
  - Optional dependencies for specific strategies:
    - scikit-learn for ML strategies
    - ta-lib for technical indicators
    - tensorflow/pytorch for deep learning strategies

## 6. Testing & Validation
- **Strategy Logic Testing:**
  Test that strategies generate expected signals for known market conditions.

- **Parameter Sensitivity Testing:**
  Verify how strategy performs with different parameters.

- **Edge Case Testing:**
  Ensure strategies handle market gaps, low liquidity, and other edge cases.

## 7. LLM-Specific Instructions
- **Code Regeneration Hints:**
  - Follow the abstract base class interface when implementing new strategies
  - Maintain a clear separation between strategy logic and backtesting mechanics
  - Use vectorized operations where possible for performance
  - Always implement required lifecycle methods: init() and next()

- **Contextual Guidance:**
  - Strategies should focus on signal generation, not order execution details
  - The strategy module handles the "brain" of the trading system
  - Strategies should be parameter-driven for optimization

- **Examples of Extension:**
  - Adding more sophisticated technical indicator strategies
  - Implementing machine learning or statistical arbitrage strategies
  - Creating multi-asset allocation strategies
  - Developing sentiment-based or news-driven strategies

## 8. Versioning & Change Log
- **Version 1.0:**
  - Initial implementation with Strategy base class
  - Example strategies: MovingAverageCross, RSI
  - Basic technical indicator support

- **Future Roadmap:**
  - Add support for multi-timeframe analysis
  - Add portfolio-level strategies (vs. single-instrument)
  - Implement optimization framework for strategy parameters
  - Add support for strategy composability (combining strategies) 