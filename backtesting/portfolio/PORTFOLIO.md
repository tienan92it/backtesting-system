# Portfolio Module Knowledge Base

## 1. Module Overview
- **Module Name & Version:**  
  `Portfolio Module v1.0`
  
- **Purpose & Scope:**  
  The Portfolio module is responsible for tracking the financial state of the trading system, including positions, cash balances, and equity curves. It manages the translation of trading signals into concrete orders and tracks the outcome of filled orders.

- **Key Responsibilities:**  
  - Track current positions for all symbols
  - Manage cash balances and portfolio value
  - Calculate equity values and returns
  - Record historical positions and trades
  - Generate OrderEvents from SignalEvents
  - Update portfolio state when orders are filled
  - Calculate performance metrics
  
- **Dependencies & Interfaces:**  
  - Receives SignalEvents from the Strategy module
  - Generates OrderEvents for the Execution module
  - Processes FillEvents from the Execution module
  - Generates PortfolioEvents for the Backtester

## 2. Detailed Functionality & Responsibilities

### Portfolio State Management
- **Position Tracking:**
  - Maintains current positions for all symbols
  - Updates positions based on filled orders
  - Tracks the monetary value of each position

- **Cash Management:**
  - Tracks available cash
  - Adjusts cash based on trades, commissions, and slippage
  - Ensures sufficient cash for new orders

- **Equity Calculation:**
  - Calculates total portfolio value (cash + position values)
  - Maintains an equity curve over time
  - Calculates returns and drawdowns

### Order Generation
- **Signal Processing:**
  - Receives trading signals from strategies
  - Determines appropriate order size based on available cash or existing positions
  - Creates orders with appropriate parameters (price, quantity, order type)

- **Risk Management:**
  - Applies position sizing rules
  - Enforces risk limits on trades
  - Prevents over-leveraging the portfolio

### Performance Tracking
- **Trade Records:**
  - Tracks individual trades with entry/exit prices, P&L, and timestamps
  - Calculates trade statistics (win rate, average P&L, etc.)
  - Provides trade history for analysis

- **Portfolio History:**
  - Records the evolution of portfolio state over time
  - Tracks equity curve and drawdowns
  - Provides data for performance visualization

## 3. API / Interface Description

### Portfolio Class
```python
class Portfolio:
    def __init__(self, initial_capital: float = 100000.0, symbols: Optional[List[str]] = None):
        # Initialize the portfolio with initial capital and symbols
    
    def initialize(self, initial_capital: float = None, data_handler = None) -> None:
        # Initialize the portfolio with initial capital and data handler
    
    def update_fill(self, fill_event: FillEvent) -> None:
        # Updates the portfolio based on a fill event
    
    def update_market(self, timestamp: datetime, market_data: Dict[str, Dict[str, float]]) -> None:
        # Updates the portfolio based on new market data
    
    def generate_signals(self, timestamp: datetime, signals: Dict[str, Any]) -> List[OrderEvent]:
        # Generates orders based on trading signals
    
    def on_signal(self, signal_event) -> Optional[OrderEvent]:
        # Processes a signal event and generates an order event
    
    def get_equity_curve(self) -> pd.DataFrame:
        # Returns the equity curve as a DataFrame
    
    def get_position_history(self) -> pd.DataFrame:
        # Returns the position history as a DataFrame
    
    def get_trades(self) -> pd.DataFrame:
        # Returns the trade history as a DataFrame
    
    def get_current_positions(self) -> Dict[str, float]:
        # Returns the current positions
    
    def get_portfolio_value(self) -> float:
        # Returns the current portfolio value
    
    def get_current_capital(self) -> float:
        # Returns the current available capital
    
    def get_returns(self) -> List[float]:
        # Returns the list of returns
    
    def get_statistics(self) -> Dict[str, Any]:
        # Returns various portfolio statistics
    
    def on_fill(self, fill_event: FillEvent) -> Optional[PortfolioEvent]:
        # Processes a fill event and generates a portfolio event
```

### Data Structures
- **Position:**
  ```python
  positions = {
      'BTC/USD': 1.5,  # Holding 1.5 BTC
      'ETH/USD': 10.0  # Holding 10 ETH
  }
  ```

- **Portfolio Value:**
  ```python
  portfolio_value = cash + sum(position_quantity * current_price for each symbol)
  ```

- **Trade Record:**
  ```python
  trade = {
      'symbol': 'BTC/USD',
      'entry_time': datetime(2022, 1, 1, 10, 0, 0),
      'entry_price': 40000.0,
      'entry_type': 'BUY',
      'exit_time': datetime(2022, 1, 5, 14, 0, 0),
      'exit_price': 42000.0,
      'exit_type': 'SELL',
      'quantity': 0.5,
      'profit_loss': 1000.0,
      'commission': 40.0,
      'pct_return': 5.0
  }
  ```

## 4. Usage Examples

### Basic Portfolio Initialization
```python
from backtesting.portfolio.portfolio import Portfolio

# Create a portfolio with initial capital and symbols
portfolio = Portfolio(initial_capital=10000.0, symbols=['BTC/USD', 'ETH/USD'])

# Link the portfolio to a data handler
portfolio.initialize(data_handler=data_handler)
```

### Processing Fill Events
```python
from backtesting.engine.event import FillEvent
from datetime import datetime

# Create a fill event representing a buy of BTC
fill_event = FillEvent(
    timestamp=datetime.now(),
    symbol='BTC/USD',
    quantity=0.5,
    direction='BUY',
    fill_price=40000.0,
    commission=20.0
)

# Update the portfolio with the fill
portfolio.update_fill(fill_event)

# Check the updated portfolio state
print(f"Current positions: {portfolio.get_current_positions()}")
print(f"Cash remaining: ${portfolio.get_current_capital():,.2f}")
print(f"Portfolio value: ${portfolio.get_portfolio_value():,.2f}")
```

### Generating Orders from Signals
```python
from backtesting.engine.event import SignalEvent

# Create a signal to buy BTC
signal_event = SignalEvent(
    timestamp=datetime.now(),
    symbol='BTC/USD',
    signal_type='BUY',
    strength=1.0,
    metadata={'price': 40000.0}
)

# Process the signal and get an order (if generated)
order_event = portfolio.on_signal(signal_event)

if order_event:
    print(f"Generated order: {order_event}")
else:
    print("No order generated from signal")
```

## 5. Configuration & Environment Setup
- **Required Python Version:** 3.7+
- **Dependencies:**
  - pandas: For data handling and calculations
  - datetime: For timestamp management
  - logging: For diagnostic output

## 6. Testing & Validation
- **Position Tracking Accuracy:**
  Verify that positions are correctly updated after trades.

- **Cash Management Testing:**
  Ensure that cash balance is properly adjusted for trades, commissions, and slippage.

- **Performance Calculation Testing:**
  Validate that equity, returns, and other performance metrics are accurately calculated.

## 7. LLM-Specific Instructions
- **Code Regeneration Hints:**
  - Maintain a clear separation between position tracking and order generation logic
  - Preserve the exact order of operations when updating portfolio state
  - Ensure proper accounting for commissions and slippage
  - Pay attention to the sign of quantities in buy vs. sell operations

- **Contextual Guidance:**
  - The portfolio module is the financial "brain" of the system
  - It translates abstract signals into concrete orders based on available capital
  - Accuracy in position and cash tracking is critical

- **Examples of Extension:**
  - Adding support for different position sizing algorithms
  - Implementing more sophisticated risk management rules
  - Adding support for margin trading and leverage
  - Implementing portfolio-level constraints and optimization

## 8. Versioning & Change Log
- **Version 1.0:**
  - Initial implementation with basic position tracking
  - Support for cash management and equity calculations
  - Simple position sizing based on signal strength

- **Future Roadmap:**
  - Add support for different base currencies
  - Implement more sophisticated position sizing algorithms
  - Add risk management constraints
  - Support for portfolio rebalancing 