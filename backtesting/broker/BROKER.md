# Broker Module Knowledge Base

## 1. Module Overview
- **Module Name & Version:**  
  `Broker Module v1.0`
  
- **Purpose & Scope:**  
  The Broker module represents the trading venue where orders are executed. It simulates a brokerage platform with order execution capabilities, account information tracking, and position management. While similar to the Execution module, the Broker module provides a higher-level interface that encompasses both execution and account management.

- **Key Responsibilities:**  
  - Execute trading orders with realistic conditions
  - Track account balances and positions
  - Apply commission and slippage models
  - Provide market data access
  - Maintain fill history and trade records
  
- **Dependencies & Interfaces:**  
  - Uses DataHandler for market data access
  - Processes OrderEvents from the Portfolio module
  - Generates FillEvents for the Portfolio module
  - Maintains independent account state from the Portfolio module

## 2. Detailed Functionality & Responsibilities

### Broker Simulation
- **Order Execution:**
  - Execute market, limit, and stop orders
  - Apply realistic slippage models to execution prices
  - Calculate and apply trading commissions
  - Generate FillEvents for executed orders

- **Account Management:**
  - Track cash balances in various currencies
  - Manage open positions across multiple symbols
  - Update account state after trades
  - Provide account information (balance, equity, margin)

- **Position Tracking:**
  - Maintain current positions for all symbols
  - Update positions based on executed orders
  - Calculate position value based on current market prices
  - Provide position information for risk management

### Market Data Interface
- **Price Data:**
  - Provide access to current market prices
  - Support historical data retrieval
  - Offer different timeframes for data access
  - Maintain synchronization with the backtesting timeframe

## 3. API / Interface Description

### Broker Abstract Base Class
```python
class Broker(ABC):
    @abstractmethod
    def execute_order(self, order_event: OrderEvent) -> Optional[FillEvent]:
        # Execute an order and generate a fill event
        
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        # Get account information such as balance, margin, etc.
        
    @abstractmethod
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        # Get current positions
        
    @abstractmethod
    def get_market_data(self, symbol: str, timeframe: str, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> Dict[str, Any]:
        # Get market data for a symbol
```

### SimulatedBroker Implementation
```python
class SimulatedBroker(Broker):
    def __init__(self, data_handler, commission_model='percentage', 
                 commission_rate=0.001, slippage_model='fixed',
                 slippage_amount=0.0):
        # Initialize the simulated broker
        
    def execute_order(self, order_event: OrderEvent) -> Optional[FillEvent]:
        # Execute an order and generate a fill event
        
    def _calculate_execution_price(self, symbol: str, direction: str, 
                                  order_type: str, limit_price: Optional[float] = None) -> float:
        # Calculate the execution price with slippage
        
    def _calculate_commission(self, price: float, quantity: float) -> float:
        # Calculate commission for a trade
        
    def _update_positions(self, fill_event: FillEvent) -> None:
        # Update positions after a fill
        
    def get_account_info(self) -> Dict[str, Any]:
        # Get account information
        
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        # Get current positions
        
    def get_market_data(self, symbol: str, timeframe: str, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None) -> Dict[str, Any]:
        # Get market data for a symbol
        
    def set_initial_balance(self, amount: float, currency: str = "cash") -> None:
        # Set initial balance for a currency
        
    def get_fill_history(self) -> List[FillEvent]:
        # Get history of all fills
```

## 4. Usage Examples

### Creating and Configuring a Broker
```python
from backtesting.broker.simulated_broker import SimulatedBroker
from backtesting.data.binance_data import BinanceDataHandler

# Create data handler
data_handler = BinanceDataHandler()

# Create broker with custom settings
broker = SimulatedBroker(
    data_handler=data_handler,
    commission_model='percentage',
    commission_rate=0.001,  # 0.1% commission
    slippage_model='fixed',
    slippage_amount=0.0005  # 0.05% slippage
)

# Set initial account balance
broker.set_initial_balance(10000.0, currency="USDT")
```

### Executing Orders
```python
from backtesting.engine.event import OrderEvent
from datetime import datetime

# Create a market buy order
order = OrderEvent(
    timestamp=datetime.now(),
    symbol='BTC/USDT',
    order_type='MARKET',
    quantity=0.1,
    direction='BUY'
)

# Execute the order
fill = broker.execute_order(order)

if fill:
    print(f"Order filled at ${fill.fill_price} with ${fill.commission} commission")
    
    # Check updated account state
    account = broker.get_account_info()
    positions = broker.get_positions()
    
    print(f"Cash balance: ${account['cash']:,.2f}")
    print(f"Current positions: {positions}")
```

### Retrieving Market Data
```python
# Get current market data for a symbol
btc_data = broker.get_market_data(
    symbol='BTC/USDT',
    timeframe='1h',
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 1, 31)
)

# Use the data for analysis
print(f"BTC/USDT price range: ${btc_data['low']:,.2f} - ${btc_data['high']:,.2f}")
```

## 5. Configuration & Environment Setup
- **Required Python Version:** 3.7+
- **Dependencies:**
  - pandas: For data handling
  - numpy: For numerical operations
  - logging: For diagnostic output

## 6. Testing & Validation
- **Order Execution Testing:**
  Verify that orders are executed with correct prices and commissions.

- **Account Balance Testing:**
  Ensure that account balances are correctly updated after trades.

- **Position Tracking Testing:**
  Validate that positions are properly maintained and updated.

## 7. LLM-Specific Instructions
- **Code Regeneration Hints:**
  - Follow the Broker interface when implementing new broker models
  - Maintain separation between execution logic and account management
  - Ensure proper order validation before execution
  - Track commissions and slippage accurately

- **Contextual Guidance:**
  - The Broker module represents the trading venue or exchange
  - It provides a more complete simulation of a real brokerage than the Execution module
  - Account state in the Broker is independent from the Portfolio module's tracking

- **Examples of Extension:**
  - Implementing a live trading broker that connects to real exchanges
  - Adding support for margin trading and leverage
  - Creating exchange-specific broker implementations with unique fee structures
  - Implementing additional order types (OCO, trailing stops, etc.)

## 8. Versioning & Change Log
- **Version 1.0:**
  - Initial implementation with basic simulated broker
  - Support for market and limit orders
  - Simple commission and slippage models
  - Basic account and position tracking

- **Future Roadmap:**
  - Add support for margin trading and short selling
  - Implement more sophisticated commission models
  - Create connectors for live trading with real brokers
  - Add support for multi-currency accounts 