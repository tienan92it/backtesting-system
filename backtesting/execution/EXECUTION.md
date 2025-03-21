# Execution Module Knowledge Base

## 1. Module Overview
- **Module Name & Version:**  
  `Execution Module v1.0`
  
- **Purpose & Scope:**  
  The Execution module is responsible for simulating the execution of orders in the backtesting system. It handles the conversion of OrderEvents into FillEvents with realistic trading conditions, including commission and slippage.

- **Key Responsibilities:**  
  - Execute orders based on order parameters and market conditions
  - Apply realistic slippage models to execution prices
  - Calculate trading commissions
  - Generate FillEvents for executed orders
  - Simulate various order types (market, limit, stop, etc.)
  
- **Dependencies & Interfaces:**  
  - Receives OrderEvents from the Portfolio module
  - Generates FillEvents for the Portfolio module
  - Interfaces with DataHandler for current market prices

## 2. Detailed Functionality & Responsibilities

### Order Execution
- **Market Orders:**
  - Execute immediately at current market price
  - Apply slippage based on order direction (buy/sell)
  - Calculate fill price as close +/- slippage for buys/sells

- **Limit Orders:**
  - Execute only if limit price is reached
  - Order fills at specified limit price with slippage
  - Return no fill if limit conditions aren't met

- **Stop and Stop-Limit Orders:**
  - Monitor price movements for stop triggers
  - Convert to market or limit orders when triggered
  - Apply appropriate slippage and execution rules

### Slippage & Commission
- **Slippage Models:**
  - Fixed rate slippage (percentage-based)
  - Configurable based on asset volatility or liquidity
  - Applied in the direction that worsens execution price

- **Commission Models:**
  - Percentage-based commission rates
  - Fixed per-trade commission
  - Tiered commission structures

## 3. API / Interface Description

### ExecutionHandler Abstract Base Class
```python
class ExecutionHandler(ABC):
    @abstractmethod
    def initialize(self, data_handler, commission: float = 0.001, slippage: float = 0.0) -> None:
        # Initialize the execution handler with data handler and execution parameters
        
    @abstractmethod
    def execute_order(self, order_event: OrderEvent) -> Optional[FillEvent]:
        # Execute an order and generate a fill event
```

### SimulatedExecutionHandler Implementation
```python
class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self):
        # Initialize the simulated execution handler
        
    def initialize(self, data_handler, commission: float = 0.001, slippage: float = 0.0) -> None:
        # Initialize with data handler and execution parameters
        
    def execute_order(self, order_event: OrderEvent) -> Optional[FillEvent]:
        # Execute order and generate fill event
        
    def _calculate_execution_price(self, bar_data: Dict[str, float], direction: str, 
                                  order_type: str, limit_price: Optional[float] = None) -> Optional[float]:
        # Calculate execution price with slippage
        
    def _calculate_commission(self, price: float, quantity: float) -> float:
        # Calculate commission for a trade
```

## 4. Usage Examples

### Basic Execution Handler Initialization
```python
from backtesting.execution.simulated_execution import SimulatedExecutionHandler
from backtesting.data.binance_data import BinanceDataHandler

# Create data handler
data_handler = BinanceDataHandler()

# Create execution handler
execution_handler = SimulatedExecutionHandler()

# Initialize with custom parameters
execution_handler.initialize(
    data_handler=data_handler,
    commission=0.001,  # 0.1% commission
    slippage=0.0005    # 0.05% slippage
)
```

### Order Execution Example
```python
from backtesting.engine.event import OrderEvent
from datetime import datetime

# Create a market buy order
order_event = OrderEvent(
    timestamp=datetime.now(),
    symbol='BTC/USD',
    order_type='MARKET',
    quantity=0.5,
    direction='BUY',
    price=None  # Market order, no specific price
)

# Execute the order
fill_event = execution_handler.execute_order(order_event)

if fill_event:
    print(f"Order filled at {fill_event.fill_price} with commission {fill_event.commission}")
else:
    print("Order could not be executed")
```

### Limit Order Example
```python
# Create a limit buy order
limit_order = OrderEvent(
    timestamp=datetime.now(),
    symbol='ETH/USD',
    order_type='LIMIT',
    quantity=5.0,
    direction='BUY',
    limit_price=1500.0  # Only buy if price is at or below $1500
)

# Execute order
fill_event = execution_handler.execute_order(limit_order)

# May return None if limit price not reached
if not fill_event:
    print("Limit order not executed - price condition not met")
```

## 5. Configuration & Environment Setup
- **Required Python Version:** 3.7+
- **Dependencies:**
  - No external dependencies beyond standard libraries
  - Relies on internal system components (DataHandler, Event classes)
- **Configuration Options:**
  - Commission rate: Percentage cost per trade
  - Slippage model: How execution price deviates from ideal price
  - Order types: Support for market, limit, stop, etc.

## 6. Testing & Validation
- **Order Execution Testing:**
  Test that orders are executed correctly with appropriate prices.

- **Commission Calculation Testing:**
  Verify that commissions are accurately calculated for different order sizes.

- **Slippage Model Testing:**
  Ensure slippage is applied in the appropriate direction based on order type.

## 7. LLM-Specific Instructions
- **Code Regeneration Hints:**
  - Follow the ExecutionHandler interface when implementing new execution models
  - Ensure proper validation of order parameters before execution
  - Apply slippage consistently in the direction that hurts the trader (higher for buys, lower for sells)
  - Log all execution details for debugging

- **Contextual Guidance:**
  - The execution module simulates what would happen in a real brokerage
  - Realistic order execution is critical for accurate backtesting results
  - Commission and slippage significantly impact strategy profitability

- **Examples of Extension:**
  - Implementing volume-dependent slippage models
  - Adding support for more exotic order types (OCO, trailing stops)
  - Creating broker-specific execution handlers with unique fee structures
  - Implementing partial fills for large orders

## 8. Versioning & Change Log
- **Version 1.0:**
  - Initial implementation with basic order execution
  - Support for market and limit orders
  - Simple commission and slippage models

- **Future Roadmap:**
  - Add more sophisticated slippage models based on liquidity
  - Implement volume-based execution constraints
  - Add support for partial fills
  - Integrate live trading execution bridge 