# Crypto Trading Backtesting System

A Python-based backtesting system for crypto trading strategies, designed for individual traders and quant engineers.

```mermaid
graph TB
    %% Main Components
    DataHandler["DataHandler\n(BinanceData/CCXTData)"]
    Strategy["Strategy\n(base.py)"]
    Portfolio["Portfolio\n(portfolio.py)"]
    ExecutionHandler["ExecutionHandler\n(simulated_execution.py)"]
    EventLoop["EventLoop\n(event_loop.py)"]
    Backtester["Backtester\n(backtester.py)"]
    Metrics["Metrics\n(performance.py)"]

    %% Event types
    MarketEvent[MarketEvent]
    SignalEvent[SignalEvent]
    OrderEvent[OrderEvent]
    FillEvent[FillEvent]
    PortfolioEvent[PortfolioEvent]

    %% Workflow initialization
    Backtester -->|1. initialize| DataHandler
    Backtester -->|1. initialize| Strategy
    Backtester -->|1. initialize| Portfolio
    Backtester -->|1. initialize| ExecutionHandler
    Backtester -->|1. initialize| EventLoop

    %% Event flow
    Backtester -->|2. call get_all_bars| DataHandler
    DataHandler -->|3. return historical data| Backtester
    Backtester -->|4. create & add| MarketEvent
    
    %% Event processing
    EventLoop -->|5. process| MarketEvent
    MarketEvent -->|6. update current_bar| DataHandler
    MarketEvent -->|7. on_data| Strategy
    Strategy -->|8. generate & add| SignalEvent
    
    EventLoop -->|9. process| SignalEvent
    SignalEvent -->|10. on_signal| Portfolio
    Portfolio -->|11. generate & add| OrderEvent
    
    EventLoop -->|12. process| OrderEvent
    OrderEvent -->|13. execute_order| ExecutionHandler
    ExecutionHandler -->|14. generate & add| FillEvent
    
    EventLoop -->|15. process| FillEvent
    FillEvent -->|16. on_fill/update_fill| Portfolio
    Portfolio -->|17. generate & add| PortfolioEvent
    
    EventLoop -->|18. process| PortfolioEvent
    PortfolioEvent -->|19. update results| Backtester

    %% Final results processing
    Backtester -->|20. calculate results| Metrics
    Metrics -->|21. return metrics| Backtester

    %% Styling
    classDef component fill:#f9f,stroke:#333,stroke-width:2px;
    classDef event fill:#bbf,stroke:#333,stroke-width:1px;
    classDef core fill:#bfb,stroke:#333,stroke-width:2px;
    
    class DataHandler,Strategy,Portfolio,ExecutionHandler component;
    class MarketEvent,SignalEvent,OrderEvent,FillEvent,PortfolioEvent event;
    class EventLoop,Backtester,Metrics core;
```

## Overview

This system allows you to:
- Fetch historical cryptocurrency data (initially from Binance)
- Backtest trading strategies on historical data
- Implement technical indicator-based or machine learning-based strategies
- Evaluate performance with standard metrics (Sharpe ratio, drawdown, win rate)
- Run backtests via command line or Python API

## Project Structure

```
backtesting-system/
├── backtesting/             # Main package
│   ├── data/                # Data handling modules
│   ├── engine/              # Backtesting engine core
│   ├── broker/              # Order execution simulation
│   ├── strategy/            # Strategy interface and examples
│   ├── portfolio/           # Portfolio tracking
│   └── metrics/             # Performance metrics
├── cli/                     # Command-line interface
├── examples/                # Example strategies
└── tests/                   # Unit tests
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/backtesting-system.git
cd backtesting-system

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run a backtest with an example strategy
python -m cli.backtest_cli --strategy examples.simple_ma_cross.MovingAverageCrossStrategy --symbol BTCUSDT --start 2022-01-01 --end 2022-12-31
```

## Creating Custom Strategies

Inherit from the base Strategy class:

```python
from backtesting.strategy.base import Strategy

class MyCustomStrategy(Strategy):
    def init(self):
        # Setup indicators, models, etc.
        self.sma_short = self.data['close'].rolling(window=50).mean()
        self.sma_long = self.data['close'].rolling(window=200).mean()
        
    def next(self):
        # Trading logic for each new price bar
        if self.sma_short[-1] > self.sma_long[-1] and self.sma_short[-2] <= self.sma_long[-2]:
            self.buy()
        elif self.sma_short[-1] < self.sma_long[-1] and self.sma_short[-2] >= self.sma_long[-2]:
            self.sell()
```

## License

MIT
