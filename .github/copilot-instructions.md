# Backtesting Engine Code Guidelines

• Use Pandas DataFrames for managing OHLCV data; retrieve market data with CCXT.

• Design an event-driven loop that invokes the strategy’s next() method and simulates order execution at the next bar’s open.

• Build a modular Strategy interface for generating trade signals, supporting pandas-ta and scikit-learn integration.

• Implement a Broker/Execution Handler to simulate orders, fees, and slippage, updating positions and equity.

• Provide a clear CLI/API for running backtests and validating performance (e.g., against buy-and-hold strategies).
