# Backtesting Engine Architecture and Features

The backtesting engine is the core that ties data and strategy together to simulate trades over historical data. We will build it with an **event-driven architecture** common in trading systems ([Backtesting Systematic Trading Strategies in Python: Considerations and Open Source Frameworks | QuantStart](https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks/#:~:text=Standard%20capabilities%20of%20open%20source,backtesting%20platforms%20seem%20to%20include)), as this provides flexibility in simulating realistic trading conditions. Here’s the proposed architecture and key components:

**Event Loop / Timeline Simulation**: The engine will iterate through the historical data timeline one step at a time (each step could be one candle interval, e.g., 1 minute or 1 day). At each step:

1. **Market Data Event**: the engine takes the next data point (price candle) and updates the strategy’s data context (so the latest price is available to the strategy).
2. **Strategy Signal Evaluation**: the strategy’s `next()` method is called with the updated data. The strategy’s logic can analyze the current (and past) data and decide to place orders.
3. **Order Execution**: If the strategy issues any buy/sell orders, the engine simulates their execution according to predefined rules (e.g., assume market order executed at the close price of the current bar, or open of next bar – this could be configurable).
4. **Portfolio Update**: The engine updates the portfolio state – positions held, available cash, inventory of coins, etc. – based on executed trades. P/L (profit and loss) is updated accordingly.
5. **Record Metrics**: The engine records the portfolio value after this step, which will be used for performance metrics. It may also log the trade in a trade list.

This loop continues until the end of the data range. The result will be a list of executed trades and a time series of portfolio value (equity curve).

**Order Types and Execution Rules**: Initially, we focus on simplicity:

- Support **market orders** (buy or sell immediately at the current market price). A buy will take the available cash in the portfolio to purchase as much crypto as specified (or as much as possible if size not specified). A sell will liquidate the position or a specified amount.
- You can also allow **limit orders** or others, but that requires more complex simulation (need to check if the high/low of subsequent bars hit the limit price, etc.). This can be an extension; to start, assume strategies use market orders at bar close or next open.
- Decide on execution price: A common approach in backtesting is if a signal is generated during bar *t* (using price at close of bar t), the order is executed at the **open of bar t+1** (next bar) to avoid look-ahead bias (since you wouldn't know the closing price of bar t until it's closed) ([Backtesting Systematic Trading Strategies in Python: Considerations and Open Source Frameworks | QuantStart](https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks/#:~:text=Standard%20capabilities%20of%20open%20source,backtesting%20platforms%20seem%20to%20include)). Another simpler approach: execute at close of the same bar (assuming the strategy acts right on the close). We need to choose one and document it. Executing on next bar’s open is safer to avoid hindsight bias.
- **Slippage and fees**: Incorporate a basic model for these:
    - **Trading fees**: Binance charges a fee per trade (e.g., 0.1%). The engine can deduct a small percentage of the trade amount as a fee. This can be a parameter in the config (so users can adjust or set to 0 to ignore). Including fees makes backtest results more realistic.
    - **Slippage**: The difference between expected price and actual fill price. In a highly liquid market like BTCUSDT with market orders, slippage might be minimal, but in volatile or low liquidity scenarios it can matter. As a starting point, we might ignore slippage or allow the user to specify a fixed slippage (e.g., buy price 0.05% worse than ideal). A more sophisticated approach (random slippage or based on volume) can be added later.

**Portfolio and Risk Management**: The engine will maintain a representation of the user’s portfolio:

- Track **cash balance** (fiat or stablecoin, e.g., USD or USDT) and **crypto holdings** (units of BTC, etc.).
- For simplicity, start with base currency as USD (or USDT) and one asset. The user can start with an initial capital (e.g., $10,000).
- Each buy trade will reduce cash and increase crypto holding; each sell does the opposite. If short selling is to be allowed (maybe via derivatives), that adds complexity (borrowed asset, margin, etc.) – possibly hold off on short support until we handle simpler long-only, unless the target user base explicitly needs shorting.
- Risk management can be partly handled by the strategy (user decides how much to buy). The engine can also enforce basics like not allowing buy orders larger than cash available (or if margin is allowed, enforce margin limits).
- We can allow position sizing either as absolute amount or fraction of portfolio. For instance, provide helper in strategy: `self.buy(percent=50)` to invest 50% of current cash, etc., to make it easier.
- The engine can also simulate **stop-loss and take-profit** if the strategy sets them (e.g., strategy could specify a stop price with an order, then engine monitors future prices to trigger it).

**Performance Metric Calculations**: After the backtest completes, the system will calculate key metrics to evaluate the strategy’s performance:

- **Sharpe Ratio**: Measures risk-adjusted return. It is calculated as the average excess return (over risk-free rate, which we can take as 0 for simplicity) divided by the standard deviation of returns, annualized ([Event-Driven Backtesting with Python - Part VII | QuantStart](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-VII/#:~:text=%5Cbegin%7Beqnarray,R_b%29%7D%7D%20%5Cend%7Beqnarray)). We will compute the strategy’s daily (or period) returns from the equity curve and then Sharpe = sqrt(N) * (mean(return) / std(return)), where N is the number of periods per year (365 if daily data, 252 trading days for daily, 525600 for minute data in a year, etc.) ([Event-Driven Backtesting with Python - Part VII | QuantStart](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-VII/#:~:text=Usually%20this%20value%20is%20set,60%3D98280)). This tells us how consistently the strategy earns returns relative to volatility.
- **Max Drawdown**: The largest peak-to-trough loss on the equity curve ([Event-Driven Backtesting with Python - Part VII | QuantStart](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-VII/#:~:text=The%20maximum%20drawdown%20and%20drawdown,periods%20over%20which%20it%20occurs)). We will scan the equity curve to find the maximum percentage drop from a historical peak. This indicates the worst-case loss an investor might have faced. We’ll also note the duration of that drawdown (how long until a new high is reached) ([Event-Driven Backtesting with Python - Part VII | QuantStart](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-VII/#:~:text=The%20maximum%20drawdown%20and%20drawdown,periods%20over%20which%20it%20occurs)).
- **Win Rate**: The percentage of trades that were profitable. We’ll count total trades and how many ended with a positive profit. For example, 40 wins out of 100 trades = 40% win rate. This helps understand how often the strategy wins, though not the magnitude of wins vs losses.
- **Profit Factor**: The ratio of total gross profit to total gross loss ([Profit Factor Explained - What Profit Factors Means and How It's Calculated](https://therobusttrader.com/profit-factor/#:~:text=The%20profit%20factor%20simply%20is,strategy%20degradation%2C%20which%20is%20inevitable)). For example, if all winning trades summed to $500 and all losing trades summed to -$250, profit factor = 500/250 = 2 ([Profit Factor Explained - What Profit Factors Means and How It's Calculated](https://therobusttrader.com/profit-factor/#:~:text=The%20profit%20factor%20simply%20is,strategy%20degradation%2C%20which%20is%20inevitable)). A profit factor > 1 means the strategy made more money than it lost ( > 1 is good, < 1 indicates an unprofitable strategy) ([Profit Factor Explained - What Profit Factors Means and How It's Calculated](https://therobusttrader.com/profit-factor/#:~:text=The%20profit%20factor%20simply%20is,strategy%20degradation%2C%20which%20is%20inevitable)). This metric emphasizes the balance between gains and losses.
- **Return and Volatility**: Overall return (percent growth of portfolio) and perhaps annualized volatility of returns (standard deviation). These feed into Sharpe but are also good standalone metrics. We can also compute **CAGR** (compound annual growth rate) if the test spans a long period, to normalize performance per year.
- **Other metrics**: Depending on user needs, we might include:
    - *Sortino ratio* (like Sharpe but penalizing only downside volatility).
    - *Calmar ratio* (annual return / max drawdown, a measure of return vs risk).
    - *Alpha/Beta* relative to a benchmark (if comparing to, say, simply holding Bitcoin).
    - *Exposure* (what % of time the strategy is in the market vs in cash).
    - *Average trade return*, *average holding period* of trades, *expectancy* (average profit per trade).

Many of these metrics are standard and supported by existing libraries ([Backtesting Systematic Trading Strategies in Python: Considerations and Open Source Frameworks | QuantStart](https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks/#:~:text=Performance%20testing%20applies%20the%20STS,statistics)). We will ensure the output includes at least the metrics mentioned and possibly more in a summary report. This can be printed to console in a readable format or output as JSON (if using API mode) for further analysis.

**Example Output** (for a backtest run):

```
Strategy: MovingAverageCross
Period: 2020-01-01 to 2021-01-01
Initial capital: $10,000
Final capital: $15,500
Total Return: +55%
Annualized Sharpe Ratio: 1.8
Max Drawdown: -20%
Win Rate: 45% (18/40 trades)
Profit Factor: 1.7

```

And possibly a breakdown of each trade or monthly returns, etc., if needed.

**Validation**: We should validate the backtest results for correctness. A best practice is to cross-check a simple strategy’s output with known results. For example, a buy-and-hold strategy on BTC should roughly match the actual price change over the period (minus fees). We can test that as a sanity check. Also, ensure that no metric is calculated with lookahead bias (all metrics are derived from the completed equity curve after simulation).

**Modularity**: Internally, structure the engine with modular components:

- A **Data Handler** (feeds data into the system, could be live or historical).
- A **Broker/Execution Handler** (simulates trade execution in backtest; could be swapped with a real broker interface later for live trading).
- A **Portfolio** object to track positions and value.
- A **Strategy** object (user-defined).
- These components interact via the event loop. This modular design aligns with how many systems (like QSTrader) are built, making it easier to extend or replace parts later ([Backtesting Systematic Trading Strategies in Python: Considerations and Open Source Frameworks | QuantStart](https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks/#:~:text=Both%20backtesting%20and%20live%20trading,often%20identical%20across%20both%20deployments)).

**Logging and Debugging**: Provide options to log details of each step for debugging. For example, a verbose mode can log each trade executed with time, price, size, and resulting balance. This helps users diagnose whether the strategy behaved as expected. Keep this optional to avoid slowing down normal runs.

**Potential Challenges**:

- Ensuring the simulation is realistic but also not overly complex for beginners. We might start with simple assumptions (instant execution at bar price, no slippage) and gradually add realism.
- Edge cases like what if strategy tries to sell when no position, or buy with insufficient funds – handle these gracefully (ignore the order or throw a warning).
- Floating point precision when calculating portfolio value (especially if dealing with many decimal places in crypto amounts) – use Python’s `decimal` or careful rounding for currency values.
- Computational efficiency: a naive loop in Python over thousands of bars is fine, but if doing minute-by-minute for multiple years (which could be >1 million data points), performance could degrade. We will address optimizations in the next section.