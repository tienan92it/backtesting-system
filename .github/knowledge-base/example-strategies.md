# Example Strategies (Technical and ML)

To illustrate how the system works, here are two example strategies implemented using the planned interface: one uses technical indicators, and the other uses a machine learning model. These examples demonstrate how a user would define their strategy and how the backtesting results would be derived.

### Example 1: Moving Average Crossover Strategy (Technical Indicator)

**Description**: A classic strategy that uses two moving averages of different lengths. The strategy buys when a short-term moving average crosses above a long-term moving average (signaling upward momentum) and sells when the short MA crosses below the long MA (signaling downward momentum). This is a simple trend-following strategy. It's widely used as a basic example and can be optimized by varying the periods ([Backtesting Systematic Trading Strategies in Python: Considerations and Open Source Frameworks | QuantStart](https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks/#:~:text=In%20the%20context%20of%20strategies,must%20be%20calculated%20%26%20ranked)).

**Implementation**:

- **Indicator setup**: We choose a 50-day and a 200-day moving average on the closing price. In `init()`, compute these:

In this pseudocode, `self.data` is assumed to be a DataFrame that the engine updates to include the current and past prices (or we maintain an index). The strategy checks the last two values of each MA to determine if a crossover occurred on this period. We buy on a golden cross and sell on a death cross. (This example assumes we either are in cash or holding one position at a time; an enhancement is to track if we already bought and only sell then, etc., but assume the engine or strategy logic manages single position).
    
    ```python
    class MovingAverageCrossStrategy(Strategy):
        def init(self):
            price = self.data['close']
            self.ma_short = price.rolling(window=50).mean()
            self.ma_long  = price.rolling(window=200).mean()
    
        def next(self):
            # Only act if both MAs have values (i.e., we've gone past the max period)
            if len(self.data) < 200:
                return  # not enough data yet
            # Get the last calculated values
            short_prev, short_curr = self.ma_short.iloc[-2], self.ma_short.iloc[-1]
            long_prev, long_curr = self.ma_long.iloc[-2], self.ma_long.iloc[-1]
            # Check for crossover events
            if short_curr > long_curr and short_prev <= long_prev:
                self.buy()   # Golden cross: go long
            elif short_curr < long_curr and short_prev >= long_prev:
                self.sell()  # Death cross: exit long (or go short, depending on strategy rules)
    
    ```
    
- **Position management**: With this logic, the strategy will buy at the first golden cross and then hold until a death cross triggers a sell. We should ensure the strategy doesn’t keep buying every bar after a cross (we can add a flag to mark that we’re in position). But the engine can also enforce that if you already have a position and you call `buy()` again, it either ignores or adds to the position. For simplicity, we could interpret subsequent `buy()` calls as increasing position (averaging in) and `sell()` as closing out.
- **Performance**: This strategy typically has a lower win rate but aims to catch big trends. The backtest would reveal metrics like Sharpe (which might not be high if there are long flat periods) and max drawdown. It’s useful to show how the system logs each trade: e.g., “Bought BTC on 2020-03-01 at $9000, Sold on 2020-09-01 at $12000, Profit = 33%”.

**Possible Output** (illustration):

```
Trades:
2020-03-01: BUY 1 BTC @ $9000
2020-09-01: SELL 1 BTC @ $12000   Profit: +$3000 (+33%)
...
Final portfolio value: $...
Sharpe: X, Max Drawdown: Y, Win rate: Z, Profit factor: W

```

This shows how a simple indicator strategy is defined and evaluated.

### Example 2: Machine Learning Mean-Reversion Strategy (ML-Based)

**Description**: This strategy uses a machine learning model to predict short-term price movements and trade accordingly. For instance, a regression model might predict the next day’s return, or a classification model might predict whether the price will go up or down tomorrow (up/down = 1/0). We’ll illustrate a mean-reversion idea: predict if the price is higher than its recent average (suggesting it might revert down) or lower (might revert up), and trade opposite to the prediction (buy if likely to go up). This is a simplified ML example.

**Implementation**:

- **Feature generation**: We create features like a 5-day return, 10-day moving average deviation, RSI, etc. In code:
    
    ```python
    class MLMeanReversionStrategy(Strategy):
        def init(self):
            # Prepare features and target for training
            df = self.data  # assume DataFrame with 'close'
            df['return_5'] = df['close'].pct_change(5)
            df['ma10'] = df['close'].rolling(10).mean()
            df['price_ma10_diff'] = df['close'] - df['ma10']
            df['rsi14'] = ta.rsi(df['close'], length=14)  # using pandas-ta for RSI
            df.dropna(inplace=True)
            # Target: will the price increase in next 5 days?
            df['future_5'] = df['close'].shift(-5)
            df['target'] = (df['future_5'] > df['close']).astype(int)
            df.dropna(inplace=True)
            # Split data into train/test (e.g., first 70% train, rest test)
            split = int(0.7 * len(df))
            train_data = df.iloc[:split]
            test_data = df.iloc[split:]
            X_train = train_data[['return_5', 'price_ma10_diff', 'rsi14']]
            y_train = train_data['target']
            # Train a classifier (e.g., Random Forest)
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, max_depth=5)
            self.model.fit(X_train, y_train)
            # Store test data for use in next()
            self.test_features = test_data[['return_5', 'price_ma10_diff', 'rsi14']]
            self.test_index = test_data.index  # the actual dates for these
            self.current_test_idx = 0
        def next(self):
            # Only start trading after training period
            if self.current_index < self.test_index[0]:
                return
            # Use the model to predict for the current index if it's in test set
            if self.current_index in self.test_index:
                i = self.test_index.get_loc(self.current_index)
                X = [self.test_features.iloc[i]]
                pred = self.model.predict(X)[0]  # 1 if price likely to rise, 0 if fall
                # Mean reversion logic: if model predicts rise, that means price was low -> buy
                # If model predicts fall, price was high -> sell/short or exit
                if pred == 1:
                    self.buy()
                else:
                    self.sell()
        ```
    This pseudocode is a bit involved. Key points:
      - We prepared features (5-day return, difference from 10-day MA, RSI) and the target (whether price 5 days later is higher).
      - We trained a RandomForest on 70% of the data.
      - In `next()`, for each new day in the remaining 30%, we take the features and predict the target. If the model predicts the price will go up, we buy (anticipating a rise). If it predicts down, we sell (anticipating a drop).
      - This is a basic example; in reality, one might retrain periodically (walk-forward) rather than one static train/test split, but the concept is shown.
      - Also, in mean reversion, often you *sell* when price is predicted to rise above average (assuming it will revert down) – but here we took a more intuitive approach: trust the model’s prediction direction.
    
    ```
    
- **Considerations**: We must be careful to avoid lookahead bias. In the above, we used future_5 shifted column to create labels, then dropped future data for training – that’s correct for training. During backtest, we ensure we only predict on data the model hasn’t seen (we split the dataset). If doing a rolling retrain, we would update the model as time advances.
- **Trading logic**: We decided if prediction = 1 (price will be higher in 5 days), then buy now to profit from that rise. We might also implement an exit after 5 days or when the model says opposite – to keep trades from running indefinitely. For simplicity, this code buys or sells whenever the model says so, which could result in frequent switching. In practice, one might hold for a fixed period or until a contrary signal.

**Expected outcome**: The performance of an ML strategy depends on the model’s accuracy. This example might yield a moderate win rate if the model finds some patterns. The profit factor and Sharpe would tell us if it’s better than random. This demonstrates the system’s ability to integrate ML – the heavy part is training the model, which we did in `init()`, and then using it in `next()` for decisions.

**Notes**: We would highlight that the above code is pseudocode for illustration. In reality, certain adjustments (like scaling features, avoiding forward-looking in validation, etc.) are needed for a robust ML strategy. But it shows how a user could embed an ML pipeline into the strategy class.

---

Both examples show that **users can customize the strategy logic arbitrarily** – whether it’s a simple indicator threshold or a complex ML prediction, as long as they adhere to the interface (computing signals and calling buy/sell), the backtesting engine will handle the rest (data feeding, trade execution, performance tracking).

### Additional Strategy Examples:

- **RSI Contrarian Strategy**: Buy when RSI (14-day Relative Strength Index) falls below 30 (oversold, anticipating a bounce) and sell when RSI rises above 70 (overbought, anticipating a drop).
- **Breakout Strategy**: Track the highest high of the past N days; buy when price exceeds that (breakout) and sell when price falls below the recent N-day low.
- **Neural Network Strategy** (ML): Use a neural network to predict next hour’s return, trained on a sliding window of past prices (sequence modeling). This could be implemented in TensorFlow/Keras and integrated similarly, just with a more complex prediction function.

Each of these can be implemented in our framework, confirming that our design supports a wide range of strategies.