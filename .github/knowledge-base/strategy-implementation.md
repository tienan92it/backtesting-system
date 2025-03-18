# Strategy Implementation (User-Defined Strategies)

The system will allow users to define and plug in their own trading strategies through scripting (Python code). Flexibility and ease of use here are crucial: both less-technical traders and experienced quants should find it straightforward to write or import their strategy logic.

**Strategy Interface**: Design a clear interface (could be a base class or a set of functions) that a user’s strategy must follow:

- One approach is an object-oriented **Strategy Base Class**. Users create a subclass (e.g., `class MyStrategy(Strategy):`) and implement specific methods:
    - `init(self)`: for one-time initialization, like calculating indicators or loading models.
    - `next(self)`: core logic that runs on each new bar/tick. This function can access current data (price, indicators) and decide whether to place orders (buy/sell) or hold. This pattern is used by Backtrader (where `next()` is called for each step) ([Backtrader for Backtesting (Python) - A Complete Guide - AlgoTrading101 Blog](https://algotrading101.com/learn/backtrader-for-backtesting/#:~:text=match%20at%20L364%20,the%20next%20new%20data%20point)). It’s intuitive: the backtesting engine iterates over the data and calls `strategy.next()` with the latest price, and inside `next()` the user can define if conditions for trades are met ([Backtrader for Backtesting (Python) - A Complete Guide - AlgoTrading101 Blog](https://algotrading101.com/learn/backtrader-for-backtesting/#:~:text=match%20at%20L364%20,the%20next%20new%20data%20point)).
    - Possibly `on_start` or `on_finish`: hooks for actions at the start or end of a backtest (e.g., to train an ML model at start, or log final results at end).
    - The base class can also provide helper methods like `self.buy(price=None, size=None)` and `self.sell(...)` to place simulated orders, which actually enqueue an order in the engine to execute at the appropriate price.
- Alternatively, a **functional interface** could be offered, where the user provides a function that takes in past data (or the current index) and returns a signal. For example, a function `generate_signal(data_slice) -> {"action": "buy"/"sell"/None, "size": amount}`. The engine would call this for each time step. This is simpler to get started, but less powerful than a class-based approach if the strategy needs to maintain internal state or do complex things. A hybrid approach is possible: accept either a function or a class implementing a known pattern.

Given our target users, providing the class-based approach with an example template is a good starting point, as it mirrors other frameworks and allows more complexity. We can supply a **strategy template file** that users can copy and modify, which includes placeholders for init and next.

**Technical Indicator Strategies**: For indicator-based strategies, the system should make it easy to compute indicators and generate signals:

- **Indicator calculation**: In the `init()` of the strategy, users can compute any required indicators. For instance, using Pandas or TA-Lib: `self.ma_short = talib.SMA(self.data['close'], timeperiod=50)` to get a 50-period moving average array, and `self.ma_long = talib.SMA(self.data['close'], timeperiod=200)` for a 200-period MA. These could be stored as attributes or within the `data` structure.
- We might also pre-compute common indicators at the data loading stage if performance is a concern (e.g., have a utility to add columns to the DataFrame for popular indicators upon request). But flexibility suggests letting the user compute what they need.
- **Signal logic**: In `next()`, the user can write conditions based on current values of these indicators. For example:

This pseudocode checks the latest two values of moving averages for a crossover event to decide buy/sell. The system will provide utility functions like a `crossover(series1, series2)` detector to simplify such common patterns.
    
    ```python
    if self.ma_short[-1] > self.ma_long[-1] and self.ma_short[-2] <= self.ma_long[-2]:
        self.buy()   # golden crossover: short MA crossed above long MA
    elif self.ma_short[-1] < self.ma_long[-1] and self.ma_short[-2] >= self.ma_long[-2]:
        self.sell()  # death crossover
    
    ```
    
- The strategy should have access to any relevant data: open, high, low, close, volume, or even multiple symbols if multi-asset strategies are supported later. Initially, focus on single-symbol strategies for simplicity (multi-asset can be an extension).

**Machine Learning Strategies**: ML-based strategies can be more involved, but the framework will allow them by giving the strategy full access to data and the ability to incorporate predictive models:

- **Feature Engineering**: The user (or a provided helper) can create features from the historical data. For example, features could be technical indicators or recent returns. This can be done in `init()`: e.g., compute a DataFrame of features like past N returns, moving avg values, RSI values, etc.
- **Training**: A common approach is to train a model on a portion of historical data (e.g., first 70% of the period) and then test on the remaining 30%. The strategy could split the data internally or use cross-validation. In a simple scenario, the user might train the model on all past data up to the current moment in a rolling fashion (though that risks forward-looking bias if not careful). For backtesting, often a two-phase approach is used: train on an initial in-sample period, then run the model on the out-of-sample period without retraining, or retrain periodically (walk-forward).
    - For instance, in `init()`, user trains a scikit-learn model:
    
    This would train on data up to 2021. Then `next()` from 2021 onward uses predictions.
        
        ```python
        X_train, y_train = prepare_dataset(self.data, train_end='2021-01-01')
        self.model = RandomForestClassifier().fit(X_train, y_train)
        
        ```
        
    - Alternatively, the user might not train in `init()` but rather use a pre-trained model saved to disk (e.g., a TensorFlow model file) and load it in `init()`.
- **Generating Signals with ML**: In the `next()` method, at each time step, generate a prediction and act on it. For example, a model might predict the probability of the price going up in the next period. Pseudocode:

If short selling isn’t allowed or desired in crypto (many exchanges allow it via futures, but spot trading is long-only), the model might simply predict “buy” vs “stay in cash”, so `pred` could be binary where 1=buy, 0=sell/exit. The strategy can interpret that accordingly.
    
    ```python
    features = extract_features(self.data, index=self.current_index)  # get features for current time
    pred = self.model.predict(features)
    if pred == 1:  # model predicts price will rise
        self.buy()
    elif pred == -1:  # model predicts price will fall (if shorting is allowed)
        self.sell()
    
    ```
    
- **Handling ML specifics**: The strategy needs to avoid using future data for training at any point. The framework should **not pass future data points into the strategy inadvertently**. A challenge is that if a user naively uses the entire data to train, they’d be peeking into the future. We should educate users (via documentation) to split their data. We could provide built-in support for walk-forward validation: e.g., allow the user to specify a training window and retrain frequency, and the engine can automate training the model on past data for them as the simulation progresses (this is an advanced feature and can be added later; initially we rely on user to do the right thing).
- **Examples of ML strategies**: Could include regression to predict next period return, classification of up/down movement, clustering for regime detection, or even deep learning on sequences. Our system’s job is mainly to provide the data and allow trades; the heavy lifting of ML is up to the user’s code (or any libraries they import).

**Scripting Custom Strategies**: We will allow users to write strategy code in a separate Python file (or multiple files). The CLI can take the path to this file and import it to instantiate the strategy. For example, `--strategy-file my_strategy.py --class MyStrategy` could load that class via Python’s importlib. We must ensure this is done securely (only user-provided code is run) and handle errors gracefully (syntax errors in the strategy, etc., should be reported). Documentation will include examples of strategy files for both a simple indicator strategy and an ML strategy as templates.

**Best Practices**: Encourage users to:

- Clearly separate calculation of signals and execution of orders (the framework helps by handling execution once `buy()` is called, etc.).
- Use the provided data only up to the current index (the framework will likely provide something like `self.data.iloc[:current_index]` or similar if needed).
- Take into account transaction costs in their strategy logic if relevant (or rely on the engine’s transaction cost simulation, see next section).
- Test their strategy on a small timeframe first to ensure it behaves as expected (the CLI could offer a “dry run” or debugging mode to step through a few intervals).

**Potential Challenges**: One challenge is making the strategy interface general enough for both simple and advanced strategies. We need to accommodate things like:

- Strategies that hold positions vs. ones that constantly flip (the engine should support maintaining a position between bars).
- Maybe strategies that trade multiple pairs (this complicates data handling, perhaps out of scope initially).
- ML strategies that may need large amounts of data loaded (memory considerations if a user wants to train on tick data, etc.). We should document reasonable data sizes or suggest downsampling for backtests if needed.
- Ensuring that the user’s strategy errors do not crash the whole engine run – catch exceptions within strategy calls and report them with context.

By designing a clear strategy API and providing good examples, users should be able to write custom strategies with minimal friction. This customization is a core feature: unlike black-box platforms, our tool is meant to be open and scriptable, which quant engineers will appreciate.