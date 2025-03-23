import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import time

from backtesting.data import BinanceDataHandler
from backtesting.strategy import Strategy


class MLMeanReversionStrategy(Strategy):
    """
    A machine learning-based mean reversion strategy.
    
    This strategy uses a machine learning model to predict short-term price movements
    and trade accordingly. It creates features like 5-day return, moving average
    deviation, RSI, etc., and uses a Random Forest classifier to predict whether
    the price will increase in the next 5 days.
    
    If the model predicts the price will rise, we buy (anticipating a rise).
    If the model predicts the price will fall, we sell (anticipating a drop).
    """
    
    def __init__(self, train_size=0.7, prediction_horizon=5, ma_period=10,
                 rsi_period=14, confidence_threshold=0.55):
        """
        Initialize the ML strategy.
        
        Parameters:
        -----------
        train_size : float
            Portion of data to use for training (0.0-1.0).
        prediction_horizon : int
            How many bars ahead to predict.
        ma_period : int
            Period for moving average calculation.
        rsi_period : int
            Period for RSI calculation.
        confidence_threshold : float
            Threshold for model confidence to trigger trades.
        """
        super().__init__()
        self.params = {
            'train_size': train_size,
            'prediction_horizon': prediction_horizon,
            'ma_period': ma_period,
            'rsi_period': rsi_period,
            'confidence_threshold': confidence_threshold
        }
        
        self.model = None
        self.test_features = None
        self.test_index = None
        self.current_test_idx = 0
        self.signals = []
    
    def init(self):
        """
        Prepare features and train the ML model.
        Called once at the start of the backtest.
        """
        print("Initializing ML Mean Reversion Strategy...")
        
        # Ensure we have the required data
        if self.data is None or len(self.data) < 100:
            print("Not enough data for ML strategy")
            return
        
        # Create feature dataframe
        df = self.data.copy()
        
        # Feature 1: 5-day return
        df['return_5'] = df['close'].pct_change(5)
        
        # Feature 2: 10-day moving average
        df['ma10'] = df['close'].rolling(window=self.params['ma_period']).mean()
        
        # Feature 3: Price deviation from moving average
        df['price_ma10_diff'] = df['close'] - df['ma10']
        
        # Feature 4: RSI-14
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.params['rsi_period']).mean()
        avg_loss = loss.rolling(window=self.params['rsi_period']).mean()
        rs = avg_gain / avg_loss
        df['rsi14'] = 100 - (100 / (1 + rs))
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        # Create target: will price increase in next 5 days?
        horizon = self.params['prediction_horizon']
        df['future_5'] = df['close'].shift(-horizon)
        df['target'] = (df['future_5'] > df['close']).astype(int)
        
        # Remove rows with NaN targets
        df.dropna(inplace=True)
        
        # Split data into train/test sets
        split = int(self.params['train_size'] * len(df))
        train_data = df.iloc[:split]
        test_data = df.iloc[split:]
        
        # Prepare training datasets
        X_train = train_data[['return_5', 'price_ma10_diff', 'rsi14']]
        y_train = train_data['target']
        
        # Train a classifier
        print("Training Random Forest model...")
        start_time = time.time()
        
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(X_train, y_train)
        
        print(f"Model training completed in {time.time() - start_time:.2f} seconds")
        
        # Store test data for use in next()
        self.test_features = test_data[['return_5', 'price_ma10_diff', 'rsi14']]
        self.test_index = test_data.index
        
        # Print feature importance
        feature_importance = self.model.feature_importances_
        features = ['return_5', 'price_ma10_diff', 'rsi14']
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        print("\nFeature Importance:")
        for idx, row in importance_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    def next(self):
        """
        Use the model predictions to make trading decisions.
        Called for each new candle.
        """
        # Only start trading after training period
        if self.current_index < len(self.data) and self.data.index[self.current_index] not in self.test_index:
            return
        
        # Use the model to predict for the current index if it's in test set
        if self.current_index < len(self.data) and self.data.index[self.current_index] in self.test_index:
            current_date = self.data.index[self.current_index]
            idx = self.test_index.get_loc(current_date)
            X = self.test_features.iloc[idx:idx+1]
            
            # Get prediction and probability
            pred = self.model.predict(X)[0]
            probas = self.model.predict_proba(X)[0]
            confidence = probas[1] if pred == 1 else probas[0]
            
            # Record the signal with confidence
            self.signals.append({
                'date': current_date,
                'price': self.data.loc[current_date, 'close'],
                'prediction': pred,
                'confidence': confidence
            })
            
            # Mean reversion logic: 
            # If model is confident price will rise, buy
            # If model is confident price will fall, sell
            threshold = self.params['confidence_threshold']
            
            if pred == 1 and confidence > threshold:
                # Model predicts price will rise (with high confidence)
                if self.position <= 0:  # If we're not already long
                    self.buy()
            elif pred == 0 and confidence > threshold:
                # Model predicts price will fall (with high confidence)
                if self.position > 0:  # If we're currently long
                    self.sell()


def create_mock_data(periods=500):
    """
    Create mock price data for demonstration.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with mock OHLCV data.
    """
    dates = pd.date_range(start='2020-01-01', periods=periods, freq='D')
    
    # Create a price series with trend, cycle and noise
    trend = np.linspace(0, 20, periods)
    cycle = np.sin(np.linspace(0, 10, periods)) * 10
    noise = np.random.normal(0, 2, periods)
    
    close_prices = 100 + trend + cycle + noise
    
    # Create OHLCV data
    data = {
        'open': close_prices - np.random.uniform(0, 2, periods),
        'high': close_prices + np.random.uniform(1, 3, periods),
        'low': close_prices - np.random.uniform(1, 3, periods),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, periods)
    }
    
    return pd.DataFrame(data, index=dates)


def simulate_strategy(strategy, data):
    """
    Simulate a strategy on historical data.
    
    Parameters:
    -----------
    strategy : Strategy
        The strategy to test.
    data : pd.DataFrame
        Historical price data.
        
    Returns:
    --------
    tuple
        (signals, portfolio_values, trades)
    """
    # Set the data for the strategy
    strategy.set_data(data)
    
    # Variables to track portfolio
    initial_capital = 10000.0
    cash = initial_capital
    position = 0
    position_size = 0
    portfolio_values = []
    
    # Lists to store trades
    trades = []
    
    # Mock buy and sell functions
    def buy_func(size=None, price=None, limit_price=None, stop_price=None, percent=None):
        nonlocal cash, position, position_size
        
        # Determine price
        current_price = price if price is not None else data['close'].iloc[strategy.current_index]
        
        # Determine size
        if size is not None:
            shares = size
        elif percent is not None:
            shares = (cash * percent / 100) / current_price
        else:
            shares = cash / current_price
        
        # Execute trade
        cost = shares * current_price
        if cost <= cash:
            cash -= cost
            position = 1
            position_size += shares
            
            trade = {
                'type': 'buy',
                'timestamp': data.index[strategy.current_index],
                'price': current_price,
                'shares': shares,
                'cost': cost,
                'portfolio_value': cash + position_size * current_price
            }
            trades.append(trade)
            
            return trade
        return None
    
    def sell_func(size=None, price=None, limit_price=None, stop_price=None, percent=None):
        nonlocal cash, position, position_size
        
        # Determine price
        current_price = price if price is not None else data['close'].iloc[strategy.current_index]
        
        # Determine size
        if size is not None:
            shares = min(size, position_size)
        elif percent is not None:
            shares = position_size * (percent / 100)
        else:
            shares = position_size
        
        # Execute trade
        if shares > 0:
            proceeds = shares * current_price
            cash += proceeds
            position_size -= shares
            if position_size <= 0:
                position = 0
                position_size = 0
            
            trade = {
                'type': 'sell',
                'timestamp': data.index[strategy.current_index],
                'price': current_price,
                'shares': shares,
                'proceeds': proceeds,
                'portfolio_value': cash + position_size * current_price
            }
            trades.append(trade)
            
            return trade
        return None
    
    # Set the buy and sell functions
    strategy.set_backtester_functions(buy_func, sell_func)
    
    # Initialize the strategy
    strategy.init()
    
    # Run the backtest
    for i in range(len(data)):
        # Update the current index
        strategy.current_index = i
        
        # Update portfolio value
        current_price = data['close'].iloc[i]
        portfolio_value = cash + position_size * current_price
        portfolio_values.append(portfolio_value)
        
        # Update strategy's portfolio info
        strategy.update_portfolio(position, position_size, cash, portfolio_value)
        
        # Call next() to generate signals for this bar
        strategy.next()
    
    # Return results
    return strategy.signals, portfolio_values, trades


def plot_results(data, portfolio_values, trades, signals, strategy_name):
    """
    Plot strategy results.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data.
    portfolio_values : list
        Portfolio values over time.
    trades : list
        List of trades.
    signals : list
        List of signals generated by the strategy.
    strategy_name : str
        Name of the strategy.
    """
    # Create a directory for the charts
    os.makedirs('ml_strategy_charts', exist_ok=True)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), 
                                      gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot price and trades on top subplot
    ax1.plot(data.index, data['close'], label='Close Price')
    
    # Plot buy and sell points
    for trade in trades:
        if trade['type'] == 'buy':
            ax1.scatter(trade['timestamp'], trade['price'], marker='^', color='g', s=100)
        else:
            ax1.scatter(trade['timestamp'], trade['price'], marker='v', color='r', s=100)
    
    ax1.set_title(f'{strategy_name} - Price and Trades')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend()
    
    # Plot portfolio value on middle subplot
    ax2.plot(data.index, portfolio_values, label='Portfolio Value', color='blue')
    ax2.set_title('Portfolio Value')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    ax2.legend()
    
    # Plot signal confidence on bottom subplot - handle strategy.signals directly
    if hasattr(signals, '__len__') and len(signals) > 0:
        try:
            signal_dates = []
            signal_confidence = []
            prediction_colors = []
            
            for signal in signals:
                if isinstance(signal, dict) and 'date' in signal and 'confidence' in signal and 'prediction' in signal:
                    signal_dates.append(signal['date'])
                    signal_confidence.append(signal['confidence'])
                    prediction_colors.append('g' if signal['prediction'] == 1 else 'r')
            
            if signal_dates:
                ax3.scatter(signal_dates, signal_confidence, color=prediction_colors, alpha=0.7)
                ax3.axhline(y=0.5, color='black', linestyle='--')
                ax3.set_title('Model Prediction Confidence')
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Confidence')
                ax3.set_ylim(0, 1)
                ax3.grid(True)
            else:
                print("No valid signals with required keys found")
        except Exception as e:
            print(f"Error plotting signals: {e}")
    
    plt.tight_layout()
    plt.savefig(f'ml_strategy_charts/{strategy_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    print(f"Chart saved to ml_strategy_charts/{strategy_name.replace(' ', '_').lower()}.png")


def run_ml_strategy_example():
    """Run ML Mean Reversion Strategy Example"""
    # Generate or load data
    try:
        # Try to fetch real data
        data_handler = BinanceDataHandler(use_cache=True, cache_dir='example_data')
        symbol = 'BTCUSDT'
        timeframe = '1d'
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365 * 2)  # Last 2 years for more training data
        
        print(f"Fetching {symbol} {timeframe} data from {start_time.date()} to {end_time.date()}...")
        
        data = data_handler.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        data_source = f"{symbol} {timeframe}"
        
    except Exception as e:
        print(f"Error fetching real data: {e}")
        print("Using mock data instead.")
        data = create_mock_data(periods=500)
        data_source = "Mock Data"
    
    print(f"Data shape: {data.shape}")
    
    # Create the ML strategy
    strategy = MLMeanReversionStrategy(
        train_size=0.7,
        prediction_horizon=5,
        ma_period=10,
        rsi_period=14,
        confidence_threshold=0.6
    )
    
    # Run the backtest
    print("\nRunning ML Mean Reversion strategy backtest...")
    signals, portfolio_values, trades = simulate_strategy(strategy, data)
    
    # Print results
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_value - 1) * 100
    
    print(f"\nResults:")
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    
    # Calculate win rate
    if len(trades) > 1:
        profits = []
        for i in range(1, len(trades)):
            if trades[i-1]['type'] == 'buy' and trades[i]['type'] == 'sell':
                profit = trades[i]['price'] / trades[i-1]['price'] - 1
                profits.append(profit)
        
        if profits:
            win_rate = sum(1 for p in profits if p > 0) / len(profits) * 100
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Average Profit per Trade: {sum(profits)/len(profits)*100:.2f}%")
    
    # Plot results
    strategy_name = "ML Mean Reversion Strategy"
    plot_results(data, portfolio_values, trades, signals, strategy_name)


if __name__ == "__main__":
    run_ml_strategy_example()
