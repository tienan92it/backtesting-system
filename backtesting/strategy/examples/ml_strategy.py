import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from backtesting.strategy.base import Strategy


class MLMeanReversionStrategy(Strategy):
    """
    A machine learning-based mean reversion strategy.
    Uses Random Forest to predict whether price will revert to mean.
    
    Features include RSI, price deviation from moving averages, 
    historical volatility, and other technical indicators.
    
    The model predicts whether the price will go up (1) or down (0)
    in the next `prediction_horizon` bars.
    """
    
    def __init__(self, train_size: float = 0.7, prediction_horizon: int = 5,
                 ma_periods: List[int] = [10, 20, 50, 200],
                 rsi_period: int = 14, volatility_period: int = 20):
        """
        Initialize the ML strategy.
        
        Parameters:
        -----------
        train_size : float
            Portion of data to use for training (0.0-1.0).
        prediction_horizon : int
            How many bars ahead to predict.
        ma_periods : list
            Periods for moving averages to use as features.
        rsi_period : int
            Period for RSI calculation.
        volatility_period : int
            Period for volatility calculation.
        """
        super().__init__()
        self.params = {
            'train_size': train_size,
            'prediction_horizon': prediction_horizon,
            'ma_periods': ma_periods,
            'rsi_period': rsi_period,
            'volatility_period': volatility_period
        }
        
        # Will be populated during init()
        self.model = None
        self.scaler = None
        self.features = None
        self.predictions = None
        self.train_test_split_idx = None
    
    def init(self) -> None:
        """
        Prepare features and train the ML model.
        Called once at the start of the backtest.
        """
        # Create features
        features_df = self._create_features()
        
        # Create target variable (future returns)
        horizon = self.params['prediction_horizon']
        future_returns = self.data['close'].pct_change(horizon).shift(-horizon)
        target = (future_returns > 0).astype(int)  # 1 if price goes up, 0 if down
        
        # Remove NaNs
        valid_idx = ~(features_df.isna().any(axis=1) | target.isna())
        features_df = features_df[valid_idx]
        target = target[valid_idx]
        
        # Split data into train/test sets
        train_size = int(len(features_df) * self.params['train_size'])
        self.train_test_split_idx = train_size
        
        X_train = features_df.iloc[:train_size]
        y_train = target.iloc[:train_size]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Store features for use in next()
        self.features = features_df
        
        # Make predictions for the test set
        test_features = features_df.iloc[train_size:]
        test_features_scaled = self.scaler.transform(test_features)
        
        # Predict probabilities (probability of price going up)
        probas = self.model.predict_proba(test_features_scaled)[:, 1]
        
        # Create a Series with predictions (NaN for training data)
        self.predictions = pd.Series(index=self.data.index, data=np.nan)
        self.predictions.iloc[train_size + valid_idx.values[train_size:].cumsum()] = probas
    
    def next(self) -> None:
        """
        Use the model predictions to make trading decisions.
        Called for each new candle.
        """
        # Skip if we're still in the training set or if we don't have a prediction
        if (self.current_index < self.train_test_split_idx or 
            np.isnan(self.predictions.iloc[self.current_index])):
            return
        
        # Get the prediction probability
        probability = self.predictions.iloc[self.current_index]
        
        # If model is confident about price going up (> 0.7) and we're not already in a position
        if probability > 0.7 and self.position <= 0:
            self.buy()
        
        # If model is confident about price going down (< 0.3) and we're in a position
        elif probability < 0.3 and self.position > 0:
            self.sell()
    
    def _create_features(self) -> pd.DataFrame:
        """
        Create technical features for the ML model.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with technical indicators as features.
        """
        df = pd.DataFrame(index=self.data.index)
        close = self.data['close']
        
        # Price-based features
        
        # Moving averages and deviations
        for period in self.params['ma_periods']:
            # Calculate MA
            ma = close.rolling(window=period).mean()
            
            # Deviation from MA (as percentage)
            df[f'close_ma{period}_ratio'] = close / ma - 1
            
            # Distance from MA in standard deviations
            std = close.rolling(window=period).std()
            df[f'ma{period}_z_score'] = (close - ma) / std
        
        # Return-based features
        
        # Returns over different periods
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}d'] = close.pct_change(period)
        
        # Volatility
        vol_period = self.params['volatility_period']
        df['volatility'] = close.pct_change().rolling(window=vol_period).std()
        
        # RSI
        rsi_period = self.params['rsi_period']
        delta = close.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        avg_gain = up.rolling(window=rsi_period).mean()
        avg_loss = abs(down.rolling(window=rsi_period).mean())
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume-based features if volume data is available
        if 'volume' in self.data.columns:
            volume = self.data['volume']
            
            # Volume moving average ratios
            for period in [5, 10, 20]:
                vol_ma = volume.rolling(window=period).mean()
                df[f'volume_ma{period}_ratio'] = volume / vol_ma - 1
            
            # On-balance volume
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            df['obv_change'] = obv.diff(5)
        
        return df
    
    def on_start(self) -> None:
        """Called when the backtest starts."""
        print(f"Starting ML strategy backtest with {self.params['prediction_horizon']}-bar prediction horizon")
        print(f"Training on {self.params['train_size']*100:.0f}% of data")
    
    def on_finish(self) -> None:
        """Called when the backtest finishes."""
        print(f"Finished ML strategy backtest with {len(self.signals)} signals generated")
        
        # For RF, we can print feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.features.columns
            
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            
            print("\nFeature importance:")
            for i, idx in enumerate(indices[:10]):  # Print top 10
                print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


class LSTMStrategy(Strategy):
    """
    This is a placeholder for a more complex LSTM-based strategy.
    In a real implementation, you would use TensorFlow or PyTorch
    to implement an LSTM model for sequence prediction.
    """
    
    def __init__(self):
        super().__init__()
    
    def init(self) -> None:
        # Placeholder - in real implementation, would prepare data and train LSTM
        pass
    
    def next(self) -> None:
        # Placeholder - in real implementation, would use LSTM predictions
        pass 