from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, Union


class DataHandler(ABC):
    """
    Abstract base class for handling market data.
    All data fetching implementations should extend this class.
    """

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., 'BTCUSDT').
        timeframe : str
            Candle interval (e.g., '1h', '1d').
        start_time : str or datetime
            Start time for data.
        end_time : str or datetime
            End time for data.
        limit : int, optional
            Maximum number of candles to fetch.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: open_time, open, high, low, close, volume, ...
            Indexed by open_time (as datetime).
        """
        pass
    
    @abstractmethod
    def get_symbols(self) -> list:
        """
        Get list of available trading symbols.
        
        Returns:
        --------
        list
            List of trading symbols.
        """
        pass
    
    @abstractmethod
    def get_timeframes(self) -> list:
        """
        Get list of supported timeframes.
        
        Returns:
        --------
        list
            List of timeframes (e.g., ['1m', '5m', '15m', '1h', '4h', '1d'])
        """
        pass
    
    @abstractmethod
    def update_bars(self) -> bool:
        """
        Update the current bar data.
        This method should advance to the next bar in the data
        and update current_bar and current_datetime properties.
        
        Returns:
        --------
        bool
            True if there are more bars, False if we've reached the end.
        """
        pass
    
    @abstractmethod
    def get_all_bars(self) -> Dict[datetime, Dict[str, Dict[str, float]]]:
        """
        Get all bars for all symbols, organized by timestamp.
        
        Returns:
        --------
        dict
            Dictionary of {timestamp: {symbol: bar_data}}
            Where bar_data is a dict with OHLCV values.
        """
        pass
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the data for backtesting.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw DataFrame with OHLCV data.
            
        Returns:
        --------
        pd.DataFrame
            Cleaned DataFrame ready for backtesting.
        """
        # Convert numeric strings to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Make sure DataFrame is sorted by time
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            
        # Check for missing data points
        if df.isnull().any().any():
            print(f"Warning: DataFrame contains {df.isnull().sum().sum()} missing values")
            # Fill missing values (could implement different strategies here)
            df = df.ffill()  # Forward fill: use last known value
            
        return df
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str, directory: str = 'data') -> str:
        """
        Save data to disk for future use.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to save.
        symbol : str
            Trading symbol.
        timeframe : str
            Candle interval.
        directory : str, optional
            Directory to save data to.
            
        Returns:
        --------
        str
            Path to saved file.
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Format filename
        start_date = df.index[0].strftime('%Y-%m-%d')
        end_date = df.index[-1].strftime('%Y-%m-%d')
        filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.csv"
        filepath = os.path.join(directory, filename)
        
        # Save data
        df.to_csv(filepath)
        return filepath
    
    def load_data(self, symbol: str, timeframe: str, start_time: Union[str, datetime], 
                 end_time: Union[str, datetime], directory: str = 'data') -> Optional[pd.DataFrame]:
        """
        Load data from disk if available.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol.
        timeframe : str
            Candle interval.
        start_time : str or datetime
            Start time for data.
        end_time : str or datetime
            End time for data.
        directory : str, optional
            Directory to load data from.
            
        Returns:
        --------
        pd.DataFrame or None
            DataFrame if file exists, otherwise None.
        """
        import os
        import glob
        
        # Convert start/end times to datetime objects if they're strings
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
            
        # Look for files that might contain the requested data
        pattern = os.path.join(directory, f"{symbol}_{timeframe}_*.csv")
        files = glob.glob(pattern)
        
        for file in files:
            # Extract dates from filename
            base = os.path.basename(file)
            parts = base.split('_')
            if len(parts) >= 4:
                file_start = pd.to_datetime(parts[2])
                file_end = pd.to_datetime(parts[3].replace('.csv', ''))
                
                # Check if file covers the requested date range
                if file_start <= start_time and file_end >= end_time:
                    df = pd.read_csv(file, index_col=0, parse_dates=True)
                    # Filter to requested range
                    df = df.loc[start_time:end_time]
                    return df
                    
        return None
