import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Optional, Dict, Any, Union, List, Tuple
import requests
import os
from tqdm import tqdm
from collections import defaultdict

from backtesting.data.data_handler import DataHandler


class BinanceDataHandler(DataHandler):
    """
    Data handler for Binance exchange.
    Fetches historical OHLCV data from Binance API.
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    MAX_LIMIT = 1000  # Maximum number of candles per request
    
    # Mapping of timeframe strings to milliseconds
    TIMEFRAME_MAPPING = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "3d": 3 * 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
        "1M": 30 * 24 * 60 * 60 * 1000,
    }
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 use_cache: bool = True, cache_dir: str = 'data'):
        """
        Initialize BinanceDataHandler.
        
        Parameters:
        -----------
        api_key : str, optional
            Binance API key (not required for public endpoints like historical data).
        api_secret : str, optional
            Binance API secret (not required for public endpoints).
        use_cache : bool, optional
            Whether to cache data to disk for reuse.
        cache_dir : str, optional
            Directory to cache data in.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # Initialize data structures for backtesting
        self.data = {}
        self.symbols = []
        self.current_index = 0
        self.current_bar = {}
        self.current_datetime = None
        
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
    
    def get_symbols(self) -> list:
        """
        Get list of available trading symbols from Binance.
        
        Returns:
        --------
        list
            List of trading symbols.
        """
        url = f"{self.BASE_URL}/exchangeInfo"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        symbols = [symbol_info['symbol'] for symbol_info in data['symbols']]
        return symbols
    
    def get_timeframes(self) -> list:
        """
        Get list of supported timeframes.
        
        Returns:
        --------
        list
            List of timeframes supported by Binance.
        """
        return list(self.TIMEFRAME_MAPPING.keys())
    
    def _timestamp_to_datetime(self, timestamp: int) -> datetime:
        """Convert Binance timestamp (ms) to datetime."""
        return datetime.fromtimestamp(timestamp / 1000)
    
    def _datetime_to_timestamp(self, dt: Union[str, datetime]) -> int:
        """Convert datetime or string to Binance timestamp (ms)."""
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        return int(dt.timestamp() * 1000)
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance.
        
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
            DataFrame with columns: open, high, low, close, volume, etc.
            Indexed by open_time (as datetime).
        """
        # Convert start/end times to timestamps
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
            
        start_ts = self._datetime_to_timestamp(start_time)
        end_ts = self._datetime_to_timestamp(end_time)
        
        # Check if data already exists in cache
        if self.use_cache:
            cached_data = self.load_data(symbol, timeframe, start_time, end_time, directory=self.cache_dir)
            if cached_data is not None:
                return cached_data
        
        # Calculate the number of candles we need to fetch
        timeframe_ms = self.TIMEFRAME_MAPPING[timeframe]
        total_time_ms = end_ts - start_ts
        estimated_candles = total_time_ms // timeframe_ms
        
        # Split into multiple requests if needed
        all_candles = []
        current_start = start_ts
        
        with tqdm(total=estimated_candles, desc=f"Fetching {symbol} {timeframe}") as pbar:
            while current_start < end_ts:
                # Prepare request parameters
                params = {
                    "symbol": symbol,
                    "interval": timeframe,
                    "startTime": current_start,
                    "endTime": end_ts,
                    "limit": self.MAX_LIMIT  # Max number of candles per request
                }
                
                url = f"{self.BASE_URL}/klines"
                response = requests.get(url, params=params)
                
                # Handle potential errors
                if response.status_code != 200:
                    # Handle rate limiting
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 10))
                        print(f"Rate limited. Waiting {retry_after} seconds...")
                        time.sleep(retry_after)
                        continue
                    else:
                        response.raise_for_status()
                
                # Process the candles
                candles = response.json()
                all_candles.extend(candles)
                
                # Update progress bar
                pbar.update(len(candles))
                
                if not candles:
                    break
                
                # Prepare for next iteration - start from the last candle's time + 1
                current_start = candles[-1][0] + 1
                
                # Respect Binance's rate limits
                time.sleep(0.5)  # Sleep to avoid hitting rate limits
        
        # Convert to DataFrame
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        if not all_candles:
            raise ValueError(f"No data returned for {symbol} {timeframe} from {start_time} to {end_time}")
        
        df = pd.DataFrame(all_candles, columns=columns)
        
        # Convert timestamps to datetime and set as index
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df = df.set_index('open_time')
        
        # Clean data
        df = self.clean_data(df)
        
        # Cache the data
        if self.use_cache:
            self.save_data(df, symbol, timeframe, directory=self.cache_dir)
            
        return df
    
    def update_bars(self) -> bool:
        """
        Update the current bar data.
        This method advances to the next bar in the data
        and updates current_bar and current_datetime properties.
        
        Returns:
        --------
        bool
            True if there are more bars, False if we've reached the end.
        """
        # Check if we have data
        if not self.symbols or not self.data:
            return False
        
        # Check if we've reached the end of the data
        symbol = self.symbols[0]
        if self.current_index >= len(self.data[symbol]):
            return False
        
        # Update current bar for each symbol
        for symbol in self.symbols:
            if self.current_index < len(self.data[symbol]):
                self.current_bar[symbol] = self.data[symbol].iloc[self.current_index].to_dict()
                # Make sure all values are float for consistency
                for key, value in self.current_bar[symbol].items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        self.current_bar[symbol][key] = float(value)
        
        # Update current datetime
        self.current_datetime = self.data[symbol].index[self.current_index]
        
        # Increment index
        self.current_index += 1
        
        return True
    
    def get_all_bars(self) -> Dict[datetime, Dict[str, Dict[str, float]]]:
        """
        Get all bars for all symbols, organized by timestamp.
        
        Returns:
        --------
        dict
            Dictionary of {timestamp: {symbol: bar_data}}
            Where bar_data is a dict with OHLCV values.
        """
        all_bars = defaultdict(dict)
        
        # For each symbol
        for symbol in self.symbols:
            # For each bar in the data
            for timestamp, row in self.data[symbol].iterrows():
                # Add the bar data to the dictionary
                all_bars[timestamp][symbol] = row.to_dict()
        
        return all_bars
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str, directory: str = None) -> None:
        """
        Save data to cache.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data.
        symbol : str
            Trading symbol (e.g., 'BTCUSDT').
        timeframe : str
            Candle interval (e.g., '1h', '1d').
        directory : str, optional
            Directory to save data to.
        """
        # Implement your cache saving logic here
        pass
    
    def initialize(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> None:
        """
        Initialize the data handler with date range constraints.
        
        Parameters:
        -----------
        start_date : datetime, optional
            The start date to filter data.
        end_date : datetime, optional
            The end date to filter data.
        """
        # If we already have data loaded, filter it by date range
        if hasattr(self, 'data') and self.data:
            for symbol in self.symbols:
                if symbol in self.data:
                    # Filter data by date range if provided
                    if start_date is not None or end_date is not None:
                        filtered_data = self.data[symbol]
                        
                        if start_date is not None:
                            filtered_data = filtered_data[filtered_data.index >= start_date]
                        
                        if end_date is not None:
                            filtered_data = filtered_data[filtered_data.index <= end_date]
                        
                        self.data[symbol] = filtered_data
        
        # Reset current index and bar
        self.current_index = 0
        self.current_bar = {}
        self.current_datetime = None
