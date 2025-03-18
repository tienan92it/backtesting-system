import pandas as pd
import numpy as np
import time
import ccxt
import os
from datetime import datetime
from typing import Optional, Dict, Any, Union, List
from tqdm import tqdm
from collections import defaultdict

from backtesting.data.data_handler import DataHandler


class CCXTDataHandler(DataHandler):
    """
    Data handler that uses CCXT library to fetch data from various exchanges.
    This enables support for multiple exchanges with a unified API.
    """
    
    # Default timeframes supported by most exchanges
    DEFAULT_TIMEFRAMES = [
        '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w', '1M'
    ]
    
    def __init__(self, exchange_id: str = 'binance', api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None, use_cache: bool = True, 
                 cache_dir: str = 'data'):
        """
        Initialize CCXTDataHandler.
        
        Parameters:
        -----------
        exchange_id : str
            CCXT exchange ID (e.g., 'binance', 'coinbase', 'kraken').
            See https://github.com/ccxt/ccxt/wiki/Exchange-Markets for supported exchanges.
        api_key : str, optional
            API key for the exchange.
        api_secret : str, optional
            API secret for the exchange.
        use_cache : bool, optional
            Whether to cache data to disk for reuse.
        cache_dir : str, optional
            Directory to cache data in.
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_cache = use_cache
        self.cache_dir = os.path.join(cache_dir, exchange_id)
        
        # Initialize data structures for backtesting
        self.data = {}
        self.symbols = []
        self.current_index = 0
        self.current_bar = {}
        self.current_datetime = None
        
        # Initialize CCXT exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True  # Automatically respect rate limits
        })
        
        # Create cache directory for this exchange
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_symbols(self) -> list:
        """
        Get list of available trading symbols from the exchange.
        
        Returns:
        --------
        list
            List of trading symbols.
        """
        self.exchange.load_markets()
        return list(self.exchange.markets.keys())
    
    def get_timeframes(self) -> list:
        """
        Get list of supported timeframes for the exchange.
        
        Returns:
        --------
        list
            List of timeframes supported by the exchange.
        """
        if hasattr(self.exchange, 'timeframes') and self.exchange.timeframes:
            return list(self.exchange.timeframes.keys())
        return self.DEFAULT_TIMEFRAMES
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from the exchange.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., 'BTC/USDT').
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
            DataFrame with columns: open, high, low, close, volume
            Indexed by open_time (as datetime).
        """
        # Format symbol for CCXT if needed (CCXT often uses '/' format like BTC/USDT)
        if '/' not in symbol and not self.exchange.markets:
            self.exchange.load_markets()
        
        # For Binance and some others, symbol might be BTCUSDT, so we try to adapt
        if '/' not in symbol:
            for market_symbol in self.exchange.markets:
                # Strip out / to compare with provided symbol
                if market_symbol.replace('/', '') == symbol:
                    symbol = market_symbol
                    break
        
        # Convert start/end times to timestamps (milliseconds)
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
            
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        # Check if data already exists in cache
        if self.use_cache:
            cached_data = self.load_data(symbol.replace('/', ''), timeframe, start_time, end_time, directory=self.cache_dir)
            if cached_data is not None:
                return cached_data
        
        # CCXT has a fetch_ohlcv method that handles pagination internally
        # but we'll implement our own to have more control and better error handling
        
        # Calculate how many candles we need to fetch based on timeframe
        ms_per_candle = self._get_timeframe_ms(timeframe)
        total_time_ms = end_ts - start_ts
        estimated_candles = total_time_ms // ms_per_candle
        
        # Some exchanges have a limit on how many candles can be fetched at once
        max_limit = getattr(self.exchange, 'ohlcv_limit', 1000)
        if limit and limit < max_limit:
            max_limit = limit
            
        # Fetch data in chunks
        all_candles = []
        current_start = start_ts
        
        with tqdm(total=estimated_candles, desc=f"Fetching {symbol} {timeframe} from {self.exchange_id}") as pbar:
            while current_start < end_ts:
                try:
                    candles = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=current_start,
                        limit=max_limit
                    )
                    
                    if not candles:
                        break
                        
                    all_candles.extend(candles)
                    pbar.update(len(candles))
                    
                    # Update the start time for the next iteration
                    current_start = candles[-1][0] + 1
                    
                except ccxt.NetworkError as e:
                    print(f"Network error: {e}. Retrying in 10 seconds...")
                    time.sleep(10)
                    continue
                except ccxt.ExchangeError as e:
                    print(f"Exchange error: {e}")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    break
        
        if not all_candles:
            raise ValueError(f"No data returned for {symbol} {timeframe} from {start_time} to {end_time}")
        
        # Convert to DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(all_candles, columns=columns)
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        # Make sure DataFrame only contains data within the requested range
        df = df[(df.index >= start_time) & (df.index <= end_time)]
        
        # Clean the data
        df = self.clean_data(df)
        
        # Cache the data
        if self.use_cache:
            # Use symbol without '/' for filenames
            self.save_data(df, symbol.replace('/', ''), timeframe, directory=self.cache_dir)
            
        return df
    
    def _get_timeframe_ms(self, timeframe: str) -> int:
        """
        Convert timeframe string to milliseconds.
        
        Parameters:
        -----------
        timeframe : str
            Timeframe string (e.g., '1h', '1d').
            
        Returns:
        --------
        int
            Milliseconds represented by the timeframe.
        """
        # Parse the timeframe string
        amount = int(timeframe[:-1])
        unit = timeframe[-1]
        
        # Convert to milliseconds
        if unit == 'm':
            return amount * 60 * 1000
        elif unit == 'h':
            return amount * 60 * 60 * 1000
        elif unit == 'd':
            return amount * 24 * 60 * 60 * 1000
        elif unit == 'w':
            return amount * 7 * 24 * 60 * 60 * 1000
        elif unit == 'M':
            return amount * 30 * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")
    
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
