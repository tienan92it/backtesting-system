import pytest
import pandas as pd
from datetime import datetime, timedelta
import os
import shutil

from backtesting.data import DataHandler, BinanceDataHandler, CCXTDataHandler


class TestDataHandler:
    """Test the data handler functionality."""
    
    # Test dates (use a small date range for quick testing)
    end_time = datetime(2023, 1, 1)
    start_time = end_time - timedelta(days=7)
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for tests."""
        cache_dir = "test_cache"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        # Clean up after tests
        shutil.rmtree(cache_dir)
    
    def test_binance_data_handler_initialization(self):
        """Test that BinanceDataHandler initializes correctly."""
        handler = BinanceDataHandler()
        assert isinstance(handler, DataHandler)
        assert handler.BASE_URL == "https://api.binance.com/api/v3"
    
    def test_binance_timeframes(self):
        """Test that BinanceDataHandler returns valid timeframes."""
        handler = BinanceDataHandler()
        timeframes = handler.get_timeframes()
        assert isinstance(timeframes, list)
        assert len(timeframes) > 0
        assert "1h" in timeframes
        assert "1d" in timeframes
    
    def test_ccxt_data_handler_initialization(self):
        """Test that CCXTDataHandler initializes correctly."""
        handler = CCXTDataHandler(exchange_id="binance")
        assert isinstance(handler, DataHandler)
        assert handler.exchange_id == "binance"
    
    def test_clean_data(self):
        """Test the clean_data method."""
        handler = BinanceDataHandler()
        
        # Create a test DataFrame with string data
        data = {
            'open': ['100.5', '101.2', '102.3'],
            'high': ['105.5', '106.2', '107.3'],
            'low': ['99.5', '100.2', '101.3'],
            'close': ['104.5', '105.2', '106.3'],
            'volume': ['1000', '1100', '1200']
        }
        index = pd.date_range(start=self.start_time, periods=3, freq='D')
        df = pd.DataFrame(data, index=index)
        
        # Clean the data
        cleaned_df = handler.clean_data(df)
        
        # Check that numeric columns were converted to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert cleaned_df[col].dtype == float
    
    def test_save_and_load_data(self, temp_cache_dir):
        """Test saving and loading data from cache."""
        handler = BinanceDataHandler(use_cache=True, cache_dir=temp_cache_dir)
        
        # Create a test DataFrame
        data = {
            'open': [100.5, 101.2, 102.3],
            'high': [105.5, 106.2, 107.3],
            'low': [99.5, 100.2, 101.3],
            'close': [104.5, 105.2, 106.3],
            'volume': [1000, 1100, 1200]
        }
        index = pd.date_range(start=self.start_time, periods=3, freq='D')
        df = pd.DataFrame(data, index=index)
        
        # Save the data
        symbol = "BTCUSDT"
        timeframe = "1d"
        filepath = handler.save_data(df, symbol, timeframe, directory=temp_cache_dir)
        
        # Check that the file was created
        assert os.path.exists(filepath)
        
        # Load the data
        loaded_df = handler.load_data(
            symbol, timeframe, self.start_time, self.end_time, directory=temp_cache_dir
        )
        
        # Check that the loaded data is the same as the original
        pd.testing.assert_frame_equal(loaded_df, df)
    
    @pytest.mark.skipif(not os.environ.get("RUN_ONLINE_TESTS"), 
                       reason="Set RUN_ONLINE_TESTS environment variable to run online tests")
    def test_binance_get_historical_data(self, temp_cache_dir):
        """Test fetching historical data from Binance (requires internet connection)."""
        handler = BinanceDataHandler(use_cache=True, cache_dir=temp_cache_dir)
        
        # Fetch historical data
        df = handler.get_historical_data(
            symbol="BTCUSDT",
            timeframe="1d",
            start_time=self.start_time,
            end_time=self.end_time
        )
        
        # Check that the data is valid
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns 