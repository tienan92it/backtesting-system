import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

from backtesting.data import BinanceDataHandler, CCXTDataHandler


def fetch_and_plot_binance_data():
    """
    Fetch Bitcoin data from Binance and create a simple price chart.
    """
    # Create a Binance data handler
    data_handler = BinanceDataHandler(use_cache=True, cache_dir='example_data')
    
    # Set the symbol, timeframe, and date range
    symbol = 'BTCUSDT'
    timeframe = '1d'
    end_time = datetime.now()
    start_time = end_time - timedelta(days=365)  # Last year
    
    print(f"Fetching {symbol} {timeframe} data from {start_time.date()} to {end_time.date()}...")
    
    # Fetch historical data
    df = data_handler.get_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time
    )
    
    # Print basic stats
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Price range: ${df['low'].min():.2f} to ${df['high'].max():.2f}")
    
    # Plot the price
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'])
    plt.title(f"{symbol} Price ({timeframe} candles)")
    plt.ylabel("Price (USDT)")
    plt.xlabel("Date")
    plt.grid(True)
    
    # Save the chart
    os.makedirs('example_charts', exist_ok=True)
    plt.savefig(f'example_charts/{symbol}_{timeframe}_price.png')
    plt.close()
    
    print(f"Chart saved to example_charts/{symbol}_{timeframe}_price.png")
    
    return df


def compare_exchanges():
    """
    Fetch the same data from different exchanges and compare them.
    """
    symbol = 'BTC/USDT'
    timeframe = '1d'
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)  # Last month
    
    exchanges = ['binance', 'kraken', 'coinbase']
    dataframes = {}
    
    for exchange_id in exchanges:
        try:
            print(f"Fetching data from {exchange_id}...")
            data_handler = CCXTDataHandler(
                exchange_id=exchange_id,
                use_cache=True,
                cache_dir='example_data'
            )
            
            df = data_handler.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            dataframes[exchange_id] = df
            print(f"  Got {len(df)} rows")
            
        except Exception as e:
            print(f"  Error fetching data from {exchange_id}: {e}")
    
    # Plot comparison if we have data from multiple exchanges
    if len(dataframes) >= 2:
        plt.figure(figsize=(12, 6))
        
        for exchange_id, df in dataframes.items():
            plt.plot(df.index, df['close'], label=exchange_id)
            
        plt.title(f"{symbol} Price Comparison ({timeframe} candles)")
        plt.ylabel("Price (USDT)")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(True)
        
        os.makedirs('example_charts', exist_ok=True)
        plt.savefig(f'example_charts/exchange_comparison_{timeframe}.png')
        plt.close()
        
        print(f"Comparison chart saved to example_charts/exchange_comparison_{timeframe}.png")


if __name__ == "__main__":
    print("===== Testing Binance Data Handler =====")
    bitcoin_data = fetch_and_plot_binance_data()
    
    print("\n===== Testing CCXT Data Handler with Multiple Exchanges =====")
    compare_exchanges() 