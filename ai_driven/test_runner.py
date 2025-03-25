"""
Direct strategy testing script for diagnosing backtesting issues.

This script bypasses the AI generation steps and directly runs a test strategy
with detailed logging to diagnose execution issues.
"""

import logging
import argparse
from typing import Dict, Any
import pandas as pd

from backtesting.data.binance_data import BinanceDataHandler
from backtesting.data.ccxt_data import CCXTDataHandler
from backtesting.engine.backtester import Backtester
from backtesting.portfolio.portfolio import Portfolio
from backtesting.execution.simulated_execution import SimulatedExecutionHandler
from ai_driven.runner import run_backtest, BacktestResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test strategy code with detailed logging
DEBUG_STRATEGY_CODE = '''
import logging
import pandas as pd
import numpy as np
from backtesting.strategy.base import Strategy

class DiagnosticStrategy(Strategy):
    """
    Diagnostic strategy with detailed logging to diagnose trade execution issues.
    Implements a simple moving average crossover strategy.
    """
    
    def __init__(self, short_window=5, long_window=20, position_size_pct=10):
        super().__init__()
        self.params = {
            'short_window': short_window,
            'long_window': long_window,
            'position_size_pct': position_size_pct  # Use 10% of portfolio for each trade
        }
        self.ma_short = None
        self.ma_long = None
        self.logger = logging.getLogger(__name__)
        self.events_processed = 0
        self.signals_generated = 0
        self.crosses_detected = 0
    
    def init(self):
        # Calculate moving averages
        short_window = self.params['short_window']
        long_window = self.params['long_window']
        
        self.logger.info(f"Data shape: {self.data.shape}")
        self.logger.info(f"Data columns: {self.data.columns}")
        self.logger.info(f"Data index type: {type(self.data.index)}")
        self.logger.info(f"First few rows: {self.data.head()}")
        
        # Check for NaN values
        nan_count = self.data.isna().sum().sum()
        self.logger.info(f"Total NaN values in data: {nan_count}")
        
        # Calculate moving averages
        self.ma_short = self.data['close'].rolling(window=short_window).mean()
        self.ma_long = self.data['close'].rolling(window=long_window).mean()
        
        # Check NaN values in indicators
        self.logger.info(f"NaN values in short MA: {self.ma_short.isna().sum()}")
        self.logger.info(f"NaN values in long MA: {self.ma_long.isna().sum()}")
        
        # Log first few values
        self.logger.info(f"Short MA (first 5): {self.ma_short.head()}")
        self.logger.info(f"Long MA (first 5): {self.ma_long.head()}")
        
        # Log indicator values near the end of the warmup period
        start_idx = max(0, long_window - 5)
        end_idx = long_window + 5
        self.logger.info(f"Short MA (idx {start_idx}-{end_idx}): {self.ma_short.iloc[start_idx:end_idx]}")
        self.logger.info(f"Long MA (idx {start_idx}-{end_idx}): {self.ma_long.iloc[start_idx:end_idx]}")
    
    def next(self):
        # Log every 20 bars or on the first few bars
        log_detailed = (self.current_index % 20 == 0) or (self.current_index < 10) 
        
        # Skip if not enough data
        if self.current_index < self.params['long_window']:
            if log_detailed:
                self.logger.info(f"Bar {self.current_index}: Still in warmup period (need {self.params['long_window']} bars)")
            return
            
        # Get current values
        curr_short = self.ma_short.iloc[self.current_index]
        curr_long = self.ma_long.iloc[self.current_index]
        
        # Get previous values
        prev_short = self.ma_short.iloc[self.current_index - 1]
        prev_long = self.ma_long.iloc[self.current_index - 1]
        
        # Check for NaN values
        if pd.isna(curr_short) or pd.isna(curr_long) or pd.isna(prev_short) or pd.isna(prev_long):
            self.logger.warning(f"Bar {self.current_index}: NaN values detected in indicators")
            return
        
        if log_detailed:
            self.logger.info(f"Bar {self.current_index}: Date={self.data.index[self.current_index]}, "
                          f"Close={self.data['close'].iloc[self.current_index]:.2f}, "
                          f"Short MA={curr_short:.2f}, Long MA={curr_long:.2f}, "
                          f"Position={self.position}, Size={self.position_size:.4f}, "
                          f"Portfolio=${self.portfolio_value:.2f}")
        
        # Check for golden cross (short MA crosses above long MA)
        golden_cross = prev_short <= prev_long and curr_short > curr_long
        death_cross = prev_short >= prev_long and curr_short < curr_long
        
        if golden_cross:
            self.crosses_detected += 1
            self.logger.info(f"GOLDEN CROSS detected at bar {self.current_index}, "
                          f"date={self.data.index[self.current_index]}, "
                          f"Short MA={curr_short:.2f}, Long MA={curr_long:.2f}")
            
            # Buy only if we don't have a position
            if self.position <= 0:
                # Calculate position size as percentage of portfolio
                pct = self.params['position_size_pct']
                
                # Place the buy order
                self.logger.info(f"Sending BUY signal at bar {self.current_index}, "
                              f"date={self.data.index[self.current_index]}, "
                              f"price={self.data['close'].iloc[self.current_index]:.2f}, "
                              f"percent={pct}")
                self.buy(percent=pct)
                self.signals_generated += 1
            else:
                self.logger.info(f"Already have a long position, not buying")
        
        # Check for death cross (short MA crosses below long MA)
        elif death_cross:
            self.crosses_detected += 1
            self.logger.info(f"DEATH CROSS detected at bar {self.current_index}, "
                          f"date={self.data.index[self.current_index]}, "
                          f"Short MA={curr_short:.2f}, Long MA={curr_long:.2f}")
            
            # Sell only if we have a position
            if self.position > 0:
                self.logger.info(f"Sending SELL signal at bar {self.current_index}, "
                              f"date={self.data.index[self.current_index]}, "
                              f"price={self.data['close'].iloc[self.current_index]:.2f}, "
                              f"position size={self.position_size:.4f}")
                self.sell()  # Sell entire position
                self.signals_generated += 1
            else:
                self.logger.info(f"No long position to sell")
    
    def on_start(self):
        self.logger.info("Strategy on_start called")
        self.logger.info(f"Parameters: {self.params}")
        self.logger.info(f"Initial position: {self.position}")
        self.logger.info(f"Initial portfolio value: ${self.portfolio_value:.2f}")
    
    def on_finish(self):
        self.logger.info("Strategy on_finish called")
        self.logger.info(f"Final position: {self.position}")
        self.logger.info(f"Final portfolio value: ${self.portfolio_value:.2f}")
        self.logger.info(f"Events processed: {self.events_processed}")
        self.logger.info(f"Crosses detected: {self.crosses_detected}")
        self.logger.info(f"Signals generated: {self.signals_generated}")
        
        # Check why no trades happened if that's the case
        if self.signals_generated == 0:
            self.logger.warning("No trade signals were generated during the backtest")
        elif self.crosses_detected == 0:
            self.logger.warning("No MA crosses were detected during the backtest")
    
    def on_data(self, event):
        self.events_processed += 1
        
        # Update current_index
        if self.data_handler is not None:
            self.current_index = self.data_handler.current_index
        
        # Log event data occasionally
        if self.events_processed % 50 == 0:
            self.logger.info(f"Processed {self.events_processed} events, "
                          f"current index: {self.current_index}, "
                          f"signals: {self.signals_generated}")
        
        # Call next method
        try:
            self.next()
        except Exception as e:
            self.logger.error(f"Error in next(): {str(e)}", exc_info=True)
'''

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test backtesting with diagnostic strategy')
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading symbol'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2022-01-01',
        help='Start date for the backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2022-12-31',
        help='End date for the backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Initial capital for the backtest'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1d',
        help='Candle timeframe (e.g., 1d, 4h, 1h)'
    )
    
    parser.add_argument(
        '--use-real-data',
        action='store_true',
        help='Use real market data instead of sample data'
    )
    
    parser.add_argument(
        '--exchange',
        type=str,
        default='binance',
        choices=['binance', 'ccxt'],
        help='Exchange to use for market data'
    )
    
    parser.add_argument(
        '--save-artifacts',
        action='store_true',
        help='Save generated artifacts'
    )
    
    return parser.parse_args()

def main():
    """Run the direct testing workflow."""
    args = parse_args()
    
    # Prepare configuration
    config = {
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_capital": args.capital,
        "timeframe": args.timeframe,
        "commission": 0.001,
        "exchange": args.exchange,
        "use_sample_data": not args.use_real_data,
        "strategy_params": {
            "short_window": 5,
            "long_window": 20,
            "position_size_pct": 10
        }
    }
    
    logger.info(f"Testing backtesting with diagnostic strategy")
    logger.info(f"Configuration: {config}")
    logger.info(f"Using {'REAL' if args.use_real_data else 'SAMPLE'} market data")
    
    try:
        # Run the backtest
        result = run_backtest(DEBUG_STRATEGY_CODE, config)
        
        # Print results summary
        logger.info("Backtest completed successfully!")
        
        if result.metrics:
            logger.info(f"Total Return: {result.metrics.get('total_return', 0):.2f}%")
            logger.info(f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.2f}%")
        else:
            logger.warning("No metrics were calculated in the backtest result")
            
        if result.trades:
            logger.info(f"Total Trades: {len(result.trades)}")
            for i, trade in enumerate(result.trades[:5]):
                logger.info(f"Trade {i+1}: {trade}")
        else:
            logger.warning("No trades were executed during the backtest")
            
        # Check equity curve
        if result.equity_curve is not None and len(result.equity_curve) > 0:
            logger.info(f"Equity curve has {len(result.equity_curve)} data points")
        else:
            logger.warning("Equity curve is empty")
            
        # Save results if requested
        if args.save_artifacts:
            import json
            import os
            from datetime import datetime
            
            # Create artifacts directory if it doesn't exist
            artifacts_dir = 'ai_driven/generated'
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Save strategy code
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            code_filename = f"{artifacts_dir}/diagnostic_strategy_{timestamp}.py"
            with open(code_filename, 'w') as f:
                f.write(DEBUG_STRATEGY_CODE)
            logger.info(f"Strategy code saved to {code_filename}")
            
            # Save results summary
            results_filename = f"{artifacts_dir}/results_summary_{timestamp}.json"
            results_summary = {
                'total_return': result.metrics.get('total_return', 0),
                'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
                'max_drawdown': result.metrics.get('max_drawdown', 0),
                'trades_count': len(result.trades),
                'trades': result.trades[:10] if result.trades else []
            }
            with open(results_filename, 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)
            logger.info(f"Results summary saved to {results_filename}")
    
    except Exception as e:
        logger.error(f"Error during backtest: {e}", exc_info=True)


if __name__ == "__main__":
    main() 