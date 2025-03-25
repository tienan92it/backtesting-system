"""
Test script for the fixed strategy implementation.
"""

import logging
import argparse
import os
from datetime import datetime
import json

from ai_driven.runner import run_backtest
from ai_driven.strategy_bugfix import get_strategy_code

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test fixed strategy implementation')
    
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
        default=True,
        help='Save generated artifacts'
    )
    
    return parser.parse_args()

def main():
    """Run the fixed strategy test."""
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
    
    logger.info(f"Testing fixed strategy implementation")
    logger.info(f"Configuration: {config}")
    logger.info(f"Using {'REAL' if args.use_real_data else 'SAMPLE'} market data")
    
    try:
        # Get the strategy code
        strategy_code = get_strategy_code()
        
        # Run the backtest
        result = run_backtest(strategy_code, config)
        
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
            if isinstance(result.equity_curve, list):
                first_val = result.equity_curve[0]['portfolio_value']
                last_val = result.equity_curve[-1]['portfolio_value']
                logger.info(f"Starting portfolio value: ${first_val:.2f}")
                logger.info(f"Final portfolio value: ${last_val:.2f}")
                logger.info(f"Simple return: {((last_val / first_val) - 1) * 100:.2f}%")
        else:
            logger.warning("Equity curve is empty")
            
        # Save results if requested
        if args.save_artifacts:
            # Create artifacts directory if it doesn't exist
            artifacts_dir = 'ai_driven/generated'
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Save strategy code
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            code_filename = f"{artifacts_dir}/fixed_strategy_{timestamp}.py"
            with open(code_filename, 'w') as f:
                f.write(strategy_code)
            logger.info(f"Strategy code saved to {code_filename}")
            
            # Save results summary
            results_filename = f"{artifacts_dir}/fixed_results_{timestamp}.json"
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