"""
Test script for the AI-driven backtesting workflow.
"""

import os
import logging
import argparse
from typing import Dict, Any
from ai_driven.workflow import run_ai_backtest_workflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default test strategy
DEFAULT_STRATEGY = """
Create a strategy that buys Bitcoin when the 50-day moving average crosses above 
the 200-day moving average, and sells when the 50-day moving average crosses below 
the 200-day moving average. Use 2% of the portfolio for each trade.
"""

# Default test configuration
DEFAULT_CONFIG = {
    "symbol": "BTC/USDT",
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "initial_capital": 10000.0,
    "timeframe": "1d",
    "commission": 0.001,
    "exchange": "binance",
    "use_sample_data": True
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test the AI-driven backtesting workflow')
    
    parser.add_argument(
        '--strategy',
        type=str,
        default=DEFAULT_STRATEGY,
        help='Natural language strategy description'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default=DEFAULT_CONFIG['symbol'],
        help='Trading symbol'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=DEFAULT_CONFIG['start_date'],
        help='Start date for the backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=DEFAULT_CONFIG['end_date'],
        help='End date for the backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=DEFAULT_CONFIG['initial_capital'],
        help='Initial capital for the backtest'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default=DEFAULT_CONFIG['timeframe'],
        help='Candle timeframe (e.g., 1d, 4h, 1h)'
    )
    
    parser.add_argument(
        '--use-sample-data',
        action='store_true',
        default=True,
        help='Use generated sample data instead of real data'
    )
    
    parser.add_argument(
        '--save-artifacts',
        action='store_true',
        default=True,
        help='Save generated strategy code and report'
    )
    
    parser.add_argument(
        '--artifacts-dir',
        type=str,
        default='ai_driven/generated',
        help='Directory to save generated artifacts'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (defaults to environment variable OPENAI_API_KEY)'
    )
    
    return parser.parse_args()

def main():
    """Run the test workflow."""
    args = parse_args()
    
    # Prepare configuration
    config = {
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_capital": args.capital,
        "timeframe": args.timeframe,
        "commission": DEFAULT_CONFIG['commission'],
        "exchange": DEFAULT_CONFIG['exchange'],
        "use_sample_data": args.use_sample_data
    }
    
    logger.info(f"Testing AI-driven backtesting workflow with config: {config}")
    
    try:
        # Run the workflow
        result, report_html = run_ai_backtest_workflow(
            strategy_description=args.strategy,
            backtest_config=config,
            save_artifacts=args.save_artifacts,
            artifacts_dir=args.artifacts_dir,
            api_key=args.api_key
        )
        
        # Print results summary
        logger.info("Backtest completed successfully!")
        logger.info(f"Total Return: {result.metrics.get('total_return', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.2f}%")
        logger.info(f"Total Trades: {len(result.trades)}")
        
        # Open report in browser if saved
        if args.save_artifacts:
            import webbrowser
            import glob
            
            # Find the most recent report file
            report_files = sorted(glob.glob(f"{args.artifacts_dir}/report_*.html"))
            if report_files:
                report_path = report_files[-1]
                report_url = f"file://{os.path.abspath(report_path)}"
                logger.info(f"Opening report in browser: {report_url}")
                webbrowser.open(report_url)
    
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 