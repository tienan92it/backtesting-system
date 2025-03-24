"""
Test script for the AI-Driven Backtesting API.

This script makes a request to the API and processes the response.
"""

import os
import sys
import json
import argparse
import requests
from urllib.parse import urljoin

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Test the AI-Driven Backtesting API')
    
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:8000',
        help='Base URL of the API'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default='Buy when the 50-day moving average crosses above the 200-day moving average, and sell when the 50-day moving average crosses below the 200-day moving average.',
        help='Strategy description'
    )
    
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
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2022-12-31',
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='backtest_report.html',
        help='File to save the HTML report to'
    )
    
    parser.add_argument(
        '--save-response',
        type=str,
        default=None,
        help='File to save the full JSON response to'
    )
    
    return parser.parse_args()

def main():
    """
    Test the API.
    """
    args = parse_args()
    
    # Prepare the request
    endpoint = urljoin(args.url, 'backtest')
    
    payload = {
        'strategy': args.strategy,
        'symbol': args.symbol,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'use_sample_data': True
    }
    
    print(f"Making request to {endpoint}")
    print(f"Strategy: {args.strategy[:50]}...")
    
    try:
        # Make the request
        response = requests.post(endpoint, json=payload)
        
        # Check the response
        if response.status_code == 200:
            result = response.json()
            
            # Print the metrics
            print("\nBacktest Results:")
            print(f"Status: {result.get('status')}")
            print(f"Message: {result.get('message')}")
            
            metrics = result.get('metrics', {})
            if metrics:
                print("\nPerformance Metrics:")
                print(f"Total Return: {metrics.get('total_return', 0):.2f}%")
                print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
                print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
                print(f"Total Trades: {metrics.get('total_trades', 0)}")
            
            # Save the HTML report
            report_html = result.get('report_html')
            if report_html and args.output:
                with open(args.output, 'w') as f:
                    f.write(report_html)
                print(f"\nReport saved to {args.output}")
            
            # Save the full response if requested
            if args.save_response:
                with open(args.save_response, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Full response saved to {args.save_response}")
            
            return 0
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return 1
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 