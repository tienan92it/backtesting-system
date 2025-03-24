"""
Test script for the Strategy Parser Module.

This script tests the functionality of the parse_strategy function
which uses a Large Language Model to extract structured information 
from natural language strategy descriptions.
"""

import argparse
import json
import logging
import os
from ai_driven.parser import parse_strategy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_parser(description: str, api_key: str = None) -> None:
    """
    Test the LLM-based parser with a given strategy description.
    
    Parameters:
    -----------
    description : str
        The trading strategy description to parse.
    api_key : str, optional
        The OpenAI API key to use. If not provided, it will attempt to use
        the OPENAI_API_KEY environment variable.
    """
    print(f"\n===== Testing Strategy Parser =====")
    print(f"Description: {description}")
    
    # Parse the strategy using LLM
    try:
        strategy_spec = parse_strategy(description, debug=True, api_key=api_key)
        
        # Print the parsed strategy structure
        print("\nParsed Strategy Spec:")
        print(json.dumps(strategy_spec.to_dict(), indent=2))
        
        # Print summary
        print("\nSummary:")
        print(f"- Strategy name: {strategy_spec.name}")
        if hasattr(strategy_spec, 'is_valid'):
            print(f"- Is valid: {strategy_spec.is_valid}")
        if hasattr(strategy_spec, 'feedback') and strategy_spec.feedback:
            print(f"- Feedback: {strategy_spec.feedback}")
        
        print(f"- Indicators found: {len(strategy_spec.indicators)}")
        for ind in strategy_spec.indicators:
            params_str = ", ".join([f"{k}={v}" for k, v in ind.get('parameters', {}).items()])
            print(f"  * {ind['type']}({params_str or 'N/A'})")
        
        print(f"- Entry rules found: {len(strategy_spec.entry_rules)}")
        for rule in strategy_spec.entry_rules:
            print(f"  * {rule.get('description', 'No description')}")
        
        print(f"- Exit rules found: {len(strategy_spec.exit_rules)}")
        for rule in strategy_spec.exit_rules:
            print(f"  * {rule.get('description', 'No description')}")
        
        if strategy_spec.risk_management:
            print("- Risk management:")
            for key, value in strategy_spec.risk_management.items():
                print(f"  * {key}: {value}")
    
    except Exception as e:
        logger.error(f"Error testing parser: {e}")

def main():
    """Run the parser test script."""
    parser = argparse.ArgumentParser(description='Test the Strategy Parser Module')
    parser.add_argument('--description', '-d', type=str, help='Strategy description to parse')
    parser.add_argument('--api-key', '-k', type=str, help='OpenAI API key')
    args = parser.parse_args()
    
    # Use provided API key or environment variable
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No OpenAI API key provided. Set it with --api-key or OPENAI_API_KEY environment variable.")
    
    # Use provided description or default to test cases
    if args.description:
        test_parser(args.description, api_key)
    else:
        # Test cases
        test_cases = [
            "Buy when the 10-day SMA crosses above the 30-day SMA, and sell when RSI(14) goes above 70. Use a 2% stop loss.",
            "Strategy name: Golden Cross Strategy. Buy when 50-day EMA crosses above 200-day EMA. Sell when price falls 5% from peak.",
            "Buy when RSI(14) is below 30, indicating oversold conditions. Sell when RSI(14) goes above 70, indicating overbought conditions.",
            "When MACD crosses above the signal line, buy. When it crosses below, sell.",
            "When the price breaks above the upper Bollinger Band, sell. When it breaks below the lower Bollinger Band, buy.",
            "Strategy name: Triple Screen System. Buy when weekly trend is up, daily MACD histogram is positive and RSI(14) is above 50. Sell when weekly trend turns down or RSI(14) is above 70. Use 2% stop loss and 8% trailing stop."
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n===== Test Case {i+1} =====")
            test_parser(test_case, api_key)

if __name__ == "__main__":
    main() 