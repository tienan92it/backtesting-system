"""
Test script for the Strategy Parser Module.

This script tests the functionality of the parse_strategy function
and the RuleBasedParser class.
"""

import argparse
import json
from ai_driven.parser import parse_strategy

def main():
    """Run test scenarios for the strategy parser."""
    parser = argparse.ArgumentParser(description='Test the Strategy Parser Module')
    parser.add_argument('--description', '-d', type=str, help='Strategy description to parse')
    args = parser.parse_args()
    
    # Use provided description or default to test cases
    if args.description:
        test_descriptions = [args.description]
    else:
        test_descriptions = [
            "Buy when the 10-day SMA crosses above the 30-day SMA, and sell when RSI(14) goes above 70. Use a 2% stop loss.",
            "Strategy name: Golden Cross Strategy. Buy when 50-day EMA crosses above 200-day EMA. Sell when price falls 5% from peak.",
            "Buy when RSI(14) is below 30, indicating oversold conditions. Sell when RSI(14) goes above 70, indicating overbought conditions.",
            "When MACD crosses above the signal line, buy. When it crosses below, sell.",
            "When the price breaks above the upper Bollinger Band, sell. When it breaks below the lower Bollinger Band, buy."
        ]
    
    # Test each description
    for i, description in enumerate(test_descriptions):
        print(f"\n===== Test Case {i+1} =====")
        print(f"Description: {description}")
        
        # Parse the strategy
        strategy_spec = parse_strategy(description)
        
        # Print the parsed strategy structure
        print("\nParsed Strategy Spec:")
        print(json.dumps(strategy_spec.to_dict(), indent=2))
        
        # Print summary
        print("\nSummary:")
        print(f"- Strategy name: {strategy_spec.name}")
        print(f"- Indicators found: {len(strategy_spec.indicators)}")
        for ind in strategy_spec.indicators:
            params = ind.get('parameters', {})
            period = params.get('period', 'N/A')
            print(f"  * {ind['type']}({period})")
        
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

if __name__ == "__main__":
    main() 