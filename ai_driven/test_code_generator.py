#!/usr/bin/env python3
"""
Test script for the Strategy Code Generator Module.

This script tests the strategy code generator by parsing a strategy description
and generating corresponding Python code.
"""

import os
import argparse
import logging
from ai_driven.parser import parse_strategy
from ai_driven.code_generator import generate_code

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the generated code directory exists
os.makedirs(os.path.join('ai_driven', 'generated'), exist_ok=True)

def main():
    """
    Main function to test the strategy code generator.
    """
    parser = argparse.ArgumentParser(description='Test the Strategy Code Generator Module')
    parser.add_argument('-d', '--description', type=str, required=True,
                        help='The strategy description to parse and generate code for')
    parser.add_argument('--api-key', type=str, help='API key for LLM service')
    
    args = parser.parse_args()
    
    print("\n===== Testing Code Generator =====")
    print(f"Description: {args.description}")
    
    # Parse the strategy description
    strategy_spec = parse_strategy(args.description, api_key=args.api_key)
    
    # Generate code for the strategy
    generated_code = generate_code(strategy_spec, api_key=args.api_key)
    
    # Display the generated code
    print("\nGenerated Strategy Code:")
    print("-" * 80)
    print(generated_code)
    print("-" * 80)
    
    # Save the generated code to a file
    sanitized_name = strategy_spec.name.lower().replace(' ', '_')
    output_file = os.path.join('ai_driven', 'generated', f"{sanitized_name}_strategy.py")
    with open(output_file, 'w') as f:
        f.write(generated_code)
    
    print(f"\nStrategy code saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main() 