"""
AI-Driven Backtesting System.

This package provides tools for interpreting natural language strategy descriptions,
generating executable code, running backtests, and reporting results.
"""

__version__ = '0.1.0'

from ai_driven.runner import run_backtest, BacktestResult
from ai_driven.parser import parse_strategy, StrategySpec 