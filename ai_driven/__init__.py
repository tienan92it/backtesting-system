"""
AI-Driven Backtesting System.

This package provides tools for interpreting natural language strategy descriptions,
generating executable code, running backtests, and reporting results.
"""

__version__ = '0.1.0'

from ai_driven.runner import run_backtest, BacktestResult
from ai_driven.parser import parse_strategy, StrategySpec
from ai_driven.code_generator import generate_code
from ai_driven.workflow import run_ai_backtest_workflow
from ai_driven.report import build_report

__all__ = [
    'run_backtest',
    'BacktestResult',
    'parse_strategy',
    'StrategySpec',
    'generate_code',
    'build_report',
    'run_ai_backtest_workflow'
] 