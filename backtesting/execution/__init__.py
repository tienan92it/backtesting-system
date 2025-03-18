"""
Execution handling module for order execution.

This module provides execution handler implementations for
order execution in a trading system.
"""

from backtesting.execution.execution_handler import ExecutionHandler
from backtesting.execution.simulated_execution import SimulatedExecutionHandler

__all__ = ['ExecutionHandler', 'SimulatedExecutionHandler'] 