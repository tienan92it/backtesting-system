"""
Metrics module for calculating performance metrics for backtesting results.
"""

from backtesting.metrics.performance import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_drawdown_duration,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_average_trade,
    calculate_annualized_return,
    calculate_volatility,
    calculate_calmar_ratio,
    calculate_performance_metrics,
    calculate_trade_metrics
)

__all__ = [
    'calculate_returns',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_drawdown_duration',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_average_trade',
    'calculate_annualized_return',
    'calculate_volatility',
    'calculate_calmar_ratio',
    'calculate_performance_metrics',
    'calculate_trade_metrics'
]
