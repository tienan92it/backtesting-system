"""
Performance metrics calculation for backtesting results.

This module provides functions to calculate various performance metrics
for evaluating trading strategies, including:
- Sharpe ratio
- Max drawdown
- Win rate
- Profit factor
- Annualized return
- Volatility
- Calmar ratio
- Sortino ratio
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple


def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate returns from an equity curve.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Series of portfolio values over time.
    
    Returns:
    --------
    pd.Series
        Series of returns.
    """
    return equity_curve.pct_change().fillna(0)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sharpe ratio.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns.
    risk_free_rate : float, optional
        Risk-free rate, annualized. Default is 0.
    periods_per_year : int, optional
        Number of periods in a year. Default is 252 (trading days).
    
    Returns:
    --------
    float
        Sharpe ratio.
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns - risk_free_rate / periods_per_year
    
    # Calculate annualized Sharpe ratio
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()
    
    if std_excess_return == 0:
        return 0.0
    
    sharpe = mean_excess_return / std_excess_return
    sharpe_annualized = sharpe * np.sqrt(periods_per_year)
    
    return sharpe_annualized


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Sortino ratio, which only penalizes downside volatility.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns.
    risk_free_rate : float, optional
        Risk-free rate, annualized. Default is 0.
    periods_per_year : int, optional
        Number of periods in a year. Default is 252 (trading days).
    
    Returns:
    --------
    float
        Sortino ratio.
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns - risk_free_rate / periods_per_year
    
    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf  # No downside returns, perfect Sortino
    
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_deviation == 0:
        return 0.0
    
    sortino = excess_returns.mean() / downside_deviation
    sortino_annualized = sortino * np.sqrt(periods_per_year)
    
    return sortino_annualized


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate the maximum drawdown and its duration.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Series of portfolio values over time.
    
    Returns:
    --------
    tuple
        (max_drawdown, peak_date, valley_date)
    """
    if len(equity_curve) == 0:
        return 0.0, None, None
    
    # Calculate running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdown in percentage terms
    drawdown = (equity_curve - running_max) / running_max
    
    # Find the maximum drawdown
    max_drawdown = drawdown.min()
    
    # Find the peak and valley dates
    valley_idx = drawdown.idxmin()
    peak_idx = equity_curve[:valley_idx].idxmax() if valley_idx is not None else None
    
    return max_drawdown, peak_idx, valley_idx


def calculate_drawdown_duration(equity_curve: pd.Series) -> int:
    """
    Calculate the average drawdown duration in days.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Series of portfolio values over time.
    
    Returns:
    --------
    int
        Average drawdown duration in days.
    """
    if len(equity_curve) == 0:
        return 0
    
    # Calculate running maximum
    running_max = equity_curve.cummax()
    
    # Identify drawdown periods
    in_drawdown = equity_curve < running_max
    
    # Calculate drawdown durations
    drawdown_durations = []
    current_duration = 0
    
    for is_drawdown in in_drawdown:
        if is_drawdown:
            current_duration += 1
        elif current_duration > 0:
            drawdown_durations.append(current_duration)
            current_duration = 0
    
    # Add the last drawdown if we're still in one
    if current_duration > 0:
        drawdown_durations.append(current_duration)
    
    # Calculate average duration
    if drawdown_durations:
        return int(np.mean(drawdown_durations))
    else:
        return 0


def calculate_win_rate(trades: pd.DataFrame) -> float:
    """
    Calculate the win rate from a list of trades.
    
    Parameters:
    -----------
    trades : pd.DataFrame
        DataFrame of trades with 'profit' or 'pnl' column.
    
    Returns:
    --------
    float
        Win rate as a decimal (0.0 to 1.0).
    """
    if len(trades) == 0:
        return 0.0
    
    # Determine which column to use for profit
    profit_col = None
    for col in ['profit', 'pnl', 'return']:
        if col in trades.columns:
            profit_col = col
            break
    
    if profit_col is None:
        # Try to calculate profit from price and quantity
        if all(col in trades.columns for col in ['price', 'quantity', 'direction']):
            # Create a profit column based on direction, price, and quantity
            trades['profit'] = trades.apply(
                lambda row: row['price'] * row['quantity'] * (1 if row['direction'] == 'SELL' else -1),
                axis=1
            )
            profit_col = 'profit'
        else:
            return 0.0
    
    # Calculate win rate
    winning_trades = (trades[profit_col] > 0).sum()
    total_trades = len(trades)
    
    return winning_trades / total_trades if total_trades > 0 else 0.0


def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """
    Calculate the profit factor from a list of trades.
    
    Parameters:
    -----------
    trades : pd.DataFrame
        DataFrame of trades with 'profit' or 'pnl' column.
    
    Returns:
    --------
    float
        Profit factor (gross profit / gross loss).
    """
    if len(trades) == 0:
        return 0.0
    
    # Determine which column to use for profit
    profit_col = None
    for col in ['profit', 'pnl', 'return']:
        if col in trades.columns:
            profit_col = col
            break
    
    if profit_col is None:
        # Try to calculate profit from price and quantity
        if all(col in trades.columns for col in ['price', 'quantity', 'direction']):
            # Create a profit column based on direction, price, and quantity
            trades['profit'] = trades.apply(
                lambda row: row['price'] * row['quantity'] * (1 if row['direction'] == 'SELL' else -1),
                axis=1
            )
            profit_col = 'profit'
        else:
            return 0.0
    
    # Calculate gross profit and gross loss
    gross_profit = trades[trades[profit_col] > 0][profit_col].sum()
    gross_loss = abs(trades[trades[profit_col] < 0][profit_col].sum())
    
    return gross_profit / gross_loss if gross_loss != 0 else float('inf')


def calculate_average_trade(trades: pd.DataFrame) -> float:
    """
    Calculate the average profit per trade.
    
    Parameters:
    -----------
    trades : pd.DataFrame
        DataFrame of trades with 'profit' or 'pnl' column.
    
    Returns:
    --------
    float
        Average profit per trade.
    """
    if len(trades) == 0:
        return 0.0
    
    # Determine which column to use for profit
    profit_col = None
    for col in ['profit', 'pnl', 'return']:
        if col in trades.columns:
            profit_col = col
            break
    
    if profit_col is None:
        # Try to calculate profit from price and quantity
        if all(col in trades.columns for col in ['price', 'quantity', 'direction']):
            # Create a profit column based on direction, price, and quantity
            trades['profit'] = trades.apply(
                lambda row: row['price'] * row['quantity'] * (1 if row['direction'] == 'SELL' else -1),
                axis=1
            )
            profit_col = 'profit'
        else:
            return 0.0
    
    # Calculate average trade
    return trades[profit_col].mean()


def calculate_annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized return.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Series of portfolio values over time.
    periods_per_year : int, optional
        Number of periods in a year. Default is 252 (trading days).
    
    Returns:
    --------
    float
        Annualized return.
    """
    if len(equity_curve) <= 1:
        return 0.0
    
    # Calculate total return
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    
    # Calculate number of years
    n_periods = len(equity_curve) - 1
    n_years = n_periods / periods_per_year
    
    # Calculate annualized return
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    
    return annualized_return


def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized volatility.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns.
    periods_per_year : int, optional
        Number of periods in a year. Default is 252 (trading days).
    
    Returns:
    --------
    float
        Annualized volatility.
    """
    if len(returns) <= 1:
        return 0.0
    
    # Calculate annualized volatility
    volatility = returns.std() * np.sqrt(periods_per_year)
    
    return volatility


def calculate_calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the Calmar ratio (annualized return / max drawdown).
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Series of portfolio values over time.
    periods_per_year : int, optional
        Number of periods in a year. Default is 252 (trading days).
    
    Returns:
    --------
    float
        Calmar ratio.
    """
    if len(equity_curve) <= 1:
        return 0.0
    
    # Calculate annualized return
    annualized_return = calculate_annualized_return(equity_curve, periods_per_year)
    
    # Calculate max drawdown
    max_drawdown, _, _ = calculate_max_drawdown(equity_curve)
    
    # Calculate Calmar ratio
    if max_drawdown == 0:
        return float('inf')  # No drawdown, perfect Calmar
    
    calmar_ratio = annualized_return / abs(max_drawdown)
    
    return calmar_ratio


def calculate_performance_metrics(equity_curve: pd.Series, trades: pd.DataFrame, 
                                 risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict:
    """
    Calculate all performance metrics.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Series of portfolio values over time.
    trades : pd.DataFrame
        DataFrame of trades.
    risk_free_rate : float, optional
        Risk-free rate, annualized. Default is 0.
    periods_per_year : int, optional
        Number of periods in a year. Default is 252 (trading days).
    
    Returns:
    --------
    dict
        Dictionary of performance metrics.
    """
    # Calculate returns
    returns = calculate_returns(equity_curve)
    
    # Calculate metrics
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1 if len(equity_curve) > 0 else 0.0
    annualized_return = calculate_annualized_return(equity_curve, periods_per_year)
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    max_drawdown, peak_date, valley_date = calculate_max_drawdown(equity_curve)
    drawdown_duration = calculate_drawdown_duration(equity_curve)
    volatility = calculate_volatility(returns, periods_per_year)
    calmar_ratio = calculate_calmar_ratio(equity_curve, periods_per_year)
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)
    avg_trade = calculate_average_trade(trades)
    
    # Create metrics dictionary
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'drawdown_duration': drawdown_duration,
        'volatility': volatility,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_trade': avg_trade,
        'total_trades': len(trades),
        'drawdown_peak_date': peak_date,
        'drawdown_valley_date': valley_date
    }
    
    return metrics


def calculate_trade_metrics(trades: pd.DataFrame) -> Dict:
    """
    Calculate trade-specific metrics.
    
    Parameters:
    -----------
    trades : pd.DataFrame
        DataFrame of trades.
    
    Returns:
    --------
    dict
        Dictionary of trade metrics.
    """
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_trade': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
    
    # Determine which column to use for profit
    profit_col = None
    for col in ['profit', 'pnl', 'return']:
        if col in trades.columns:
            profit_col = col
            break
    
    if profit_col is None:
        # Try to calculate profit from price and quantity
        if all(col in trades.columns for col in ['price', 'quantity', 'direction']):
            # Create a profit column based on direction, price, and quantity
            trades['profit'] = trades.apply(
                lambda row: row['price'] * row['quantity'] * (1 if row['direction'] == 'SELL' else -1),
                axis=1
            )
            profit_col = 'profit'
        else:
            return {
                'total_trades': len(trades),
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
    
    # Split trades into winning and losing
    winning_trades = trades[trades[profit_col] > 0]
    losing_trades = trades[trades[profit_col] < 0]
    
    # Calculate metrics
    total_trades = len(trades)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
    
    gross_profit = winning_trades[profit_col].sum() if len(winning_trades) > 0 else 0.0
    gross_loss = abs(losing_trades[profit_col].sum()) if len(losing_trades) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_trade = trades[profit_col].mean() if total_trades > 0 else 0.0
    avg_win = winning_trades[profit_col].mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades[profit_col].mean() if len(losing_trades) > 0 else 0.0
    
    largest_win = winning_trades[profit_col].max() if len(winning_trades) > 0 else 0.0
    largest_loss = losing_trades[profit_col].min() if len(losing_trades) > 0 else 0.0
    
    # Create metrics dictionary
    metrics = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_trade': avg_trade,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss
    }
    
    return metrics
