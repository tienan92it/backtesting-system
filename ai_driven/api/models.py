"""
Pydantic models for the AI-Driven Backtesting API.

These models define the structure of API requests and responses.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class BacktestRequest(BaseModel):
    """
    Request model for the backtest endpoint.
    
    Contains the strategy description and configuration parameters.
    """
    strategy: str = Field(
        ...,
        description="Natural language description of the trading strategy"
    )
    symbol: str = Field(
        default="BTC/USDT",
        description="Trading symbol"
    )
    start_date: str = Field(
        default="2022-01-01",
        description="Start date for the backtest (YYYY-MM-DD)"
    )
    end_date: str = Field(
        default="2022-12-31",
        description="End date for the backtest (YYYY-MM-DD)"
    )
    initial_capital: float = Field(
        default=10000.0,
        description="Initial capital for the backtest"
    )
    timeframe: str = Field(
        default="1d",
        description="Candle timeframe (e.g., 1d, 4h, 1h)"
    )
    commission: float = Field(
        default=0.001,
        description="Commission rate (e.g., 0.001 for 0.1%)"
    )
    exchange: str = Field(
        default="binance",
        description="Exchange to use (binance or ccxt)"
    )
    use_sample_data: bool = Field(
        default=True,
        description="Whether to use generated sample data instead of real data"
    )
    save_artifacts: bool = Field(
        default=False,
        description="Whether to save the generated artifacts (strategy code, report)"
    )


class BacktestMetrics(BaseModel):
    """
    Model for backtest performance metrics.
    """
    total_return: float = Field(
        default=0.0,
        description="Total percentage return of the strategy"
    )
    sharpe_ratio: Optional[float] = Field(
        default=None,
        description="Sharpe ratio of the strategy"
    )
    max_drawdown: Optional[float] = Field(
        default=None,
        description="Maximum drawdown percentage"
    )
    win_rate: Optional[float] = Field(
        default=None,
        description="Percentage of winning trades"
    )
    total_trades: Optional[int] = Field(
        default=None,
        description="Total number of trades executed"
    )
    additional_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional performance metrics"
    )


class Trade(BaseModel):
    """
    Model for a single trade in the backtest.
    """
    timestamp: Union[str, datetime]
    symbol: str
    direction: str
    quantity: float
    price: float
    commission: Optional[float] = None
    profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[Union[str, datetime]] = None


class BacktestResponse(BaseModel):
    """
    Response model for the backtest endpoint.
    
    Contains backtest results and the generated report.
    """
    status: str = Field(
        default="success",
        description="Status of the backtest (success or error)"
    )
    message: Optional[str] = Field(
        default=None,
        description="Additional information or error message"
    )
    metrics: Optional[BacktestMetrics] = Field(
        default=None,
        description="Performance metrics from the backtest"
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Configuration used for the backtest"
    )
    trades: Optional[List[Trade]] = Field(
        default=None,
        description="List of trades executed during the backtest (limited to first 20)"
    )
    report_html: Optional[str] = Field(
        default=None,
        description="HTML report of the backtest results"
    )
    strategy_code: Optional[str] = Field(
        default=None,
        description="Generated strategy code"
    )


class ErrorResponse(BaseModel):
    """
    Response model for API errors.
    """
    status: str = Field(
        default="error",
        description="Status of the request"
    )
    message: str = Field(
        ...,
        description="Error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    ) 