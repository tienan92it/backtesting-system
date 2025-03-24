"""
FastAPI application for the AI-Driven Backtesting System.

This module provides a REST API for running backtests with natural
language strategy descriptions.
"""

import os
import logging
import traceback
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from ai_driven.api.models import (
    BacktestRequest, 
    BacktestResponse, 
    BacktestMetrics,
    Trade,
    ErrorResponse
)
from ai_driven.workflow import run_ai_backtest_workflow
from ai_driven.runner import BacktestResult

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI-Driven Backtesting API",
    description="API for running backtests with natural language strategy descriptions",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path for storing artifacts
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "ai_driven/generated")

def get_openai_api_key():
    """
    Get OpenAI API key from environment variable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY environment variable not set")
    return api_key

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint - returns a simple HTML page with API information.
    """
    return """
    <html>
        <head>
            <title>AI-Driven Backtesting API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }
                h1 {
                    color: #333;
                }
                a {
                    color: #0066cc;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
                code {
                    background-color: #f5f5f5;
                    padding: 2px 4px;
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <h1>AI-Driven Backtesting API</h1>
            <p>Welcome to the AI-Driven Backtesting API. This service allows you to run backtests with natural language strategy descriptions.</p>
            <p>For API documentation, visit <a href="/docs">/docs</a> or <a href="/redoc">/redoc</a>.</p>
            <p>Example usage:</p>
            <pre><code>
POST /backtest
{
    "strategy": "Buy when the 50-day moving average crosses above the 200-day moving average, and sell when the 50-day moving average crosses below the 200-day moving average.",
    "symbol": "BTC/USDT",
    "start_date": "2022-01-01",
    "end_date": "2022-12-31"
}
            </code></pre>
        </body>
    </html>
    """

@app.post(
    "/backtest",
    response_model=BacktestResponse,
    responses={
        200: {"description": "Backtest completed successfully"},
        422: {"description": "Validation Error", "model": ErrorResponse},
        500: {"description": "Server Error", "model": ErrorResponse}
    },
    tags=["Backtest"]
)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_openai_api_key)
):
    """
    Run a backtest with a natural language strategy description.
    
    This endpoint processes a natural language strategy description, 
    generates executable strategy code, runs a backtest, and returns 
    the results with performance metrics and visualizations.
    
    The backtest is run synchronously for simplicity in the MVP. For
    longer backtests, consider using the background_tasks parameter
    to run the backtest asynchronously.
    """
    logger.info(f"Received backtest request for strategy: {request.strategy[:50]}...")
    
    try:
        # Prepare backtest configuration
        config = {
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "initial_capital": request.initial_capital,
            "timeframe": request.timeframe,
            "commission": request.commission,
            "exchange": request.exchange,
            "use_sample_data": request.use_sample_data
        }
        
        # Run the workflow
        result, report_html = run_ai_backtest_workflow(
            strategy_description=request.strategy,
            backtest_config=config,
            save_artifacts=request.save_artifacts,
            artifacts_dir=ARTIFACTS_DIR,
            api_key=api_key
        )
        
        # Process the result
        return process_backtest_result(result, report_html)
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Backtest failed: {str(e)}",
                "details": {
                    "traceback": traceback.format_exc()
                }
            }
        )

@app.post(
    "/backtest/async",
    response_model=BacktestResponse,
    responses={
        202: {"description": "Backtest accepted and processing in background"},
        422: {"description": "Validation Error", "model": ErrorResponse},
        500: {"description": "Server Error", "model": ErrorResponse}
    },
    tags=["Backtest"]
)
async def run_backtest_async(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_openai_api_key)
):
    """
    Run a backtest asynchronously with a natural language strategy description.
    
    This endpoint accepts a backtest request and processes it in the background.
    It immediately returns with a 202 Accepted status, and the client can poll
    for results or be notified through a webhook (in a future version).
    
    NOTE: This is a placeholder for future implementation. Currently it runs
    synchronously like the /backtest endpoint but is structured for future
    async implementation.
    """
    logger.info(f"Received async backtest request for strategy: {request.strategy[:50]}...")
    
    # This is a placeholder for future background task implementation
    # In a real implementation, this would return immediately and run the task in the background
    # For now, it just calls the synchronous version
    
    try:
        # Prepare backtest configuration
        config = {
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "initial_capital": request.initial_capital,
            "timeframe": request.timeframe,
            "commission": request.commission,
            "exchange": request.exchange,
            "use_sample_data": request.use_sample_data
        }
        
        # Run the workflow (synchronously for now)
        result, report_html = run_ai_backtest_workflow(
            strategy_description=request.strategy,
            backtest_config=config,
            save_artifacts=request.save_artifacts,
            artifacts_dir=ARTIFACTS_DIR,
            api_key=api_key
        )
        
        # Process the result
        return process_backtest_result(result, report_html)
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Backtest failed: {str(e)}",
                "details": {
                    "traceback": traceback.format_exc()
                }
            }
        )

def process_backtest_result(result: BacktestResult, report_html: str) -> BacktestResponse:
    """
    Process a backtest result into an API response.
    
    Parameters:
    -----------
    result : BacktestResult
        The result of the backtest.
    report_html : str
        The HTML report generated from the backtest.
        
    Returns:
    --------
    BacktestResponse
        API response with backtest results.
    """
    # Create metrics
    metrics = BacktestMetrics(
        total_return=result.metrics.get('total_return', 0.0),
        sharpe_ratio=result.metrics.get('sharpe_ratio'),
        max_drawdown=result.metrics.get('max_drawdown'),
        win_rate=result.metrics.get('win_rate'),
        total_trades=len(result.trades),
    )
    
    # Process additional metrics
    additional_metrics = {}
    for key, value in result.metrics.items():
        if key not in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
            additional_metrics[key] = value
    
    if additional_metrics:
        metrics.additional_metrics = additional_metrics
    
    # Process trades (limit to first 20 for response size)
    trades = []
    for trade_data in result.trades[:20]:
        # Convert trade dict to Trade model
        trade = Trade(
            timestamp=trade_data.get('timestamp', datetime.now()),
            symbol=trade_data.get('symbol', ''),
            direction=trade_data.get('direction', ''),
            quantity=trade_data.get('quantity', 0.0),
            price=trade_data.get('price', 0.0),
            commission=trade_data.get('commission'),
            profit=trade_data.get('profit'),
            exit_price=trade_data.get('exit_price'),
            exit_timestamp=trade_data.get('exit_timestamp')
        )
        trades.append(trade)
    
    # Create response
    response = BacktestResponse(
        status="success",
        message="Backtest completed successfully",
        metrics=metrics,
        config=result.config,
        trades=trades,
        report_html=report_html,
        strategy_code=result.strategy_code
    )
    
    return response 