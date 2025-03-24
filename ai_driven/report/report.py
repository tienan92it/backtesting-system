"""
Report Builder Module for the AI-Driven Backtesting System.

This module is responsible for generating HTML reports from backtest results,
including performance metrics and visualizations.
"""

import os
import io
import base64
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import matplotlib.dates as mdates
from ai_driven.runner import BacktestResult
import jinja2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up Jinja2 environment
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(template_dir),
    autoescape=jinja2.select_autoescape(['html', 'xml'])
)

def generate_equity_curve_chart(equity_curve: pd.Series) -> str:
    """
    Generate a base64-encoded image of the equity curve chart.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        A pandas Series with dates as index and portfolio values
        
    Returns:
    --------
    str
        Base64-encoded image data for the chart
    """
    logger.info("Generating equity curve chart")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve.index, equity_curve.values, color='#0066cc', linewidth=2)
    
    # Add labels and title
    plt.title('Equity Curve', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    
    # Format the x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # Add grid and tight layout
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    
    # Encode the buffer to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_str}"

def generate_drawdown_chart(equity_curve: pd.Series) -> str:
    """
    Generate a base64-encoded image of the drawdown chart.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        A pandas Series with dates as index and portfolio values
        
    Returns:
    --------
    str
        Base64-encoded image data for the chart
    """
    logger.info("Generating drawdown chart")
    
    # Calculate drawdown series
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.fill_between(drawdown.index, drawdown.values, 0, color='#d13b40', alpha=0.3)
    plt.plot(drawdown.index, drawdown.values, color='#d13b40', linewidth=1)
    
    # Add labels and title
    plt.title('Drawdown (%)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    
    # Format the x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    
    # Add grid and tight layout
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    
    # Encode the buffer to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_str}"

def generate_monthly_returns_heatmap(returns: pd.Series) -> str:
    """
    Generate a base64-encoded image of monthly returns heatmap.
    
    Parameters:
    -----------
    returns : pd.Series
        A pandas Series with dates as index and daily returns
        
    Returns:
    --------
    str
        Base64-encoded image data for the chart
    """
    logger.info("Generating monthly returns heatmap")
    
    # Ensure we have a datetime index
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)
    
    # Group by year and month to calculate monthly returns
    monthly_returns = returns.groupby([lambda x: x.year, lambda x: x.month]).apply(
        lambda x: (x + 1).prod() - 1
    ) * 100  # Convert to percentage
    
    # Reshape into a year x month matrix
    monthly_matrix = monthly_returns.unstack()
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.RdYlGn  # Red for negative, green for positive
    
    # Create the heatmap
    im = ax.imshow(monthly_matrix.values, cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Returns (%)', rotation=-90, va="bottom")
    
    # Set labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(np.arange(len(month_labels)))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(np.arange(len(monthly_matrix.index)))
    ax.set_yticklabels(monthly_matrix.index)
    
    # Add the text labels with the values
    for i in range(len(monthly_matrix.index)):
        for j in range(len(month_labels)):
            if not np.isnan(monthly_matrix.values[i, j]):
                text = ax.text(j, i, f"{monthly_matrix.values[i, j]:.1f}%",
                              ha="center", va="center", color="black")
    
    # Add title and layout
    ax.set_title("Monthly Returns (%)")
    fig.tight_layout()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    
    # Encode the buffer to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_str}"

def format_trades_table(trades: List[Dict[str, Any]]) -> str:
    """
    Format the trades list into an HTML table.
    
    Parameters:
    -----------
    trades : list
        List of trade dictionaries
        
    Returns:
    --------
    str
        HTML table showing the trades
    """
    if not trades:
        return "<p>No trades executed during the backtest.</p>"
    
    html = """
    <div class="table-responsive">
        <table class="table table-striped table-hover">
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Symbol</th>
                    <th>Direction</th>
                    <th>Quantity</th>
                    <th>Price</th>
                    <th>Commission</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for trade in trades[:20]:  # Limit to first 20 trades for brevity
        timestamp = trade.get('timestamp')
        date_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, datetime) else str(timestamp)
        
        html += f"""
                <tr>
                    <td>{date_str}</td>
                    <td>{trade.get('symbol', '')}</td>
                    <td>{trade.get('direction', '')}</td>
                    <td>{trade.get('quantity', 0):.4f}</td>
                    <td>${trade.get('price', 0):.2f}</td>
                    <td>${trade.get('commission', 0):.2f}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
    """
    
    if len(trades) > 20:
        html += f"<p>Showing 20 out of {len(trades)} trades.</p>"
    
    html += "</div>"
    return html

def process_backtest_result(result: BacktestResult) -> Dict[str, Any]:
    """
    Process the backtest result to prepare data for the report.
    
    Parameters:
    -----------
    result : BacktestResult
        The backtest result object
        
    Returns:
    --------
    dict
        A dictionary with processed data for the report
    """
    logger.info("Processing backtest result for report generation")
    
    # Extract basic data from the result
    metrics = result.metrics.copy() if result.metrics else {}
    
    # Process equity curve data
    equity_curve = None
    if isinstance(result.equity_curve, pd.Series):
        equity_curve = result.equity_curve
    elif isinstance(result.equity_curve, list) and result.equity_curve:
        # Convert list of dicts to DataFrame
        equity_df = pd.DataFrame(result.equity_curve)
        equity_curve = pd.Series(equity_df['portfolio_value'].values, index=pd.to_datetime(equity_df['timestamp']))
    
    # Process returns data
    returns = None
    if isinstance(result.returns, pd.Series):
        returns = result.returns
    elif isinstance(result.returns, list) and result.returns:
        # Convert list format to Series if needed
        returns_df = pd.DataFrame(result.returns)
        returns = pd.Series(returns_df['return'].values, index=pd.to_datetime(returns_df['timestamp']))
    elif equity_curve is not None:
        # Calculate returns from equity curve if returns not provided
        returns = equity_curve.pct_change().fillna(0)
    
    # Generate additional metrics if not already present
    if equity_curve is not None and len(equity_curve) > 1:
        metrics['total_return'] = metrics.get('total_return', 
                                             (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100)
    
    if 'max_drawdown' not in metrics and equity_curve is not None:
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min() * 100
    
    if 'win_rate' not in metrics and result.trades:
        profitable_trades = sum(1 for trade in result.trades 
                               if trade.get('profit', 0) > 0 or 
                               (trade.get('direction') == 'BUY' and trade.get('exit_price', 0) > trade.get('price', 0)) or
                               (trade.get('direction') == 'SELL' and trade.get('exit_price', 0) < trade.get('price', 0)))
        metrics['win_rate'] = (profitable_trades / len(result.trades)) * 100 if result.trades else 0
    
    # Generate charts
    charts = {}
    if equity_curve is not None and len(equity_curve) > 1:
        charts['equity_curve'] = generate_equity_curve_chart(equity_curve)
        charts['drawdown'] = generate_drawdown_chart(equity_curve)
    
    if returns is not None and len(returns) > 30:  # Only if we have enough data
        try:
            charts['monthly_returns'] = generate_monthly_returns_heatmap(returns)
        except Exception as e:
            logger.warning(f"Could not generate monthly returns heatmap: {e}")
    
    # Format trades table
    trades_html = format_trades_table(result.trades)
    
    # Prepare result dictionary
    processed_result = {
        'strategy_name': result.strategy_params.get('name', 'Generated Strategy'),
        'backtest_params': {
            'symbol': result.config.get('symbol', 'Unknown'),
            'start_date': result.config.get('start_date', 'Unknown'),
            'end_date': result.config.get('end_date', 'Unknown'),
            'initial_capital': result.config.get('initial_capital', 0),
            'timeframe': result.config.get('timeframe', '1d')
        },
        'metrics': metrics,
        'charts': charts,
        'trades_html': trades_html,
        'strategy_code': result.strategy_code
    }
    
    return processed_result

def build_default_report(result: BacktestResult) -> str:
    """
    Build a default HTML report when no template is available.
    
    Parameters:
    -----------
    result : BacktestResult
        The backtest result
        
    Returns:
    --------
    str
        An HTML string containing the report
    """
    logger.info("Building default HTML report")
    
    # Process the backtest result
    data = process_backtest_result(result)
    
    # Basic CSS styles
    css = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            min-width: 200px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin-top: 10px;
        }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .chart-container {
            background-color: #fff;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .trades-container {
            background-color: #fff;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
        }
        .table {
            width: 100%;
            border-collapse: collapse;
        }
        .table th, .table td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .table th {
            background-color: #f5f5f5;
        }
        .table-striped tbody tr:nth-of-type(odd) {
            background-color: rgba(0,0,0,.05);
        }
        .table-hover tbody tr:hover {
            background-color: rgba(0,0,0,.075);
        }
        .code-container {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 20px;
            margin-top: 20px;
            overflow-x: auto;
        }
        pre {
            white-space: pre-wrap;
            margin: 0;
        }
        .section-title {
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-top: 40px;
            margin-bottom: 20px;
        }
    </style>
    """
    
    # Construct the HTML report
    strategy_name = data.get('strategy_name', 'Generated Strategy')
    backtest_params = data.get('backtest_params', {})
    metrics = data.get('metrics', {})
    charts = data.get('charts', {})
    
    # Format metrics for display
    total_return = f"{metrics.get('total_return', 0):.2f}%"
    total_return_class = 'positive' if metrics.get('total_return', 0) >= 0 else 'negative'
    
    sharpe_ratio = f"{metrics.get('sharpe_ratio', 0):.2f}"
    sharpe_class = 'positive' if metrics.get('sharpe_ratio', 0) >= 1 else 'negative'
    
    max_drawdown = f"{metrics.get('max_drawdown', 0):.2f}%"
    max_drawdown_class = 'negative'
    
    win_rate = f"{metrics.get('win_rate', 0):.2f}%"
    win_rate_class = 'positive' if metrics.get('win_rate', 0) >= 50 else 'negative'
    
    # Start building the HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Backtest Report - {strategy_name}</title>
        {css}
    </head>
    <body>
        <div class="header">
            <h1>Backtest Report - {strategy_name}</h1>
            <p>Symbol: {backtest_params.get('symbol', 'Unknown')} | Period: {backtest_params.get('start_date', 'Unknown')} to {backtest_params.get('end_date', 'Unknown')} | Timeframe: {backtest_params.get('timeframe', '1d')}</p>
        </div>
        
        <h2 class="section-title">Performance Metrics</h2>
        <div class="metrics-container">
            <div class="metric-card">
                <h3>Total Return</h3>
                <div class="metric-value {total_return_class}">{total_return}</div>
            </div>
            <div class="metric-card">
                <h3>Sharpe Ratio</h3>
                <div class="metric-value {sharpe_class}">{sharpe_ratio}</div>
            </div>
            <div class="metric-card">
                <h3>Max Drawdown</h3>
                <div class="metric-value {max_drawdown_class}">{max_drawdown}</div>
            </div>
            <div class="metric-card">
                <h3>Win Rate</h3>
                <div class="metric-value {win_rate_class}">{win_rate}</div>
            </div>
        </div>
        
        <h2 class="section-title">Equity Curve</h2>
    """
    
    # Add equity curve chart if available
    if 'equity_curve' in charts:
        html += f"""
        <div class="chart-container">
            <img src="{charts['equity_curve']}" alt="Equity Curve" style="width: 100%; height: auto;">
        </div>
        """
    
    # Add drawdown chart if available
    if 'drawdown' in charts:
        html += f"""
        <h2 class="section-title">Drawdown</h2>
        <div class="chart-container">
            <img src="{charts['drawdown']}" alt="Drawdown" style="width: 100%; height: auto;">
        </div>
        """
    
    # Add monthly returns chart if available
    if 'monthly_returns' in charts:
        html += f"""
        <h2 class="section-title">Monthly Returns</h2>
        <div class="chart-container">
            <img src="{charts['monthly_returns']}" alt="Monthly Returns" style="width: 100%; height: auto;">
        </div>
        """
    
    # Add trades section
    html += f"""
    <h2 class="section-title">Trades</h2>
    <div class="trades-container">
        {data.get('trades_html', '<p>No trades available.</p>')}
    </div>
    """
    
    # Add strategy code section
    html += f"""
    <h2 class="section-title">Strategy Code</h2>
    <div class="code-container">
        <pre><code>{data.get('strategy_code', 'No code available.')}</code></pre>
    </div>
    """
    
    # Close HTML tags
    html += """
    </body>
    </html>
    """
    
    return html

def build_report(result: BacktestResult, template_name: Optional[str] = None) -> str:
    """
    Build an HTML report from backtest results.
    
    Parameters:
    -----------
    result : BacktestResult
        The result of the backtest to generate a report for.
    template_name : str, optional
        The name of the template to use. If None, uses default template or fallback.
        
    Returns:
    --------
    str
        HTML string containing the report.
    """
    logger.info(f"Building report with template: {template_name}")
    
    # Process the backtest result
    data = process_backtest_result(result)
    
    try:
        # Try to load the specified template or default
        template_file = f"{template_name}.html" if template_name else "report_template.html"
        template = jinja_env.get_template(template_file)
        return template.render(**data)
    except jinja2.exceptions.TemplateNotFound:
        logger.warning(f"Template not found: {template_file}, using default HTML generation")
        return build_default_report(result)
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return build_default_report(result) 