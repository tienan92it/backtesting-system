"""
Streamlit frontend for the AI-Driven Backtesting System.

This module provides a web UI for creating and running backtests
based on natural language strategy descriptions.
"""

import os
import json
import time
import base64
from datetime import datetime, timedelta
import streamlit as st
import requests
import pandas as pd
from urllib.parse import urljoin
import asyncio

# Set page configuration
st.set_page_config(
    page_title="AI-Driven Backtesting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default API URL
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]
DEFAULT_TIMEFRAMES = ["1d", "4h", "1h", "30m", "15m", "5m"]

# Example strategies
EXAMPLE_STRATEGIES = [
    "Buy when the 50-day moving average crosses above the 200-day moving average, and sell when the 50-day moving average crosses below the 200-day moving average.",
    "Buy when RSI falls below 30 and sell when RSI rises above 70.",
    "Buy when price breaks above the upper Bollinger Band (20, 2) and sell when it falls below the middle band.",
    "Buy when MACD line crosses above the signal line, and sell when it crosses below.",
    "Create a dual-moving average strategy with 10 and 30 period EMAs. Go long when the fast EMA crosses above the slow EMA and go short when it crosses below."
]

def get_api_url():
    """Get the API URL from session state or environment variable."""
    if "api_url" not in st.session_state:
        st.session_state.api_url = os.environ.get("BACKTEST_API_URL", DEFAULT_API_URL)
    return st.session_state.api_url

def get_backtest_config():
    """Get the backtest configuration from session state."""
    if "backtest_config" not in st.session_state:
        st.session_state.backtest_config = {
            "symbol": "BTC/USDT",
            "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "initial_capital": 10000.0,
            "timeframe": "1d",
            "use_sample_data": True
        }
    return st.session_state.backtest_config

def update_backtest_config(key, value):
    """Update a specific key in the backtest configuration."""
    config = get_backtest_config()
    config[key] = value
    st.session_state.backtest_config = config

def run_backtest(strategy, config):
    """
    Run a backtest using the API.
    
    Parameters:
    -----------
    strategy : str
        The natural language strategy description
    config : dict
        The backtest configuration
        
    Returns:
    --------
    dict or None
        The API response if successful, None otherwise
    """
    api_url = get_api_url()
    endpoint = urljoin(api_url, "backtest")
    
    # Create the request payload
    payload = {
        "strategy": strategy,
        **config
    }
    
    try:
        # Make the request
        response = requests.post(endpoint, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def display_metrics(metrics):
    """Display backtest metrics in a clean UI."""
    if not metrics:
        return
    
    # Create metric columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return", 
            f"{metrics.get('total_return', 0):.2f}%",
            delta=None
        )
    
    with col2:
        sharpe = metrics.get('sharpe_ratio')
        if sharpe is not None:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    with col3:
        drawdown = metrics.get('max_drawdown')
        if drawdown is not None:
            st.metric("Max Drawdown", f"{drawdown:.2f}%")
    
    with col4:
        win_rate = metrics.get('win_rate')
        if win_rate is not None:
            st.metric("Win Rate", f"{win_rate:.2f}%")
    
    # Additional metrics in expandable section
    with st.expander("Additional Metrics"):
        # Show total trades
        st.info(f"Total Trades: {metrics.get('total_trades', 0)}")
        
        # Show any other metrics
        additional = metrics.get('additional_metrics', {})
        if additional:
            for key, value in additional.items():
                st.info(f"{key.replace('_', ' ').title()}: {value}")

def display_trades(trades):
    """Display backtest trades in a table."""
    if not trades:
        st.info("No trades were executed during this backtest.")
        return
    
    # Convert to DataFrame for display
    trades_df = pd.DataFrame(trades)
    
    # Format the timestamp columns
    if 'timestamp' in trades_df.columns:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    if 'exit_timestamp' in trades_df.columns:
        trades_df['exit_timestamp'] = pd.to_datetime(trades_df['exit_timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Display the trades table
    st.dataframe(trades_df, use_container_width=True)

def display_report(report_html):
    """Display the HTML report."""
    if not report_html:
        st.warning("No report was generated.")
        return
    
    # Display the HTML report in an iframe
    st.components.v1.html(report_html, height=800, scrolling=True)

def display_strategy_code(strategy_code):
    """Display the generated strategy code."""
    if not strategy_code:
        st.warning("No strategy code was generated.")
        return
    
    # Display the code in a code block
    st.code(strategy_code, language="python")

def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("Backtest Configuration")
    
    config = get_backtest_config()
    
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Symbol",
        options=DEFAULT_SYMBOLS,
        index=DEFAULT_SYMBOLS.index(config["symbol"]) if config["symbol"] in DEFAULT_SYMBOLS else 0
    )
    update_backtest_config("symbol", symbol)
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.strptime(config["start_date"], "%Y-%m-%d"),
            format="YYYY-MM-DD"
        )
        update_backtest_config("start_date", start_date.strftime("%Y-%m-%d"))
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.strptime(config["end_date"], "%Y-%m-%d"),
            format="YYYY-MM-DD"
        )
        update_backtest_config("end_date", end_date.strftime("%Y-%m-%d"))
    
    # Timeframe
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=DEFAULT_TIMEFRAMES,
        index=DEFAULT_TIMEFRAMES.index(config["timeframe"]) if config["timeframe"] in DEFAULT_TIMEFRAMES else 0
    )
    update_backtest_config("timeframe", timeframe)
    
    # Initial capital
    initial_capital = st.sidebar.number_input(
        "Initial Capital (USD)",
        min_value=100.0,
        max_value=1000000.0,
        value=float(config["initial_capital"]),
        step=1000.0
    )
    update_backtest_config("initial_capital", initial_capital)
    
    # Sample data toggle
    use_sample_data = st.sidebar.checkbox(
        "Use Sample Data",
        value=config["use_sample_data"],
        help="Use generated sample data instead of real data"
    )
    update_backtest_config("use_sample_data", use_sample_data)
    
    # API settings in expander
    with st.sidebar.expander("API Settings"):
        api_url = st.text_input("API URL", value=get_api_url())
        if api_url != get_api_url():
            st.session_state.api_url = api_url
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2024 AI-Driven Backtesting System")

def render_strategy_input():
    """Render the strategy input section."""
    st.title("AI-Driven Backtesting")
    
    # Strategy input
    st.subheader("Strategy Description")
    
    # Example selector
    use_example = st.checkbox("Use example strategy", value=False)
    
    if use_example:
        example_index = st.selectbox(
            "Select an example",
            options=range(len(EXAMPLE_STRATEGIES)),
            format_func=lambda i: EXAMPLE_STRATEGIES[i][:50] + "..."
        )
        strategy = st.text_area(
            "Describe your trading strategy in plain English",
            value=EXAMPLE_STRATEGIES[example_index],
            height=150
        )
    else:
        strategy = st.text_area(
            "Describe your trading strategy in plain English",
            placeholder="Buy when the 50-day moving average crosses above the 200-day moving average...",
            height=150
        )
    
    # Run button
    col1, col2 = st.columns([1, 5])
    with col1:
        run_button = st.button("Run Backtest", type="primary", use_container_width=True)
    
    return strategy, run_button

def simulate_streaming_response(strategy, config):
    """
    Simulate a streaming response for the backtest.
    This function will be replaced with actual streaming when the API supports it.
    
    Parameters:
    -----------
    strategy : str
        The natural language strategy description
    config : dict
        The backtest configuration
    """
    # Create a placeholder for the progress bar
    progress_placeholder = st.empty()
    
    # Create a placeholder for status updates
    status_placeholder = st.empty()
    
    # Create a placeholder for the streaming content
    content_placeholder = st.empty()
    
    try:
        # Step 1: Parsing the strategy
        progress_placeholder.progress(0, text="Starting backtest...")
        status_placeholder.info("Parsing strategy description...")
        time.sleep(0.5)
        
        # Step 2: Generating code
        progress_placeholder.progress(25, text="Parsing strategy...")
        status_placeholder.info("Generating strategy code...")
        time.sleep(1)
        
        # Step 3: Running backtest
        progress_placeholder.progress(50, text="Generating code...")
        status_placeholder.info("Running backtest...")
        time.sleep(1.5)
        
        # Step 4: Collecting results
        progress_placeholder.progress(75, text="Running backtest...")
        status_placeholder.info("Collecting results...")
        time.sleep(1)
        
        # Make the actual API call
        result = run_backtest(strategy, config)
        
        # Complete the progress bar
        progress_placeholder.progress(100, text="Complete!")
        
        # Clear the status message
        status_placeholder.empty()
        
        # Return the result
        return result
        
    except Exception as e:
        # Show error
        progress_placeholder.empty()
        status_placeholder.error(f"Error: {e}")
        return None
    
def main():
    """Main function to run the Streamlit app."""
    # Render the sidebar
    render_sidebar()
    
    # Render the strategy input
    strategy, run_button = render_strategy_input()
    
    # Run the backtest if the button is clicked
    if run_button:
        if not strategy:
            st.error("Please enter a strategy description.")
            return
        
        # Disable the button while running (this doesn't actually work in Streamlit yet)
        st.session_state.running = True
        
        # Get the configuration
        config = get_backtest_config()
        
        # Create tabs for results
        tabs_placeholder = st.empty()
        
        with st.spinner("Running backtest..."):
            # Simulate streaming response
            result = simulate_streaming_response(strategy, config)
            
            if result:
                # Create tabs for different sections of the result
                tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Trades", "Report", "Code"])
                
                with tab1:
                    # Display metrics
                    st.subheader("Performance Metrics")
                    display_metrics(result.get("metrics"))
                
                with tab2:
                    # Display trades
                    st.subheader("Trades")
                    display_trades(result.get("trades"))
                
                with tab3:
                    # Display report
                    st.subheader("Backtest Report")
                    display_report(result.get("report_html"))
                
                with tab4:
                    # Display strategy code
                    st.subheader("Generated Strategy Code")
                    display_strategy_code(result.get("strategy_code"))
        
        # Reset the running state
        st.session_state.running = False

if __name__ == "__main__":
    main() 