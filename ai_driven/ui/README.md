# Streamlit UI for AI-Driven Backtesting

## Overview

This module provides a user-friendly web interface for the AI-Driven Backtesting System using Streamlit. Users can input natural language strategy descriptions, configure backtest parameters, and view comprehensive results with visualizations.

## Key Components

- **Strategy Input**: Text area for entering natural language strategy descriptions
- **Backtest Configuration**: Sidebar controls for configuring backtest parameters
- **Example Strategies**: Pre-built examples to help users get started
- **Results Dashboard**: Interactive display of backtest results with metrics
- **Visual Report**: HTML report with performance charts and metrics
- **Generated Code**: Display of the AI-generated strategy code

## Technical Approach

The UI is built with Streamlit, a Python framework for creating data apps. It:

1. Connects to the FastAPI backend to execute backtests
2. Provides an intuitive interface for entering strategies in plain English
3. Simulates streaming responses during backtest execution
4. Displays results in a tabbed interface with rich visualizations
5. Offers a responsive, modern design with minimal complexity

## Usage

### Running the UI

```bash
# Simple way to run
python -m ai_driven.ui.run

# With custom port
python -m ai_driven.ui.run --port 8888

# With custom API URL
python -m ai_driven.ui.run --api-url http://api.example.com
```

### Direct Streamlit command
```bash
streamlit run ai_driven/ui/app.py
```

## Workflow

1. User enters a natural language description of their trading strategy
2. User configures backtest parameters (symbol, date range, etc.)
3. User clicks "Run Backtest" to initiate the backtest
4. UI displays a progress bar and status updates during execution
5. Results are displayed in a tabbed interface with:
   - Dashboard of key performance metrics
   - Table of executed trades
   - Visual HTML report with charts
   - Generated strategy code

## Example Strategies

The UI includes several example strategies to help users get started:

1. Moving Average Crossover
2. RSI Oversold/Overbought
3. Bollinger Bands Breakout
4. MACD Signal Line Cross
5. Dual EMA Strategy

## Configuration Options

The sidebar allows users to configure:

- Trading symbol (BTC/USDT, ETH/USDT, etc.)
- Date range for backtesting
- Timeframe (1d, 4h, 1h, etc.)
- Initial capital
- Sample data vs. real data option
- API connection settings

## Future Enhancements

- Real-time streaming of results as they are processed
- Strategy template builder
- Comparison of multiple strategy backtests
- Strategy parameter optimization
- Advanced charting with interactive plots
- User authentication and saved strategies 