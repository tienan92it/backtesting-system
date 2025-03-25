# AI-Driven Backtesting System

A system that allows users to define trading strategies in natural language and automatically convert them to executable code for backtesting against historical crypto data.

## Overview

The AI-Driven Backtesting System combines natural language processing with quantitative finance to make algorithmic trading more accessible. It enables users without programming experience to create and test trading strategies using plain English descriptions.

## Key Features

- **Natural Language Input**: Define strategies in plain English
- **AI-Powered Code Generation**: Automatically convert descriptions to executable code
- **Backtesting Engine**: Test strategies against historical crypto data
- **Performance Metrics**: Analyze returns, drawdowns, and other key metrics
- **Interactive Reports**: Visualize performance with charts and tables
- **Web UI**: User-friendly interface for running backtests
- **REST API**: Programmable access for integration with other tools

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages (install with `pip install -r requirements.txt`)
- OpenAI API key for the code generation features

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/backtesting-system.git
cd backtesting-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

### Running the System

#### All-in-One

The easiest way to run both the API backend and UI frontend:

```bash
python -m ai_driven.run_all
```

This will start both services and open the UI in your browser.

#### Components Separately

Run the API backend:
```bash
python -m ai_driven.api.run
```

Run the UI frontend:
```bash
python -m ai_driven.ui.run
```

Run a CLI test:
```bash
python -m ai_driven.test_workflow
```

### Usage Examples

#### Web UI

1. Navigate to `http://localhost:8501` in your browser
2. Enter a strategy description in the text area
3. Configure backtest parameters in the sidebar
4. Click "Run Backtest" to execute
5. View the results in the dashboard

#### API

```python
import requests

# API endpoint
url = "http://localhost:8000/backtest"

# Strategy and configuration
payload = {
    "strategy": "Buy when RSI falls below 30 and sell when RSI rises above 70",
    "symbol": "BTC/USDT",
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "use_sample_data": True
}

# Make the request
response = requests.post(url, json=payload)

# Check response
if response.status_code == 200:
    result = response.json()
    print(f"Total Return: {result['metrics']['total_return']}%")
    
    # Save the HTML report
    with open("backtest_report.html", "w") as f:
        f.write(result["report_html"])
```

## System Architecture

The system uses a modular pipeline architecture:

1. **Strategy Parser**: Interprets natural language into structured specifications
2. **Code Generator**: Transforms specifications into executable Python strategies
3. **Backtest Runner**: Executes strategies against historical data
4. **Report Builder**: Transforms results into visual reports with metrics
5. **FastAPI Backend**: Exposes the workflow via a REST API
6. **Streamlit UI**: Provides a user-friendly web interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 