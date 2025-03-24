# FastAPI Backend for AI-Driven Backtesting

## Overview

This module provides a REST API for the AI-Driven Backtesting System. It allows users to submit natural language trading strategy descriptions and receive backtest results via HTTP requests.

## Key Components

- **FastAPI Application**: A modern, fast web framework for building APIs with Python
- **Pydantic Models**: Type-annotated request and response models
- **API Documentation**: Auto-generated OpenAPI documentation
- **CORS Support**: Cross-Origin Resource Sharing configuration
- **Error Handling**: Comprehensive error handling and reporting

## Technical Approach

The API follows RESTful principles and provides:

1. A `/backtest` endpoint that accepts strategy descriptions and configuration parameters
2. JSON-formatted responses with backtest results and metrics
3. HTML report content for frontend rendering
4. Proper error handling and status codes
5. OpenAPI documentation for client integration

## Endpoints

### POST /backtest

Runs a backtest with a natural language strategy description.

**Request Body:**
```json
{
  "strategy": "Buy when the 50-day moving average crosses above the 200-day moving average...",
  "symbol": "BTC/USDT",
  "start_date": "2022-01-01",
  "end_date": "2022-12-31",
  "initial_capital": 10000.0,
  "timeframe": "1d",
  "use_sample_data": true,
  "save_artifacts": false
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Backtest completed successfully",
  "metrics": {
    "total_return": 15.23,
    "sharpe_ratio": 1.2,
    "max_drawdown": -12.5,
    "win_rate": 62.5,
    "total_trades": 8
  },
  "config": {
    "symbol": "BTC/USDT",
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "initial_capital": 10000.0,
    "timeframe": "1d"
  },
  "trades": [
    {
      "timestamp": "2022-02-15T00:00:00",
      "symbol": "BTC/USDT",
      "direction": "BUY",
      "quantity": 0.05,
      "price": 44325.12,
      "commission": 2.22
    },
    // ...more trades
  ],
  "report_html": "<!DOCTYPE html>...",
  "strategy_code": "class GeneratedStrategy(Strategy):\n..."
}
```

### POST /backtest/async

Asynchronous version of the `/backtest` endpoint (placeholder for future implementation).

## Usage Example

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
    
    print("Report saved to backtest_report.html")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

## Running the API Server

To run the API server:

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure the environment file
cp .env.example .env
# Edit .env with your configuration

# Run the server
python -m ai_driven.api.run

# With custom host and port
python -m ai_driven.api.run --host 0.0.0.0 --port 8080

# For development with auto-reload
python -m ai_driven.api.run --reload
```

## Security Considerations

- The API currently accepts requests from all origins (CORS wildcard). In production, restrict this to specific trusted origins.
- For public deployment, implement authentication to prevent abuse.
- OpenAI API key is loaded from environment variables rather than hardcoded.
- Future enhancements could include API rate limiting and HTTPS configuration. 