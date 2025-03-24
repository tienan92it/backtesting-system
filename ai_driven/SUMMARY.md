# AI-Driven Backtesting System: Module Summary

## Primary Responsibilities

The AI-Driven Backtesting System enables users to define trading strategies in natural language, which are then automatically converted to executable code and backtested against historical data. The system returns comprehensive performance reports with visualizations.

## Technical Approach

The system uses a modular pipeline architecture with four primary components:

1. **Strategy Parser** - Interprets natural language into structured strategy specifications
2. **Code Generator** - Transforms specifications into executable Python trading strategies
3. **Backtest Runner** - Executes strategies against historical data and collects results
4. **Report Builder** - Transforms results into visual reports with metrics and charts

The components are coordinated by a **Workflow Orchestrator** that provides an end-to-end interface, and a **FastAPI Backend** that exposes the workflow via a REST API.

## Module Structure

```
ai_driven/
├── __init__.py             # Package exports
├── parser.py               # Strategy parsing (natural language → structured spec)
├── code_generator.py       # Strategy code generation (spec → Python code)
├── runner.py               # Backtest execution (code → performance results)
├── workflow.py             # End-to-end workflow orchestration
├── test_workflow.py        # CLI tool for testing the workflow
├── report/                 # Report generation module
│   ├── __init__.py         # Report module exports
│   ├── report.py           # Report building functionality
│   └── templates/          # Jinja2 HTML templates
│       └── report_template.html  # Default report template
├── api/                    # FastAPI backend module
│   ├── __init__.py         # API module exports
│   ├── app.py              # FastAPI application
│   ├── models.py           # Pydantic models for API
│   ├── run.py              # Script to run the API server
│   └── test_api.py         # Script to test the API
└── generated/              # Directory for storing generated artifacts
```

## Component Interactions

1. **Input**: Natural language strategy description + backtest configuration
2. **Parser**: Converts the description to a structured `StrategySpec` object
3. **Generator**: Transforms the spec into a Python strategy class as a string
4. **Runner**: Executes the strategy code in the backtesting engine
5. **Report Builder**: Converts the backtest results into an HTML report
6. **API**: Exposes the workflow as a REST API for client applications

## Integration with Core Backtesting Engine

The AI-driven system integrates with the core backtesting framework through:

- **Strategy Base Class**: Generated code extends the framework's `Strategy` class
- **Backtester**: The `run_backtest` function uses the framework's `Backtester` class
- **Event System**: Generated strategies use the event-driven architecture of the core framework

## Usage Flow

### Command-line Interface
1. User provides a natural language description of a trading strategy
2. System parses the description into a structured specification
3. Code generator creates executable Python strategy code
4. Backtester runs the strategy against historical market data
5. Report builder generates a visual HTML report of performance
6. User receives the report to analyze strategy effectiveness

### REST API
1. Client sends a POST request to `/backtest` with strategy description and parameters
2. API server processes the request using the workflow module
3. Results are returned as JSON with metrics and HTML report
4. Client can display the HTML report or extract metrics for further processing

## Future Extensions

The modular design allows for easy extension in several directions:

- Supporting more complex strategy patterns and indicators
- Adding more visualization types to the report
- Integration with alternative LLM providers
- Real-time strategy execution against live market data
- Advanced asynchronous processing for long-running backtests
- User authentication and request rate limiting in the API 