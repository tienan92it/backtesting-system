# Metrics Module Knowledge Base

## 1. Module Overview
- **Module Name & Version:**  
  `Metrics Module v1.0`
  
- **Purpose & Scope:**  
  The Metrics module is responsible for calculating and evaluating the performance of trading strategies. It provides a comprehensive set of financial metrics that help quantify the effectiveness, risk, and profitability of backtested strategies.

- **Key Responsibilities:**  
  - Calculate portfolio performance metrics (returns, Sharpe ratio, drawdowns, etc.)
  - Analyze trade statistics (win rate, profit factor, average trade, etc.)
  - Provide risk assessment measurements
  - Support strategy comparison and optimization
  - Generate performance reports
  
- **Dependencies & Interfaces:**  
  - Dependencies: pandas, numpy
  - Interfaces with Portfolio module for equity curve and trade data
  - Used by Backtester to evaluate strategy performance

## 2. Detailed Functionality & Responsibilities

### Performance Metrics
- **Return Metrics:**
  - Total return: Overall percentage gain or loss
  - Annualized return: Return normalized to a yearly basis
  - Returns series: Period-by-period percentage changes

- **Risk Metrics:**
  - Maximum drawdown: Largest peak-to-trough decline
  - Volatility: Standard deviation of returns
  - Drawdown duration: Average and maximum time in drawdown
  - Downside deviation: Standard deviation of negative returns only

- **Risk-Adjusted Returns:**
  - Sharpe ratio: Return per unit of risk (volatility)
  - Sortino ratio: Return per unit of downside risk
  - Calmar ratio: Annualized return divided by maximum drawdown

### Trade Analysis
- **Trade Statistics:**
  - Win rate: Percentage of profitable trades
  - Profit factor: Gross profit divided by gross loss
  - Average trade: Mean profit/loss per trade
  - Average win/loss: Mean profit of winning/losing trades
  - Largest win/loss: Maximum profit/loss of individual trades

- **Trade Distribution:**
  - Holding periods: How long trades are typically held
  - Profit distribution: Statistical distribution of trade outcomes
  - Trade frequency: Number of trades per period

## 3. API / Interface Description

### Performance Calculation Functions
```python
def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    # Calculate returns from an equity curve
    
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    # Calculate the Sharpe ratio (return per unit of risk)
    
def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    # Calculate the Sortino ratio (return per unit of downside risk)
    
def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    # Calculate the maximum drawdown and its duration
    
def calculate_drawdown_duration(equity_curve: pd.Series) -> int:
    # Calculate the average drawdown duration in days
    
def calculate_win_rate(trades: pd.DataFrame) -> float:
    # Calculate the win rate from a list of trades
    
def calculate_profit_factor(trades: pd.DataFrame) -> float:
    # Calculate the profit factor (gross profit / gross loss)
    
def calculate_average_trade(trades: pd.DataFrame) -> float:
    # Calculate the average profit per trade
    
def calculate_annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    # Calculate the annualized return
    
def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    # Calculate the annualized volatility
    
def calculate_calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    # Calculate the Calmar ratio (annualized return / max drawdown)
```

### Aggregate Performance Functions
```python
def calculate_performance_metrics(equity_curve: pd.Series, trades: pd.DataFrame, 
                                 risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict:
    # Calculate all performance metrics in one function
    
def calculate_trade_metrics(trades: pd.DataFrame) -> Dict:
    # Calculate trade-specific metrics
```

## 4. Usage Examples

### Calculating Basic Metrics
```python
from backtesting.metrics.performance import calculate_sharpe_ratio, calculate_max_drawdown
import pandas as pd

# Sample equity curve
equity_curve = pd.Series([10000, 10100, 10050, 10200, 10300, 10150, 10400], 
                        index=pd.date_range('2022-01-01', periods=7, freq='D'))

# Calculate returns
returns = equity_curve.pct_change().fillna(0)

# Calculate Sharpe ratio
sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
print(f"Sharpe Ratio: {sharpe:.2f}")

# Calculate maximum drawdown
max_dd, peak_date, valley_date = calculate_max_drawdown(equity_curve)
print(f"Maximum Drawdown: {max_dd:.2%}")
print(f"Drawdown Period: {peak_date} to {valley_date}")
```

### Comprehensive Performance Analysis
```python
from backtesting.metrics.performance import calculate_performance_metrics
import pandas as pd

# Sample equity curve and trades
equity_curve = pd.Series([10000, 10500, 11000, 10800, 11500, 12000], 
                         index=pd.date_range('2022-01-01', periods=6, freq='M'))

trades = pd.DataFrame({
    'entry_time': pd.date_range('2022-01-05', periods=4, freq='M'),
    'exit_time': pd.date_range('2022-02-05', periods=4, freq='M'),
    'profit': [500, -200, 700, 500],
    'entry_price': [100, 105, 108, 115],
    'exit_price': [105, 103, 115, 120],
    'quantity': [100, 100, 100, 100]
})

# Calculate all metrics
metrics = calculate_performance_metrics(equity_curve, trades, risk_free_rate=0.01)

# Print key metrics
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Annualized Return: {metrics['annualized_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
```

## 5. Configuration & Environment Setup
- **Required Python Version:** 3.7+
- **Dependencies:**
  - pandas: For data manipulation and time series analysis
  - numpy: For numerical operations

## 6. Testing & Validation
- **Edge Case Testing:**
  Test metrics functions with empty datasets, single points, and other edge cases.

- **Known Results Testing:**
  Verify metrics calculations against known examples and established financial libraries.

- **Numerical Stability Testing:**
  Ensure calculations remain stable with extreme values (very large returns, zero volatility, etc.).

## 7. LLM-Specific Instructions
- **Code Regeneration Hints:**
  - Maintain numerical stability in calculations (handle divisions by zero, etc.)
  - Preserve support for different time periods (daily, hourly, etc.)
  - Ensure metrics are properly annualized based on the period frequency
  - Handle edge cases gracefully

- **Contextual Guidance:**
  - The metrics module is critical for evaluating strategy effectiveness
  - Performance metrics should be consistent with industry standards
  - Trade-offs between different metrics should be considered

- **Examples of Extension:**
  - Adding more advanced metrics (Omega ratio, Kappa ratio, etc.)
  - Implementing performance attribution analysis
  - Adding benchmark comparison metrics
  - Developing visualization tools for performance reporting

## 8. Versioning & Change Log
- **Version 1.0:**
  - Initial implementation with basic performance metrics
  - Support for equity curve and trade analysis
  - Key risk-adjusted return calculations

- **Future Roadmap:**
  - Add benchmark comparison metrics
  - Implement rolling window analysis for metrics
  - Add statistical significance testing
  - Develop performance visualization tools 