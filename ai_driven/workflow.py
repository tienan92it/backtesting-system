"""
Workflow Orchestration Module for the AI-Driven Backtesting System.

This module combines the different components of the system to provide
a cohesive workflow from natural language strategy description to visual report.
"""

import logging
import time
from typing import Dict, Any, Optional, Union, Tuple
import os

from ai_driven.parser import parse_strategy, StrategySpec
from ai_driven.code_generator import generate_code
from ai_driven.runner import run_backtest, BacktestResult
from ai_driven.report import build_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_ai_backtest_workflow(
    strategy_description: str,
    backtest_config: Dict[str, Any],
    save_artifacts: bool = False,
    artifacts_dir: Optional[str] = "ai_driven/generated",
    api_key: Optional[str] = None
) -> Tuple[BacktestResult, str]:
    """
    Run the complete AI-driven backtesting workflow.
    
    Parameters:
    -----------
    strategy_description : str
        Natural language description of the trading strategy.
    backtest_config : dict
        Configuration for the backtest (symbol, dates, initial capital, etc.).
    save_artifacts : bool, optional
        Whether to save the generated artifacts (strategy code, report) to disk.
    artifacts_dir : str, optional
        Directory to save artifacts if save_artifacts is True.
    api_key : str, optional
        OpenAI API key for strategy parsing and code generation.
        
    Returns:
    --------
    tuple
        (BacktestResult, HTML report string)
    """
    start_time = time.time()
    logger.info("Starting AI-driven backtesting workflow")
    
    try:
        # Step 1: Parse the strategy description
        logger.info("Parsing strategy description")
        strategy_spec = parse_strategy(strategy_description, api_key=api_key)
        
        if not strategy_spec.is_valid:
            error_msg = f"Strategy parsing failed: {strategy_spec.feedback}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Step 2: Generate strategy code
        logger.info("Generating strategy code")
        strategy_code = generate_code(strategy_spec, api_key=api_key)
        
        # Step 3: Save the generated code if requested
        if save_artifacts and artifacts_dir:
            os.makedirs(artifacts_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            code_filename = f"{artifacts_dir}/strategy_{timestamp}.py"
            
            with open(code_filename, 'w') as f:
                f.write(strategy_code)
            logger.info(f"Strategy code saved to {code_filename}")
        
        # Step 4: Run the backtest
        logger.info("Running backtest")
        backtest_result = run_backtest(strategy_code, backtest_config)
        
        # Step 5: Generate the report
        logger.info("Generating backtest report")
        report_html = build_report(backtest_result)
        
        # Step 6: Save the report if requested
        if save_artifacts and artifacts_dir:
            os.makedirs(artifacts_dir, exist_ok=True)
            report_filename = f"{artifacts_dir}/report_{timestamp}.html"
            
            with open(report_filename, 'w') as f:
                f.write(report_html)
            logger.info(f"Backtest report saved to {report_filename}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"AI-driven backtesting workflow completed in {elapsed_time:.2f} seconds")
        
        return backtest_result, report_html
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"AI-driven backtesting workflow failed after {elapsed_time:.2f} seconds: {e}")
        raise 