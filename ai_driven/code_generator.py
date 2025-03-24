"""
Strategy Code Generator Module for the AI-Driven Backtesting System.

This module is responsible for generating executable Python code for trading strategies
based on the structured specifications produced by the Strategy Parser Module.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from ai_driven.parser import StrategySpec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StrategyCodeGenerator:
    """
    A generator that uses Large Language Models to create strategy code
    from structured strategy specifications.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM generator.
        
        Parameters:
        -----------
        api_key : str, optional
            The API key for accessing the LLM service. If not provided,
            it will attempt to use the OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        try:
            import openai
        except ImportError:
            logger.error("OpenAI package is required for LLM-based code generation.")
            logger.error("Please install it with: pip install openai")
            raise ImportError("OpenAI package is required for LLM-based code generation.")
    
    def _construct_prompt(self, strategy_spec: StrategySpec) -> str:
        """
        Construct a detailed prompt for the LLM to generate strategy code.
        
        Parameters:
        -----------
        strategy_spec : StrategySpec
            A structured representation of the trading strategy.
            
        Returns:
        --------
        str
            The prompt for the LLM.
        """
        # Convert the strategy spec to JSON for the prompt
        spec_json = json.dumps(strategy_spec.to_dict(), indent=2)
        
        prompt = f"""
You are an expert in generating Python code for trading strategies. Your task is to generate a full Python strategy class that is compatible with our backtesting engine. Let me explain the system:

1. We have a backtesting engine with a Strategy base class that looks like this:

```python
from abc import ABC
import pandas as pd
import numpy as np

class Strategy(ABC):
    def __init__(self):
        # Initialize strategy state
        self.data = None  # DataFrame with OHLCV data
        self.current_index = 0  # Current bar index
        self.position = 0  # Current position (1=long, -1=short, 0=flat)
        self.position_size = 0  # Current position size
        self.cash = 0  # Available cash
        self.portfolio_value = 0  # Total portfolio value
        
    def init(self) -> None:
        # Called once with full data before backtesting
        # Calculate indicators here
        pass
        
    def next(self) -> None:
        # Called for each bar during backtesting
        # Generate trading signals here
        pass
        
    def buy(self, size=None, price=None, limit_price=None, stop_price=None, percent=None):
        # Place a buy order
        pass
        
    def sell(self, size=None, price=None, limit_price=None, stop_price=None, percent=None):
        # Place a sell order
        pass
        
    def get_current_price(self) -> float:
        # Get the current close price
        return self.data.iloc[self.current_index]['close']
        
    def get_current_bar(self) -> pd.Series:
        # Get the current bar (OHLCV)
        return self.data.iloc[self.current_index]
        
    def crossover(self, series1, series2) -> bool:
        # Check if series1 crosses above series2
        if self.current_index <= 0:
            return False
        return series1[self.current_index-1] <= series2[self.current_index-1] and series1[self.current_index] > series2[self.current_index]
        
    def crossunder(self, series1, series2) -> bool:
        # Check if series1 crosses below series2
        if self.current_index <= 0:
            return False
        return series1[self.current_index-1] >= series2[self.current_index-1] and series1[self.current_index] < series2[self.current_index]
```

2. Here is a structured specification of the strategy to implement:

{spec_json}

3. Please generate a complete Python class called "GeneratedStrategy" that inherits from Strategy and implements the strategy according to the specification. Make sure to:

- Begin your code with all necessary imports (e.g., from backtesting.strategy.base import Strategy, import pandas as pd, import numpy as np)
- Calculate all required indicators in the `init()` method
- Implement the entry and exit rules in the `next()` method
- Handle risk management (stop loss, trailing stop, take profit) if specified
- Use proper variable and method naming according to Python conventions
- Include comments to explain the logic
- Always check conditions for NaN/None values to prevent errors
- Ensure the strategy doesn't trade before it has enough data (warmup period)

4. The strategy code should be compatible with pandas and numpy for data manipulation.

5. Your code MUST include the following imports at the top:
   - from backtesting.strategy.base import Strategy
   - import pandas as pd
   - import numpy as np

6. Only include the strategy class definition in your response, without any explanations or instructions.
"""
        return prompt
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API to generate the strategy code.
        
        Parameters:
        -----------
        prompt : str
            The prompt for the LLM.
            
        Returns:
        --------
        str
            The LLM's response.
        """
        import os
        import openai
        
        # Use provided API key or environment variable
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Please provide it as an argument or set the OPENAI_API_KEY environment variable.")
        
        client = openai.OpenAI(api_key=api_key)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",  # Use the latest GPT-4 model
                messages=[
                    {"role": "system", "content": "You are an expert Python programmer specialized in algorithmic trading strategies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for more deterministic outputs
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    def _extract_code(self, llm_response: str) -> str:
        """
        Extract the Python code from the LLM response.
        
        Parameters:
        -----------
        llm_response : str
            The response from the LLM.
            
        Returns:
        --------
        str
            The extracted Python code.
        """
        import re
        
        # Try to find Python code block in the response
        code_match = re.search(r'```(?:python)?(.*?)```', llm_response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # If no code block, use the entire response
            code = llm_response
        
        # Basic validation - check if the code contains the strategy class
        if 'class GeneratedStrategy' not in code:
            logger.warning("Generated code does not contain a GeneratedStrategy class.")
        
        # Validate and add necessary imports if missing
        required_imports = [
            'from backtesting.strategy.base import Strategy',
            'import pandas as pd',
            'import numpy as np'
        ]
        
        # Check if imports are present
        missing_imports = []
        for imp in required_imports:
            if imp not in code:
                missing_imports.append(imp)
                logger.warning(f"Missing import in generated code: {imp}")
        
        # Add missing imports at the beginning of the code
        if missing_imports:
            code = '\n'.join(missing_imports) + '\n\n' + code
            logger.info("Added missing imports to generated code")
            
        return code
    
    def generate(self, strategy_spec: StrategySpec) -> str:
        """
        Generate Python code for a strategy based on the provided specification using an LLM.
        
        Parameters:
        -----------
        strategy_spec : StrategySpec
            A structured representation of the trading strategy.
            
        Returns:
        --------
        str
            The generated Python code for the strategy.
        """
        logger.info(f"Generating code with LLM for strategy: {strategy_spec.name}")
        
        try:
            # Construct prompt for the LLM
            prompt = self._construct_prompt(strategy_spec)
            
            # Call LLM API
            llm_response = self._call_llm_api(prompt)
            
            # Extract code from the response
            code = self._extract_code(llm_response)
            
            logger.info(f"LLM code generation successful for: {strategy_spec.name}")
            return code
        
        except Exception as e:
            logger.error(f"Error during LLM code generation: {e}")
            
            # Return a simple error strategy
            return """
from backtesting.strategy.base import Strategy
import pandas as pd
import numpy as np

class GeneratedStrategy(Strategy):
    '''
    Error Strategy
    
    This strategy was generated due to an error in the code generation process.
    '''
    
    def init(self):
        pass
        
    def next(self):
        pass
"""


def generate_code(strategy_spec: StrategySpec, api_key: Optional[str] = None) -> str:
    """
    Generate Python code for a trading strategy based on the provided strategy specification.
    
    Parameters:
    -----------
    strategy_spec : StrategySpec
        A structured representation of the trading strategy.
    api_key : str, optional
        The API key for accessing the LLM service.
        If not provided, it will attempt to use the OPENAI_API_KEY environment variable.
        
    Returns:
    --------
    str
        The generated Python code for the strategy.
    """
    logger.info(f"Generating code for strategy: {strategy_spec.name}")
    
    try:
        generator = StrategyCodeGenerator(api_key=api_key)
        code = generator.generate(strategy_spec)
        return code
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        # Return a simple error strategy
        code = """
from backtesting.strategy.base import Strategy
import pandas as pd
import numpy as np

class GeneratedStrategy(Strategy):
    '''
    Error Strategy
    
    This strategy was generated due to an error in the code generation process.
    '''
    
    def init(self):
        pass
        
    def next(self):
        pass
"""
    
    return code


# Example usage:
if __name__ == "__main__":
    from ai_driven.parser import parse_strategy
    
    # Test the code generator with a simple strategy description
    description = "Buy when the 10-day SMA crosses above the 30-day SMA, and sell when RSI(14) goes above 70. Use a 2% stop loss."
    
    spec = parse_strategy(description, debug=True)
    code = generate_code(spec)
    
    print("Generated Strategy Code:")
    print(code) 