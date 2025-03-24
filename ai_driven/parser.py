"""
Strategy Parser Module for the AI-Driven Backtesting System.

This module uses Large Language Models to parse natural language descriptions of trading strategies
into structured representations that can be used to generate executable code.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StrategySpec:
    """Class to hold a structured representation of a trading strategy."""
    name: str = "GeneratedStrategy"
    description: str = ""
    indicators: List[Dict[str, Any]] = field(default_factory=list)
    entry_rules: List[Dict[str, Any]] = field(default_factory=list)
    exit_rules: List[Dict[str, Any]] = field(default_factory=list)
    risk_management: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    is_valid: bool = True
    feedback: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the strategy spec to a dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert the strategy spec to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategySpec':
        """Create a StrategySpec from a dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StrategySpec':
        """Create a StrategySpec from a JSON string."""
        return cls.from_dict(json.loads(json_str))


class StrategyParser:
    """
    LLM-based parser that uses Large Language Models to interpret trading strategy descriptions
    and extract structured components.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM parser.
        
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
            logger.error("OpenAI package is required for LLM-based strategy parsing.")
            logger.error("Please install it with: pip install openai")
            raise ImportError("OpenAI package is required for LLM-based strategy parsing.")
    
    def _construct_prompt(self, description: str) -> str:
        """
        Construct a detailed prompt for the LLM to parse the strategy description.
        
        Parameters:
        -----------
        description : str
            The natural language description of the trading strategy.
            
        Returns:
        --------
        str
            The prompt for the LLM.
        """
        prompt = f"""
You are a sophisticated trading strategy parser. Your task is to analyze a natural language description of a trading strategy and extract its components in a structured format.

I will provide you with a description of a trading strategy. Please parse it and return a JSON object with the following structure:

```json
{{
  "name": "Strategy name extracted from the description or a suitable name if none is provided",
  "description": "The original strategy description text",
  "indicators": [
    {{
      "type": "indicator type (e.g., SMA, EMA, RSI, MACD, Bollinger, Stochastic)",
      "parameters": {{
        "period": 14,  // for indicators like SMA, EMA, RSI
        "k_period": 14,  // for Stochastic
        "d_period": 3,   // for Stochastic
        "slowing": 3,    // for Stochastic
        "fast_period": 12,  // for MACD
        "slow_period": 26,  // for MACD
        "signal_period": 9, // for MACD
        "std_dev": 2.0      // for Bollinger Bands
      }} // Include only relevant parameters for each indicator type
    }}
  ],
  "entry_rules": [
    {{
      "description": "The specific entry rule in natural language",
      "indicator": "the indicator used for the rule (e.g., SMA, RSI, price, volume)",
      "parameters": {{"component": "k"}}, // Optional parameters like Stochastic component
      "condition_type": "condition type (e.g., crosses_above, crosses_below, greater_than, less_than, equals)",
      "target_indicator": "the target indicator to compare against (if applicable)",
      "target_parameters": {{"component": "d"}}, // Optional target parameters
      "target_value": 70  // Numeric value to compare against (if applicable)
    }}
  ],
  "exit_rules": [
    {{
      "description": "The specific exit rule in natural language",
      "indicator": "the indicator used for the rule",
      "parameters": {{}}, // Optional parameters
      "condition_type": "condition type",
      "target_indicator": "the target indicator to compare against (if applicable)",
      "target_parameters": {{}}, // Optional target parameters
      "target_value": 30  // Numeric value to compare against (if applicable)
    }}
  ],
  "risk_management": {{
    "stop_loss": {{"type": "percent", "value": 2.0}},  // If stop loss is specified
    "trailing_stop": {{"type": "percent", "value": 5.0}},  // If trailing stop is specified
    "take_profit": {{"type": "percent", "value": 10.0}},  // If take profit is specified
    "position_size": {{"type": "percent", "value": 20.0}}  // If position size is specified
  }},
  "parameters": {{
    "custom_param1": "value1",
    "custom_param2": "value2"
  }}
}}
```

Please ensure that:
1. You extract all indicators mentioned in the description, with their parameters if specified
2. You identify all entry rules (when to buy/enter a position)
3. You identify all exit rules (when to sell/exit a position)
4. You extract any risk management parameters (stop loss, trailing stop, take profit, position sizing)
5. You extract any custom parameters mentioned

The description may be ambiguous or incomplete in some areas. Use your judgment to interpret the strategy as a trader would, and fill in reasonable defaults where needed.

Remember:
- For indicator parameters, use numeric values only (no strings)
- For condition types, use one of: crosses_above, crosses_below, greater_than, less_than, greater_than_equal, less_than_equal, equals
- The risk management values should be numeric (e.g., 2.0 for 2%)
- If the description mentions "both" conditions, create separate entries for each condition

Here is the strategy description to parse:

"""
        prompt += description
        prompt += """

Please return only valid JSON without any additional text or explanation. Ensure your response is properly formatted and can be parsed with a standard JSON parser.
"""
        return prompt
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API to parse the strategy description.
        
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
                    {"role": "system", "content": "You are an expert trading strategy parser. You extract structured information from natural language strategy descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more deterministic outputs
                response_format={"type": "json_object"},
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    def _validate_and_extract_json(self, llm_response: str) -> Dict[str, Any]:
        """
        Validate and extract the JSON from the LLM response.
        
        Parameters:
        -----------
        llm_response : str
            The response from the LLM.
            
        Returns:
        --------
        Dict[str, Any]
            The extracted JSON as a Python dictionary.
        """
        try:
            # Parse the JSON response
            strategy_data = json.loads(llm_response)
            
            # Basic validation
            required_keys = ['name', 'description', 'indicators', 'entry_rules', 'exit_rules']
            for key in required_keys:
                if key not in strategy_data:
                    logger.warning(f"Missing required key in parsed strategy: {key}")
                    strategy_data[key] = [] if key in ['indicators', 'entry_rules', 'exit_rules'] else ""
            
            return strategy_data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from LLM response: {e}")
            logger.error(f"LLM response: {llm_response}")
            
            # Create a minimal valid structure
            return {
                "name": "ParseError",
                "description": "Error parsing strategy description",
                "indicators": [],
                "entry_rules": [],
                "exit_rules": [],
                "risk_management": {},
                "parameters": {},
                "feedback": f"Error parsing strategy: {str(e)}"
            }
    
    def parse(self, description: str) -> StrategySpec:
        """
        Parse a natural language description of a trading strategy into a structured representation.
        
        Parameters:
        -----------
        description : str
            The natural language description of the trading strategy.
            
        Returns:
        --------
        StrategySpec
            A structured representation of the trading strategy.
        """
        logger.info(f"Parsing strategy: {description}")
        
        try:
            # Construct prompt for the LLM
            prompt = self._construct_prompt(description)
            
            # Call LLM API
            llm_response = self._call_llm_api(prompt)
            
            # Extract and validate the JSON
            strategy_data = self._validate_and_extract_json(llm_response)
            
            # Create a StrategySpec from the data
            spec = StrategySpec(
                name=strategy_data.get('name', 'GeneratedStrategy'),
                description=description,  # Use the original description
                indicators=strategy_data.get('indicators', []),
                entry_rules=strategy_data.get('entry_rules', []),
                exit_rules=strategy_data.get('exit_rules', []),
                risk_management=strategy_data.get('risk_management', {}),
                parameters=strategy_data.get('parameters', {}),
                raw_text=description,
                is_valid=True,
                feedback=""
            )
            
            logger.info(f"Successfully parsed strategy: {spec.name}")
            
            # Log extracted components
            logger.info(f"Extracted indicators: {spec.indicators}")
            logger.info(f"Extracted entry rules: {spec.entry_rules}")
            logger.info(f"Extracted exit rules: {spec.exit_rules}")
            logger.info(f"Extracted risk management: {spec.risk_management}")
            logger.info(f"Extracted parameters: {spec.parameters}")
            
            return spec
            
        except Exception as e:
            logger.error(f"Error parsing strategy: {e}")
            
            # Return an error spec
            return StrategySpec(
                name="ParseError",
                description=description,
                raw_text=description,
                is_valid=False,
                feedback=f"Error parsing strategy: {str(e)}"
            )


def parse_strategy(description: str, debug: bool = False, api_key: Optional[str] = None) -> StrategySpec:
    """
    Parse a natural language description of a trading strategy into a structured representation.
    
    Parameters:
    -----------
    description : str
        The natural language description of the trading strategy.
    debug : bool, optional
        Whether to enable debug logging.
    api_key : str, optional
        The API key for accessing the LLM service.
        If not provided, it will attempt to use the OPENAI_API_KEY environment variable.
        
    Returns:
    --------
    StrategySpec
        A structured representation of the trading strategy.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    
    parser = StrategyParser(api_key=api_key)
    return parser.parse(description)


# Example usage:
if __name__ == "__main__":
    # Test the parser with a simple strategy description
    description = "Buy when the 10-day SMA crosses above the 30-day SMA, and sell when RSI(14) goes above 70. Use a 2% stop loss."
    
    spec = parse_strategy(description, debug=True)
    print("Parsed Strategy:")
    print(spec.to_json()) 