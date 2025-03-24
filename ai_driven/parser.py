"""
Strategy Parser Module for the AI-Driven Backtesting System.

This module is responsible for parsing natural language descriptions of trading strategies
into structured representations that can be used to generate executable code.
"""

import re
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


class RuleBasedParser:
    """
    A rule-based parser that uses regular expressions to extract strategy components
    from natural language descriptions.
    
    This is a simple implementation that can recognize basic patterns related to 
    common technical indicators and trading conditions.
    """
    
    def __init__(self):
        """Initialize the parser with regex patterns for common indicators and conditions."""
        # Regex patterns for common indicators
        self.indicator_patterns = {
            'SMA': r'(?:(?:(\d+)[\s-]*(?:day|period|bar|candle)?[\s-]*(?:simple[\s-]*)?(?:moving[\s-]*)?average)|(?:(?:simple[\s-]*)?(?:moving[\s-]*)?average[\s-]*\(?(\d+)\)?))',
            'EMA': r'(?:(?:(\d+)[\s-]*(?:day|period|bar|candle)?[\s-]*exponential[\s-]*moving[\s-]*average)|(?:exponential[\s-]*moving[\s-]*average[\s-]*\(?(\d+)\)?))',
            'RSI': r'(?:relative[\s-]*strength[\s-]*index|rsi)[\s-]*\(?(\d+)\)?',
            'MACD': r'macd',
            'Bollinger': r'bollinger[\s-]*bands?'
        }
        
        # Regex patterns for conditions
        self.condition_patterns = {
            'crosses_above': r'cross(?:es)?[\s-]*(?:above|over|up)',
            'crosses_below': r'cross(?:es)?[\s-]*(?:below|under|down)',
            'greater_than': r'(?:>|greater[\s-]*than|above|over|is[\s-]*above|go(?:es)?[\s-]*above)',
            'less_than': r'(?:<|less[\s-]*than|below|under|is[\s-]*below|go(?:es)?[\s-]*below|fall(?:s)?[\s-]*below)',
            'equals': r'(?:=|equals|equal[\s-]*to|is)',
        }
        
        # Special patterns for buy/sell conditions
        self.buy_patterns = [
            r'buy[\s-]*when',
            r'enter[\s-]*(?:long|position)[\s-]*when',
            r'go[\s-]*long[\s-]*when',
            r'buy[\s-]*on',
            r'buy[\s-]*if',
            r'when[\s-]*.*,[\s-]*buy'
        ]
        
        self.sell_patterns = [
            r'sell[\s-]*when',
            r'exit[\s-]*(?:long|position)[\s-]*when',
            r'close[\s-]*(?:long|position)[\s-]*when',
            r'sell[\s-]*on',
            r'sell[\s-]*if',
            r'when[\s-]*.*,[\s-]*sell'
        ]
        
        # Risk management patterns
        self.risk_patterns = {
            'stop_loss': [
                r'(?:use|set|with)[\s-]*(?:a)?[\s-]*(\d+(?:\.\d+)?)%[\s-]*stop[\s-]*loss',
                r'stop[\s-]*loss[\s-]*(?:at|of)[\s-]*(\d+(?:\.\d+)?)%',
                r'(\d+(?:\.\d+)?)%[\s-]*stop[\s-]*loss'
            ],
            'trailing_stop': [
                r'(?:use|set|with)[\s-]*(?:a)?[\s-]*(\d+(?:\.\d+)?)%[\s-]*trailing[\s-]*stop',
                r'trailing[\s-]*stop[\s-]*(?:at|of)[\s-]*(\d+(?:\.\d+)?)%',
                r'(\d+(?:\.\d+)?)%[\s-]*trailing[\s-]*stop'
            ],
            'take_profit': [
                r'(?:use|set|with)[\s-]*(?:a)?[\s-]*(\d+(?:\.\d+)?)%[\s-]*take[\s-]*profit',
                r'take[\s-]*profit[\s-]*(?:at|of)[\s-]*(\d+(?:\.\d+)?)%',
                r'(\d+(?:\.\d+)?)%[\s-]*take[\s-]*profit',
                r'profit[\s-]*target[\s-]*(?:at|of)[\s-]*(\d+(?:\.\d+)?)%',
                r'when[\s-]*(\d+(?:\.\d+)?)%[\s-]*profit[\s-]*is[\s-]*reached'
            ],
            'position_size': [
                r'(?:use|set|with)[\s-]*(?:a)?[\s-]*(\d+(?:\.\d+)?)%[\s-]*position[\s-]*size',
                r'position[\s-]*size[\s-]*(?:at|of)[\s-]*(\d+(?:\.\d+)?)%',
                r'(\d+(?:\.\d+)?)%[\s-]*position[\s-]*size',
                r'position[\s-]*size[\s-]*(?:at|of)[\s-]*(\d+(?:\.\d+)?)%[\s-]*of[\s-]*account',
                r'limit[\s-]*position[\s-]*size[\s-]*to[\s-]*(?:a)?[\s-]*(?:maximum[\s-]*of)?[\s-]*(\d+(?:\.\d+)?)%',
                r'maximum[\s-]*(?:position|allocation)[\s-]*(?:of|at)[\s-]*(\d+(?:\.\d+)?)%'
            ]
        }
    
    def extract_indicators(self, text: str) -> List[Dict[str, Any]]:
        """Extract indicators mentioned in the text."""
        indicators = []
        
        for indicator_name, pattern in self.indicator_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                indicator = {
                    'type': indicator_name,
                    'parameters': {}
                }
                
                # Extract period parameter if available
                if match.groups():
                    # Handle patterns with multiple capture groups
                    for group in match.groups():
                        if group and group.isdigit():
                            indicator['parameters']['period'] = int(group)
                            break
                
                indicators.append(indicator)
        
        return indicators
    
    def extract_numeric_value(self, text: str) -> Optional[float]:
        """Extract a numeric value from text."""
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))
        return None
    
    def extract_indicator_periods(self, text: str, indicator_type: str) -> List[int]:
        """Extract indicator periods mentioned in the text."""
        periods = []
        
        if indicator_type in self.indicator_patterns:
            pattern = self.indicator_patterns[indicator_type]
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if match.groups():
                    for group in match.groups():
                        if group and group.isdigit():
                            periods.append(int(group))
        
        return periods
    
    def extract_rules(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entry and exit rules from the text."""
        entry_rules = []
        exit_rules = []
        
        # Split text into sentences or clauses
        sentences = re.split(r'[.;]', text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            logger.debug(f"Processing sentence: {sentence.strip()}")
            
            # Check if this is a buy/entry rule
            is_buy_rule = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in self.buy_patterns)
            
            # Check if this is a sell/exit rule
            is_sell_rule = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in self.sell_patterns)
            
            if not is_buy_rule and not is_sell_rule:
                logger.debug(f"No buy/sell pattern found in: {sentence.strip()}")
                
                # Check if this sentence contains "buy" or "sell" without the specific patterns
                if re.search(r'\bbuy\b', sentence, re.IGNORECASE):
                    is_buy_rule = True
                    logger.debug(f"Found standalone 'buy' keyword")
                elif re.search(r'\bsell\b', sentence, re.IGNORECASE):
                    is_sell_rule = True
                    logger.debug(f"Found standalone 'sell' keyword")
                else:
                    continue
            
            # Extract condition type
            condition_type = None
            for cond_name, pattern in self.condition_patterns.items():
                if re.search(pattern, sentence, re.IGNORECASE):
                    condition_type = cond_name
                    logger.debug(f"Found condition type: {condition_type}")
                    break
            
            # Create rule dictionary
            rule = {
                'description': sentence.strip(),
                'condition_type': condition_type if condition_type else 'custom'
            }
            
            # Check for indicators in the rule
            for indicator_name, pattern in self.indicator_patterns.items():
                if re.search(pattern, sentence, re.IGNORECASE):
                    rule['indicator'] = indicator_name
                    logger.debug(f"Found indicator in rule: {indicator_name}")
                    
                    # Extract periods for this indicator
                    periods = self.extract_indicator_periods(sentence, indicator_name)
                    if periods:
                        rule['period'] = periods[0]
                        if len(periods) > 1:
                            rule['target_period'] = periods[1]
                        logger.debug(f"Extracted periods: {periods}")
                    
                    break
            
            # Add threshold value if available
            values = re.findall(r'(\d+(?:\.\d+)?)', sentence)
            if values:
                numeric_values = [float(val) for val in values]
                logger.debug(f"Found numeric values: {numeric_values}")
                
                # Skip values that are likely indicator periods
                if 'period' in rule and rule['period'] in numeric_values:
                    numeric_values.remove(rule['period'])
                if 'target_period' in rule and rule['target_period'] in numeric_values:
                    numeric_values.remove(rule['target_period'])
                
                if numeric_values:
                    rule['target_value'] = numeric_values[0]
                    if len(numeric_values) > 1:
                        rule['secondary_value'] = numeric_values[1]
            
            # Add to appropriate rules list
            if is_buy_rule:
                entry_rules.append(rule)
                logger.debug(f"Added entry rule: {rule}")
            elif is_sell_rule:
                exit_rules.append(rule)
                logger.debug(f"Added exit rule: {rule}")
        
        return {
            'entry_rules': entry_rules,
            'exit_rules': exit_rules
        }
    
    def extract_risk_management(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract risk management parameters from the text."""
        risk_management = {}
        
        for risk_type, patterns in self.risk_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and match.group(1):
                    value = float(match.group(1))
                    if risk_type not in risk_management:
                        risk_management[risk_type] = {
                            'type': 'percent',
                            'value': value
                        }
                        logger.debug(f"Extracted {risk_type}: {value}%")
                    break
        
        return risk_management
    
    def extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract any other strategy parameters from the text."""
        parameters = {}
        
        # Extract timeframe
        timeframe_match = re.search(r'(daily|weekly|monthly|hourly|[1-9][0-9]*[mhdw])', text, re.IGNORECASE)
        if timeframe_match:
            parameters['timeframe'] = timeframe_match.group(1).lower()
            logger.debug(f"Extracted timeframe: {parameters['timeframe']}")
        
        # Add any other parameter extraction logic here
        
        return parameters
    
    def deduplicate_indicators(self, indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate indicator entries."""
        unique_indicators = []
        seen_indicators = set()
        
        for indicator in indicators:
            ind_type = indicator['type']
            period = indicator['parameters'].get('period', 'default')
            
            key = f"{ind_type}_{period}"
            if key not in seen_indicators:
                seen_indicators.add(key)
                unique_indicators.append(indicator)
        
        return unique_indicators
    
    def parse(self, description: str) -> StrategySpec:
        """
        Parse a natural language strategy description into a structured StrategySpec.
        
        Parameters:
        -----------
        description : str
            The natural language description of the trading strategy.
            
        Returns:
        --------
        StrategySpec
            A structured representation of the strategy.
        """
        logger.info(f"Parsing strategy description: {description}")
        
        # Extract strategy name if provided
        name_match = re.search(r'strategy[\s-]*name[\s-]*[:=][\s-]*([a-zA-Z0-9_\s]+)', description, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else "GeneratedStrategy"
        logger.debug(f"Extracted strategy name: {name}")
        
        # Extract indicators
        indicators = self.extract_indicators(description)
        indicators = self.deduplicate_indicators(indicators)
        logger.info(f"Extracted indicators: {indicators}")
        
        # Extract rules
        rules = self.extract_rules(description)
        entry_rules = rules['entry_rules']
        exit_rules = rules['exit_rules']
        logger.info(f"Extracted entry rules: {entry_rules}")
        logger.info(f"Extracted exit rules: {exit_rules}")
        
        # Extract risk management
        risk_management = self.extract_risk_management(description)
        logger.info(f"Extracted risk management: {risk_management}")
        
        # Extract other parameters
        parameters = self.extract_parameters(description)
        logger.info(f"Extracted parameters: {parameters}")
        
        # Create and return the StrategySpec
        spec = StrategySpec(
            name=name,
            description=description,
            indicators=indicators,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            risk_management=risk_management,
            parameters=parameters,
            raw_text=description
        )
        
        return spec


class LLMParser:
    """
    A parser that uses Large Language Models to extract strategy components
    from natural language descriptions.
    
    This implementation can handle more complex and nuanced strategy descriptions
    compared to the rule-based parser.
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
            logger.error("OpenAI package is required for LLM-based parsing.")
            logger.error("Please install it with: pip install openai")
            raise ImportError("OpenAI package is required for LLM-based parsing.")
    
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
        prompt = """
You are an expert trading strategy parser. Your task is to parse the natural language description 
of a trading strategy into a structured format that can be used for backtesting.

Please analyze the following strategy description and extract the following components:

1. Strategy name
2. Technical indicators with their parameters (e.g., SMA(10), RSI(14), MACD(12,26,9), Bollinger Bands(20,2))
3. Entry rules (when to buy)
4. Exit rules (when to sell)
5. Risk management parameters (stop loss, trailing stop, take profit, position sizing)
6. Other parameters (e.g., timeframe)

If any component is missing or unclear, use reasonable defaults or mark it as missing.
If the input is not a trading strategy or doesn't contain enough information, provide appropriate feedback.

Please format your response as a valid JSON object with the following structure:
```json
{
  "name": "Strategy name",
  "indicators": [
    {
      "type": "SMA",
      "parameters": {
        "period": 10
      }
    },
    {
      "type": "RSI",
      "parameters": {
        "period": 14
      }
    }
  ],
  "entry_rules": [
    {
      "description": "Description of when to buy",
      "condition_type": "crosses_above", 
      "indicator": "SMA",
      "target_value": 50,
      "period": 10,
      "target_period": 20
    }
  ],
  "exit_rules": [
    {
      "description": "Description of when to sell",
      "condition_type": "crosses_below",
      "indicator": "RSI",
      "target_value": 70,
      "period": 14
    }
  ],
  "risk_management": {
    "stop_loss": {
      "type": "percent",
      "value": 2.0
    },
    "trailing_stop": {
      "type": "percent",
      "value": 1.0
    },
    "take_profit": {
      "type": "percent",
      "value": 5.0
    },
    "position_size": {
      "type": "percent",
      "value": 10.0
    }
  },
  "parameters": {
    "timeframe": "daily"
  },
  "is_valid": true,
  "feedback": "The strategy description is complete and valid."
}
```

Condition types can be: "crosses_above", "crosses_below", "greater_than", "less_than", "equals", or "custom".

If you cannot parse the input as a trading strategy or it lacks critical information, set "is_valid" to false
and provide appropriate feedback. Only include components that are explicitly or implicitly mentioned in the description.

Strategy description to parse:
"""
        prompt += description
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
                model="gpt-4",  # Use GPT-4 for better parsing accuracy
                messages=[
                    {"role": "system", "content": "You are an expert trading strategy parser that extracts structured data from natural language descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for more deterministic outputs
                max_tokens=2000,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    def _validate_and_extract_json(self, llm_response: str) -> Dict[str, Any]:
        """
        Extract and validate the JSON response from the LLM.
        
        Parameters:
        -----------
        llm_response : str
            The response from the LLM.
            
        Returns:
        --------
        Dict[str, Any]
            The extracted JSON data.
        """
        # Extract JSON from the response
        import json
        import re
        
        # Try to find JSON block in the response
        json_match = re.search(r'```(?:json)?(.*?)```', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # If no code block, try to parse the entire response
            json_str = llm_response
        
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response as JSON: {e}")
            logger.error(f"LLM response: {llm_response}")
            
            # Return a default structure with error feedback
            return {
                "name": "ParseError",
                "indicators": [],
                "entry_rules": [],
                "exit_rules": [],
                "risk_management": {},
                "parameters": {},
                "is_valid": False,
                "feedback": f"Failed to parse LLM response as JSON: {e}"
            }
    
    def _fallback_to_rule_based(self, description: str) -> StrategySpec:
        """
        Fallback to rule-based parsing if LLM parsing fails.
        
        Parameters:
        -----------
        description : str
            The natural language description of the trading strategy.
            
        Returns:
        --------
        StrategySpec
            A structured representation of the strategy.
        """
        logger.warning("Falling back to rule-based parsing due to LLM parsing failure.")
        rule_based_parser = RuleBasedParser()
        return rule_based_parser.parse(description)
    
    def parse(self, description: str) -> StrategySpec:
        """
        Parse a natural language strategy description into a structured StrategySpec using an LLM.
        
        Parameters:
        -----------
        description : str
            The natural language description of the trading strategy.
            
        Returns:
        --------
        StrategySpec
            A structured representation of the strategy.
        """
        logger.info(f"Parsing strategy with LLM: {description}")
        
        if not description or not description.strip():
            logger.warning("Empty strategy description provided.")
            return StrategySpec(
                name="InvalidStrategy",
                description="",
                raw_text="",
                is_valid=False,
                feedback="Empty strategy description provided."
            )
        
        # Construct prompt for the LLM
        prompt = self._construct_prompt(description)
        
        try:
            # Call LLM API
            llm_response = self._call_llm_api(prompt)
            
            # Extract and validate JSON
            parsed_data = self._validate_and_extract_json(llm_response)
            
            # Check if the parsing was successful
            if parsed_data.get("is_valid", True):
                # Create StrategySpec from parsed data
                spec = StrategySpec(
                    name=parsed_data.get("name", "GeneratedStrategy"),
                    description=description,
                    indicators=parsed_data.get("indicators", []),
                    entry_rules=parsed_data.get("entry_rules", []),
                    exit_rules=parsed_data.get("exit_rules", []),
                    risk_management=parsed_data.get("risk_management", {}),
                    parameters=parsed_data.get("parameters", {}),
                    raw_text=description
                )
                
                # Add any feedback from the LLM
                if "feedback" in parsed_data:
                    spec.feedback = parsed_data["feedback"]
                
                logger.info(f"LLM parsing successful for: {description}")
                return spec
            else:
                # LLM indicated parsing failure
                logger.warning(f"LLM parsing indicated failure: {parsed_data.get('feedback', 'No feedback provided')}")
                
                # Fallback to rule-based parser
                return self._fallback_to_rule_based(description)
        
        except Exception as e:
            logger.error(f"Error during LLM parsing: {e}")
            
            # Fallback to rule-based parser
            return self._fallback_to_rule_based(description)


def parse_strategy(description: str, use_llm: bool = False, debug: bool = False, api_key: Optional[str] = None) -> StrategySpec:
    """
    Parse a natural language strategy description into a structured form.
    
    Parameters:
    -----------
    description : str
        The natural language description of the trading strategy.
    use_llm : bool, optional
        Whether to use an LLM for parsing instead of rule-based parsing.
    debug : bool, optional
        Whether to enable debug logging.
    api_key : str, optional
        The API key for accessing the LLM service. Only used if use_llm is True.
        If not provided when use_llm is True, it will attempt to use the OPENAI_API_KEY environment variable.
        
    Returns:
    --------
    StrategySpec
        A structured representation of the strategy.
    """
    logger.info(f"Parsing strategy: {description}")
    
    # Set debug level if requested
    if debug:
        logger.setLevel(logging.DEBUG)
    
    try:
        if use_llm:
            try:
                # Use LLM-based parser
                logger.info("Using LLM-based parser")
                parser = LLMParser(api_key=api_key)
                spec = parser.parse(description)
            except ImportError:
                # Fallback to rule-based parser if OpenAI package is not installed
                logger.warning("OpenAI package not found. Falling back to rule-based parser.")
                parser = RuleBasedParser()
                spec = parser.parse(description)
            except Exception as e:
                logger.error(f"Error using LLM-based parser: {e}. Falling back to rule-based parser.")
                parser = RuleBasedParser()
                spec = parser.parse(description)
        else:
            # Use rule-based parser
            logger.info("Using rule-based parser")
            parser = RuleBasedParser()
            spec = parser.parse(description)
    except Exception as e:
        logger.error(f"Error parsing strategy: {e}")
        # Return an empty spec with error indication
        spec = StrategySpec(
            name="ParseError",
            description=description,
            raw_text=description,
            feedback=f"Error parsing strategy: {str(e)}"
        )
    
    # Reset log level
    if debug:
        logger.setLevel(logging.INFO)
    
    return spec


# Example usage:
if __name__ == "__main__":
    # Test the parser with a simple strategy description
    description = "Buy when the 10-day SMA crosses above the 30-day SMA, and sell when RSI(14) goes above 70. Use a 2% stop loss."
    
    spec = parse_strategy(description, debug=True)
    print("Parsed Strategy:")
    print(spec.to_json()) 