import logging
import uuid
from typing import Dict, List, Any, Optional, Type, Union
from datetime import datetime
import pandas as pd

from backtesting.engine.event import (
    EventType, Event, MarketEvent, SignalEvent, 
    OrderEvent, FillEvent, PortfolioEvent
)
from backtesting.engine.event_loop import EventLoop
from backtesting.metrics.performance import (
    calculate_performance_metrics,
    calculate_trade_metrics,
    calculate_returns
)


logger = logging.getLogger(__name__)


class Backtester:
    """
    Main backtesting engine that coordinates the flow of data, strategy signals,
    order execution, and portfolio updates.
    
    The Backtester connects all components of the system:
    - Data Handler: Provides historical market data
    - Strategy: Generates trading signals based on market data
    - Portfolio: Tracks positions, cash, and overall performance
    - Execution Handler: Simulates order execution with fees and slippage
    
    It uses the event-driven architecture to coordinate these components.
    """
    
    def __init__(
        self,
        data_handler,
        strategy,
        portfolio,
        execution_handler,
        initial_capital: float = 100000.0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        commission: float = 0.001,  # 0.1% default commission
        slippage: float = 0.0,      # No slippage by default
    ):
        """
        Initialize the backtester with all required components.
        
        Parameters:
        -----------
        data_handler : DataHandler
            Component that provides historical market data.
        strategy : Strategy
            Trading strategy that generates signals.
        portfolio : Portfolio
            Tracks positions, cash, and performance.
        execution_handler : ExecutionHandler
            Simulates order execution.
        initial_capital : float, optional
            Starting capital for the portfolio.
        start_date : datetime, optional
            Start date for the backtest.
        end_date : datetime, optional
            End date for the backtest.
        commission : float, optional
            Commission rate as a decimal (e.g., 0.001 for 0.1%).
        slippage : float, optional
            Slippage model as a decimal (e.g., 0.001 for 0.1%).
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.commission = commission
        self.slippage = slippage
        
        # Create the event loop
        self.event_loop = EventLoop()
        
        # Set up event handlers
        self._register_event_handlers()
        
        # Statistics and results
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_bars': 0,
            'signals_generated': 0,
            'orders_placed': 0,
            'orders_filled': 0,
        }
        
        # Results storage
        self.results = {
            'equity_curve': [],
            'trades': [],
            'positions': [],
            'returns': [],
        }
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Backtester initialized")
    
    def _initialize_components(self):
        """Initialize all components with necessary references."""
        # Set up the data handler
        self.data_handler.initialize(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Set up the portfolio
        self.portfolio.initialize(
            initial_capital=self.initial_capital,
            data_handler=self.data_handler
        )
        
        # Set up the strategy
        self.strategy.initialize(
            data_handler=self.data_handler,
            portfolio=self.portfolio,
            event_loop=self.event_loop
        )
        
        # Set up the execution handler
        self.execution_handler.initialize(
            data_handler=self.data_handler,
            commission=self.commission,
            slippage=self.slippage
        )
    
    def _register_event_handlers(self):
        """Register handlers for different event types."""
        # Market event handlers
        self.event_loop.register_handler(
            EventType.MARKET,
            self._on_market_event
        )
        
        # Signal event handlers
        self.event_loop.register_handler(
            EventType.SIGNAL,
            self._on_signal_event
        )
        
        # Order event handlers
        self.event_loop.register_handler(
            EventType.ORDER,
            self._on_order_event
        )
        
        # Fill event handlers
        self.event_loop.register_handler(
            EventType.FILL,
            self._on_fill_event
        )
        
        # Portfolio event handlers
        self.event_loop.register_handler(
            EventType.PORTFOLIO,
            self._on_portfolio_event
        )
    
    def _on_market_event(self, event: MarketEvent):
        """
        Handle market events by updating the strategy.
        
        Parameters:
        -----------
        event : MarketEvent
            The market event to handle.
        """
        logger.debug(f"Processing market event: {event}")
        
        try:
            # Update the data handler's current bar and datetime
            self.data_handler.current_bar[event.symbol] = event.data
            self.data_handler.current_datetime = event.timestamp
            
            # Increment the current index to track position in the data
            self.data_handler.current_index += 1
            
            # Update the strategy's current index and call next()
            self.strategy.current_index = self.data_handler.current_index - 1
            self.strategy.next()
        except Exception as e:
            logger.error(f"Error processing market event: {e}")
        
        # Update statistics
        self.stats['total_bars'] += 1
    
    def _on_signal_event(self, event: SignalEvent):
        """
        Handle signal events by creating orders.
        
        Parameters:
        -----------
        event : SignalEvent
            The signal event to handle.
        """
        logger.debug(f"Processing signal event: {event}")
        
        # Create an order based on the signal
        order = self.portfolio.on_signal(event)
        
        # If an order was created, add it to the event loop
        if order is not None:
            self.event_loop.add_event(order)
        
        # Update statistics
        self.stats['signals_generated'] += 1
    
    def _on_order_event(self, event: OrderEvent):
        """
        Handle order events by executing them.
        
        Parameters:
        -----------
        event : OrderEvent
            The order event to handle.
        """
        logger.debug(f"Processing order event: {event}")
        
        # Execute the order
        fill = self.execution_handler.execute_order(event)
        
        # If the order was filled, add the fill event to the event loop
        if fill is not None:
            self.event_loop.add_event(fill)
        
        # Update statistics
        self.stats['orders_placed'] += 1
    
    def _on_fill_event(self, event: FillEvent):
        """
        Handle fill events by updating the portfolio.
        
        Parameters:
        -----------
        event : FillEvent
            The fill event to handle.
        """
        logger.debug(f"Processing fill event: {event}")
        
        # Update the portfolio with the fill
        portfolio_event = self.portfolio.on_fill(event)
        
        # If a portfolio event was created, add it to the event loop
        if portfolio_event is not None:
            self.event_loop.add_event(portfolio_event)
        
        # Update statistics
        self.stats['orders_filled'] += 1
        
        # Record the trade
        self.results['trades'].append({
            'timestamp': event.timestamp,
            'symbol': event.symbol,
            'direction': event.direction,
            'quantity': event.quantity,
            'price': event.fill_price,
            'commission': event.commission
        })
    
    def _on_portfolio_event(self, event: PortfolioEvent):
        """
        Handle portfolio events by updating results.
        
        Parameters:
        -----------
        event : PortfolioEvent
            The portfolio event to handle.
        """
        logger.debug(f"Processing portfolio event: {event}")
        
        # Record portfolio state
        self.results['equity_curve'].append({
            'timestamp': event.timestamp,
            'portfolio_value': event.portfolio_value,
            'cash': event.cash
        })
        
        # Record positions
        self.results['positions'].append({
            'timestamp': event.timestamp,
            'positions': event.positions.copy()
        })
    
    def run(self):
        """
        Run the backtest.
        
        This method starts the data handler to generate market events,
        which then flow through the event-driven system.
        """
        logger.info("Starting backtest")
        self.stats['start_time'] = datetime.now()
        
        # Generate market events from the data handler
        self._generate_market_events()
        
        # Log event count before running
        total_events = self.event_loop.events.qsize()
        logger.info(f"Generated {total_events} market events")
        
        # Run the event loop
        self.event_loop.run()
        
        # Calculate final results
        self._calculate_results()
        
        self.stats['end_time'] = datetime.now()
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        logger.info(f"Backtest completed in {duration:.2f} seconds")
        logger.info(f"Processed {self.stats['total_bars']} bars")
        logger.info(f"Generated {self.stats['signals_generated']} signals")
        logger.info(f"Placed {self.stats['orders_placed']} orders")
        logger.info(f"Filled {self.stats['orders_filled']} orders")
        
        # Log results counts
        equity_count = len(self.results['equity_curve']) if 'equity_curve' in self.results and self.results['equity_curve'] else 0
        trades_count = len(self.results['trades']) if 'trades' in self.results and self.results['trades'] else 0
        logger.info(f"Final equity curve entries: {equity_count}")
        logger.info(f"Final trades count: {trades_count}")
        
        return self.results
    
    def _generate_market_events(self):
        """Generate market events from the data handler."""
        try:
            # Get all data from the data handler
            data = self.data_handler.get_all_bars()
            
            # Create market events for each bar
            for timestamp, bars in data.items():
                for symbol, bar_data in bars.items():
                    # Create a market event
                    event = MarketEvent(
                        timestamp=timestamp,
                        symbol=symbol,
                        data=bar_data
                    )
                    
                    # Add the event to the event loop
                    self.event_loop.add_event(event)
        except Exception as e:
            logger.error(f"Error generating market events: {e}")
            # Fallback to using the data property if get_all_bars fails
            if hasattr(self.data_handler, 'data') and self.data_handler.data:
                for symbol, data_df in self.data_handler.data.items():
                    for timestamp, row in data_df.iterrows():
                        event = MarketEvent(
                            timestamp=timestamp,
                            symbol=symbol,
                            data=row.to_dict()
                        )
                        self.event_loop.add_event(event)
            else:
                logger.error("Could not generate market events from data handler.")
    
    def _calculate_results(self):
        """Calculate final performance metrics."""
        # Convert results to DataFrames for easier analysis
        if self.results['equity_curve']:
            equity_df = pd.DataFrame(self.results['equity_curve'])
            equity_df.set_index('timestamp', inplace=True)
            equity_df.sort_index(inplace=True)
            
            # Calculate returns
            equity_df['returns'] = equity_df['portfolio_value'].pct_change()
            
            # Store returns for further analysis
            self.results['returns'] = equity_df['returns'].dropna()
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics from the equity curve and trades."""
        if 'equity_curve' not in self.results or not self.results['equity_curve']:
            logger.warning("No equity curve available to calculate performance metrics")
            return
        
        # If equity_curve is a list, convert it to DataFrame
        if isinstance(self.results['equity_curve'], list):
            if not self.results['equity_curve']:  # Check if list is empty
                logger.warning("Empty equity curve list, cannot calculate performance metrics")
                return
                
            equity_df = pd.DataFrame(self.results['equity_curve'])
            if equity_df.empty:
                logger.warning("Empty equity DataFrame, cannot calculate performance metrics")
                return
                
            equity_df.set_index('timestamp', inplace=True)
            equity_curve = equity_df['portfolio_value'] if 'portfolio_value' in equity_df.columns else None
        else:
            # Assume it's already a DataFrame
            equity_curve = self.results['equity_curve']['portfolio_value']
        
        if equity_curve is None or len(equity_curve) == 0:
            logger.warning("Empty equity curve, cannot calculate performance metrics")
            return
        
        # Calculate performance metrics
        if 'trades' in self.results and self.results['trades']:
            # If trades is a list, convert it to DataFrame
            if isinstance(self.results['trades'], list):
                if not self.results['trades']:  # Check if list is empty
                    logger.warning("No trades available to calculate trade metrics")
                    trades_df = pd.DataFrame()
                else:
                    trades_df = pd.DataFrame(self.results['trades'])
            else:
                trades_df = self.results['trades']
            
            if not trades_df.empty:
                try:
                    # Calculate trade metrics
                    self.results['trade_metrics'] = calculate_trade_metrics(trades_df)
                    
                    # Calculate comprehensive performance metrics
                    self.results['performance_metrics'] = calculate_performance_metrics(
                        equity_curve=equity_curve,
                        trades=trades_df,
                        risk_free_rate=0.0,  # Assuming 0% risk-free rate
                        periods_per_year=252  # Trading days in a year
                    )
                except Exception as e:
                    logger.error(f"Error calculating performance metrics: {e}")
                    # Provide basic metrics as fallback
                    self._calculate_basic_metrics(equity_curve)
            else:
                # If no trades, calculate basic metrics
                self._calculate_basic_metrics(equity_curve)
        else:
            # If no trades, calculate basic metrics
            self._calculate_basic_metrics(equity_curve)
    
    def _calculate_basic_metrics(self, equity_curve):
        """Calculate basic metrics when trades are not available."""
        try:
            returns = calculate_returns(equity_curve)
            
            # Calculate total return
            total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1 if len(equity_curve) > 0 else 0.0
            
            # Store performance metrics
            self.results['performance_metrics'] = {
                'total_return': total_return,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            self.results['performance_metrics'] = {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
    
    def get_results(self):
        """
        Get the results of the backtest.
        
        Returns:
        --------
        dict
            Dictionary containing the results of the backtest.
        """
        return self.results
    
    def get_stats(self):
        """
        Get statistics about the backtest run.
        
        Returns:
        --------
        dict
            Dictionary containing statistics about the backtest run.
        """
        return self.stats
    
    def get_equity_curve(self):
        """
        Get the equity curve as a DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the equity curve.
        """
        if self.results['equity_curve']:
            equity_df = pd.DataFrame(self.results['equity_curve'])
            equity_df.set_index('timestamp', inplace=True)
            equity_df.sort_index(inplace=True)
            return equity_df
        return pd.DataFrame()
    
    def get_trades(self):
        """
        Get the trades as a DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the trades.
        """
        if self.results['trades']:
            trades_df = pd.DataFrame(self.results['trades'])
            trades_df.set_index('timestamp', inplace=True)
            trades_df.sort_index(inplace=True)
            return trades_df
        return pd.DataFrame()
    
    def get_positions(self):
        """
        Get the positions history as a DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the positions history.
        """
        if self.results['positions']:
            positions_df = pd.DataFrame(self.results['positions'])
            positions_df.set_index('timestamp', inplace=True)
            positions_df.sort_index(inplace=True)
            return positions_df
        return pd.DataFrame()
    
    def get_returns(self):
        """
        Get the returns as a Series.
        
        Returns:
        --------
        pd.Series
            Series containing the returns.
        """
        return self.results.get('returns', pd.Series())
