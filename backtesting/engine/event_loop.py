import queue
import logging
from typing import Callable, Dict, Any, Optional, List, Type
from datetime import datetime

from backtesting.engine.event import Event, EventType


logger = logging.getLogger(__name__)


class EventLoop:
    """
    Event loop for processing events in the backtesting system.
    
    The event loop manages a priority queue of events and dispatches
    them to registered handlers based on event type. Events are
    processed in chronological order according to their timestamps.
    """
    
    def __init__(self):
        """Initialize the event loop."""
        # Priority queue for events, ordered by timestamp
        self.events = queue.PriorityQueue()
        
        # Dictionary of handlers for each event type
        self.handlers: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        
        # Flag to control the event loop
        self.running = False
        
        # Statistics
        self.stats = {
            'processed_events': 0,
            'events_by_type': {event_type: 0 for event_type in EventType},
            'start_time': None,
            'end_time': None
        }
    
    def register_handler(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """
        Register a handler function for a specific event type.
        
        Parameters:
        -----------
        event_type : EventType
            Type of event to handle.
        handler : callable
            Function to call when an event of this type is processed.
            Should accept an Event object as its only argument.
        """
        self.handlers[event_type].append(handler)
        logger.debug(f"Registered handler for event type {event_type}")
    
    def register_handler_all_events(self, handler: Callable[[Event], None]) -> None:
        """
        Register a handler function for all event types.
        
        Parameters:
        -----------
        handler : callable
            Function to call when any event is processed.
            Should accept an Event object as its only argument.
        """
        for event_type in EventType:
            self.handlers[event_type].append(handler)
        logger.debug("Registered handler for all event types")
    
    def add_event(self, event: Event) -> None:
        """
        Add an event to the queue.
        
        Parameters:
        -----------
        event : Event
            Event to add to the queue.
        """
        self.events.put(event)
        logger.debug(f"Added event to queue: {event}")
    
    def add_events(self, events: List[Event]) -> None:
        """
        Add multiple events to the queue.
        
        Parameters:
        -----------
        events : list
            List of events to add to the queue.
        """
        for event in events:
            self.add_event(event)
        logger.debug(f"Added {len(events)} events to queue")
    
    def process_next_event(self) -> Optional[Event]:
        """
        Process the next event in the queue.
        
        Returns:
        --------
        Event or None
            The processed event, or None if the queue is empty.
        """
        if self.events.empty():
            return None
        
        # Get the next event
        event = self.events.get()
        
        # Update statistics
        self.stats['processed_events'] += 1
        self.stats['events_by_type'][event.event_type] += 1
        
        # Call all handlers for this event type
        for handler in self.handlers[event.event_type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
                logger.exception(e)
        
        logger.debug(f"Processed event: {event}")
        return event
    
    def run(self, max_events: Optional[int] = None) -> None:
        """
        Run the event loop until there are no more events or max_events is reached.
        
        Parameters:
        -----------
        max_events : int, optional
            Maximum number of events to process. If None, process all events.
        """
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        event_count = 0
        while self.running and not self.events.empty():
            if max_events is not None and event_count >= max_events:
                break
            
            self.process_next_event()
            event_count += 1
        
        self.stats['end_time'] = datetime.now()
        self.running = False
        
        logger.info(f"Event loop finished. Processed {event_count} events.")
    
    def stop(self) -> None:
        """Stop the event loop."""
        self.running = False
        logger.info("Event loop stopped.")
    
    def clear(self) -> None:
        """Clear all events from the queue."""
        # Create a new empty queue
        self.events = queue.PriorityQueue()
        
        # Reset statistics
        self.stats = {
            'processed_events': 0,
            'events_by_type': {event_type: 0 for event_type in EventType},
            'start_time': None,
            'end_time': None
        }
        
        logger.info("Event queue cleared.")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the event loop.
        
        Returns:
        --------
        dict
            Dictionary of statistics.
        """
        duration = None
        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        return {
            'processed_events': self.stats['processed_events'],
            'events_by_type': {str(k): v for k, v in self.stats['events_by_type'].items()},
            'start_time': self.stats['start_time'],
            'end_time': self.stats['end_time'],
            'duration_seconds': duration,
            'events_in_queue': self.events.qsize()
        }
