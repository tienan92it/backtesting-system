"""
Script to run both the API and UI services for the AI-Driven Backtesting System.
"""

import argparse
import os
import subprocess
import sys
import time
import signal
import threading
from typing import List, Dict

# Define process management variables
processes = []
stop_event = threading.Event()

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run the AI-Driven Backtesting System')
    
    parser.add_argument(
        '--api-host',
        type=str,
        default='localhost',
        help='Host for the API server'
    )
    
    parser.add_argument(
        '--api-port',
        type=int,
        default=8000,
        help='Port for the API server'
    )
    
    parser.add_argument(
        '--ui-port',
        type=int,
        default=8501,
        help='Port for the UI server'
    )
    
    parser.add_argument(
        '--openai-key',
        type=str,
        default=None,
        help='OpenAI API key (if not set in environment)'
    )
    
    return parser.parse_args()

def signal_handler(sig, frame):
    """Handle termination signals."""
    print("\nShutting down all services...")
    stop_event.set()
    
    for process in processes:
        if process.poll() is None:  # If process is still running
            print(f"Terminating {process.args[0]}")
            try:
                process.terminate()
                # Wait a bit for graceful shutdown
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                process.kill()
    
    print("All services stopped.")
    sys.exit(0)

def run_api(host, port, env_vars):
    """
    Run the API server.
    
    Parameters:
    -----------
    host : str
        The host to bind to
    port : int
        The port to listen on
    env_vars : dict
        Environment variables to set
    """
    # Create environment with variables
    env = os.environ.copy()
    env.update(env_vars)
    
    # Command to run the API
    cmd = [
        sys.executable, "-m", "ai_driven.api.run",
        "--host", host,
        "--port", str(port)
    ]
    
    # Run the process
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    processes.append(process)
    
    # Stream output
    def stream_output():
        while not stop_event.is_set():
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(f"[API] {line.strip()}")
        
        # Read any remaining output
        remaining_output, remaining_error = process.communicate()
        if remaining_output:
            print(f"[API] {remaining_output.strip()}")
        if remaining_error:
            print(f"[API ERROR] {remaining_error.strip()}")
    
    # Start output streaming thread
    threading.Thread(target=stream_output, daemon=True).start()
    
    return process

def run_ui(port, api_url, env_vars):
    """
    Run the UI server.
    
    Parameters:
    -----------
    port : int
        The port to listen on
    api_url : str
        The URL of the API server
    env_vars : dict
        Environment variables to set
    """
    # Create environment with variables
    env = os.environ.copy()
    env.update(env_vars)
    env['BACKTEST_API_URL'] = api_url
    
    # Command to run the UI
    cmd = [
        sys.executable, "-m", "ai_driven.ui.run",
        "--port", str(port),
        "--api-url", api_url
    ]
    
    # Run the process
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    processes.append(process)
    
    # Stream output
    def stream_output():
        while not stop_event.is_set():
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(f"[UI] {line.strip()}")
        
        # Read any remaining output
        remaining_output, remaining_error = process.communicate()
        if remaining_output:
            print(f"[UI] {remaining_output.strip()}")
        if remaining_error:
            print(f"[UI ERROR] {remaining_error.strip()}")
    
    # Start output streaming thread
    threading.Thread(target=stream_output, daemon=True).start()
    
    return process

def main():
    """
    Run both the API and UI services.
    """
    args = parse_args()
    
    # Set up environment variables
    env_vars = {}
    if args.openai_key:
        env_vars['OPENAI_API_KEY'] = args.openai_key
    
    # Print header
    print("=" * 80)
    print("AI-Driven Backtesting System")
    print("=" * 80)
    print(f"API will be available at: http://{args.api_host}:{args.api_port}")
    print(f"UI will be available at: http://localhost:{args.ui_port}")
    print("Press Ctrl+C to stop all services")
    print("=" * 80)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the API server
    api_url = f"http://{args.api_host}:{args.api_port}"
    print(f"Starting API server on {api_url}...")
    api_process = run_api(args.api_host, args.api_port, env_vars)
    
    # Wait a moment for the API to start
    time.sleep(2)
    
    # Start the UI server
    print(f"Starting UI server on port {args.ui_port}...")
    ui_process = run_ui(args.ui_port, api_url, env_vars)
    
    # Wait for all processes to complete
    try:
        while True:
            # Check if any process has exited
            for process in processes:
                if process.poll() is not None:
                    # Process has exited
                    if process.returncode != 0:
                        print(f"Process {process.args[0]} exited with code {process.returncode}")
                        # Stop all other processes
                        stop_event.set()
                        for p in processes:
                            if p != process and p.poll() is None:
                                p.terminate()
                        return process.returncode
            
            # Sleep to avoid busy waiting
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle Ctrl+C
        signal_handler(signal.SIGINT, None)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 