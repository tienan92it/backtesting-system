"""
Script to run the Streamlit UI for the AI-Driven Backtesting System.
"""

import os
import argparse
import subprocess
import sys

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run the AI-Driven Backtesting UI')
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port to run the Streamlit app on'
    )
    
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8000',
        help='URL of the backend API'
    )
    
    return parser.parse_args()

def main():
    """
    Run the Streamlit app.
    """
    args = parse_args()
    
    # Set environment variables for the Streamlit app
    os.environ['BACKTEST_API_URL'] = args.api_url
    
    # Get the path to the app.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, 'app.py')
    
    print(f"Starting AI-Driven Backtesting UI on port {args.port}")
    print(f"Using API URL: {args.api_url}")
    print(f"App will be available at http://localhost:{args.port}")
    
    # Build the command
    command = [
        'streamlit',
        'run',
        app_path,
        '--server.port',
        str(args.port),
        '--browser.serverAddress',
        'localhost',
    ]
    
    try:
        # Run the Streamlit app
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\nShutting down Streamlit app...")
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 