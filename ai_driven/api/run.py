"""
Script to run the FastAPI server for the AI-Driven Backtesting API.
"""

import os
import argparse
import uvicorn

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run the AI-Driven Backtesting API server')
    
    parser.add_argument(
        '--host',
        type=str,
        default="127.0.0.1",
        help='Host to listen on'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to listen on'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    return parser.parse_args()

def main():
    """
    Run the FastAPI server.
    """
    args = parse_args()
    
    print(f"Starting AI-Driven Backtesting API on {args.host}:{args.port}")
    print(f"API documentation will be available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "ai_driven.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main() 