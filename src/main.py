#!/usr/bin/env python3
"""
Crypto Trading Data Fetcher - Main Entry Point

Usage:
    python main.py
    python main.py --coins btc,eth,sol --interval 30 --websocket
"""

import argparse
import sys
import time
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_fetcher import CryptoDataFetcher
from websocket_client import AggregatedDataClient
import logging

# Setup logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'crypto_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n👋 Goodbye!")
    sys.exit(0)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Crypto Trading Data Fetcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                              # Use default settings
    python main.py --websocket                  # Real-time WebSocket mode
    python main.py --coins btc,eth --interval 60
    python main.py --source binance
        """
    )
    
    parser.add_argument(
        '--coins',
        type=str,
        default='btc,eth,sol',
        help='Comma-separated list of coins to monitor (default: btc,eth,sol)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Update interval in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='coingecko',
        choices=['coingecko', 'binance'],
        help='Data source to use (default: coingecko)'
    )
    
    parser.add_argument(
        '--websocket',
        action='store_true',
        help='Use WebSocket for real-time updates'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for saving data'
    )
    
    return parser.parse_args()


def run_rest_mode(args):
    """Run in REST API polling mode."""
    coins = [c.strip() for c in args.coins.split(',')]
    fetcher = CryptoDataFetcher()
    fetcher.coins = coins
    
    print(f"🚀 Starting Crypto Trading Data Fetcher")
    print(f"📊 Coins: {', '.join(c.upper() for c in coins)}")
    print(f"⏱️  Interval: {args.interval}s")
    print(f"📡 Source: {args.source}")
    print(f"\nPress Ctrl+C to stop\n")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    while True:
        try:
            data = fetcher.get_all_coins_data(source=args.source)
            fetcher.display_data(data)
            
            # Save to file if specified
            if args.output:
                save_data(data, args.output)
            
            time.sleep(args.interval)
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(5)


def run_websocket_mode(args):
    """Run in WebSocket real-time mode."""
    coins = [c.strip() for c in args.coins.split(',')]
    client = AggregatedDataClient(coins=coins)
    
    signal.signal(signal.SIGINT, signal_handler)
    client.start()


def save_data(data, filepath):
    """Save data to file."""
    import json
    from datetime import datetime
    
    try:
        output = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with open(filepath, 'a') as f:
            f.write(json.dumps(output) + '\n')
            
    except Exception as e:
        logger.error(f"Error saving data: {e}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.websocket:
        run_websocket_mode(args)
    else:
        run_rest_mode(args)


if __name__ == "__main__":
    main()
