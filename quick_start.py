#!/usr/bin/env python3
"""
Quick example script to get crypto prices
Run this to test if everything is working!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_fetcher import CryptoDataFetcher

def main():
    print("🚀 Quick Crypto Price Check")
    print("=" * 50)
    
    fetcher = CryptoDataFetcher()
    
    # Get data for all coins
    print("\n📊 Fetching data from CoinGecko...\n")
    data = fetcher.get_all_coins_data(source='coingecko')
    
    # Display results
    for coin, info in data.items():
        print(f"{info['coin']:<10} ${info['price']:>12,.2f}", end="")
        
        if 'price_change_24h' in info and info['price_change_24h'] is not None:
            change = info['price_change_24h']
            symbol = "📈" if change >= 0 else "📉"
            print(f"   {symbol} {change:>+.2f}%")
        else:
            print()
    
    print("\n" + "=" * 50)
    print("✅ Done! Run 'python src/main.py' for continuous monitoring")

if __name__ == "__main__":
    main()
