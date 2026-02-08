# Crypto Trading Data Fetcher

Real-time cryptocurrency trading data collection system for BTC, ETH, and SOL.

## Features

- 📊 Real-time price, volume, and market data
- 🔄 Multiple data sources (CoinGecko, Binance)
- 💾 Data persistence to CSV/JSON
- 🚀 WebSocket support for live updates
- 📈 Configurable update intervals

## Supported Cryptocurrencies

- BTC (Bitcoin)
- ETH (Ethereum)
- SOL (Solana)

## Installation

```bash
cd ~/code/crypto_trading
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python src/main.py
```

### With Custom Settings

```bash
python src/main.py --coins btc,eth,sol --interval 10 --format csv
```

## Project Structure

```
crypto_trading/
├── config/
│   └── settings.yaml         # Configuration file
├── data/
│   ├── raw/                  # Raw data storage
│   └── processed/            # Processed data
├── logs/
│   └── crypto_fetcher.log    # Log files
├── src/
│   ├── main.py               # Main entry point
│   ├── data_fetcher.py       # Data fetching module
│   ├── websocket_client.py   # WebSocket client
│   └── utils.py              # Utility functions
└── requirements.txt          # Python dependencies
```

## Data Sources

1. **CoinGecko API** - Free tier available
2. **Binance API** - Real-time WebSocket streams

## License

MIT
