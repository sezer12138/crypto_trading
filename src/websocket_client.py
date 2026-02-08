"""
WebSocket Client for Real-time Crypto Data
Uses Binance WebSocket API for live price updates
"""

import websocket
import json
import threading
import time
import ssl
from datetime import datetime
from typing import Callable, Dict
import logging

logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """WebSocket client for real-time Binance data."""
    
    def __init__(self, coins: list = None, on_message_callback: Callable = None):
        """Initialize WebSocket client.
        
        Args:
            coins: List of coins to monitor (btc, eth, sol)
            on_message_callback: Callback function for new messages
        """
        self.coins = coins or ['btc', 'eth', 'sol']
        self.on_message_callback = on_message_callback
        self.ws = None
        self.is_running = False
        self.reconnect_interval = 5
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # Symbol mappings
        self.symbols = {
            'btc': 'btcusdt',
            'eth': 'ethusdt',
            'sol': 'solusdt'
        }
    
    def _get_stream_url(self, use_port_443: bool = True) -> str:
        """Generate WebSocket stream URL."""
        streams = "/".join([
            f"{self.symbols[coin.lower()]}@ticker"
            for coin in self.coins
        ])
        # 使用 443 端口避免防火墙问题
        if use_port_443:
            return f"wss://stream.binance.com:443/ws/{streams}"
        return f"wss://stream.binance.com:9443/ws/{streams}"
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            formatted_data = self._format_ticker_data(data)
            
            if self.on_message_callback:
                self.on_message_callback(formatted_data)
            else:
                self._default_display(formatted_data)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_running = False
        
        # Auto-reconnect with limit
        if close_status_code != 1000 and self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            wait_time = min(self.reconnect_interval * self.reconnect_attempts, 60)
            logger.info(f"Reconnecting in {wait_time} seconds... (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
            time.sleep(wait_time)
            self.start()
        elif self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached. Please check your network connection.")
    
    def _on_open(self, ws):
        """Handle WebSocket connection open."""
        logger.info("WebSocket connection established")
        self.is_running = True
        self.reconnect_attempts = 0  # Reset counter on successful connection
    
    def _format_ticker_data(self, data: Dict) -> Dict:
        """Format ticker data from WebSocket."""
        return {
            'coin': data['s'].replace('USDT', ''),
            'symbol': data['s'],
            'price': float(data['c']),
            'price_change': float(data['p']),
            'price_change_percent': float(data['P']),
            'high_24h': float(data['h']),
            'low_24h': float(data['l']),
            'volume_24h': float(data['v']),
            'quote_volume': float(data['q']),
            'open_price': float(data['o']),
            'weighted_avg_price': float(data['w']),
            'trades_count': int(data['n']),
            'timestamp': datetime.now().isoformat(),
            'event_time': data['E']
        }
    
    def _default_display(self, data: Dict):
        """Default display method for WebSocket data."""
        print(f"\n🔴 LIVE | {data['coin']} @ ${data['price']:,.2f}", end="")
        change = data['price_change_percent']
        emoji = "📈" if change >= 0 else "📉"
        print(f" | {emoji} {change:+.2f}% | Vol: ${data['quote_volume']:,.0f}")
    
    def start(self):
        """Start WebSocket connection."""
        websocket.enableTrace(False)
        
        # Create SSL context that works behind firewalls/proxies
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        self.ws = websocket.WebSocketApp(
            self._get_stream_url(use_port_443=True),
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # Run in separate thread with SSL context
        self.ws_thread = threading.Thread(
            target=self.ws.run_forever,
            kwargs={"sslopt": {"context": ssl_context}}
        )
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def stop(self):
        """Stop WebSocket connection."""
        if self.ws:
            self.ws.close()
        self.is_running = False
        logger.info("WebSocket client stopped")


class AggregatedDataClient:
    """Aggregates data from multiple sources."""
    
    def __init__(self, coins: list = None):
        self.coins = coins or ['btc', 'eth', 'sol']
        self.latest_data = {}
        self.ws_client = BinanceWebSocketClient(
            coins=self.coins,
            on_message_callback=self._update_data
        )
    
    def _update_data(self, data: Dict):
        """Update latest data from WebSocket and display."""
        coin = data['coin'].lower()
        self.latest_data[coin] = data
        
        # 实时显示数据
        timestamp = datetime.now().strftime('%H:%M:%S')
        price = data['price']
        change = data['price_change_percent']
        emoji = "🟢" if change >= 0 else "🔴"
        symbol = data['symbol'].replace('USDT', '')
        
        print(f"[{timestamp}] {symbol:5} ${price:>10,.2f} {emoji} {change:>+6.2f}% | "
              f"Vol: ${data['quote_volume']/1e6:>6.2f}M | Trades: {data['trades_count']:,}")
    
    def start(self):
        """Start aggregated client."""
        print("🚀 Starting real-time crypto data stream...")
        print(f"Monitoring: {', '.join(self.coins).upper()}")
        print("Press Ctrl+C to stop\n")
        print(f"{'Time':8} {'Coin':6} {'Price':>12} {'Change':>10} {'Volume':>12} {'Trades':>10}")
        print("-" * 70)
        self.ws_client.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.stop()
    
    def stop(self):
        """Stop aggregated client."""
        self.ws_client.stop()
    
    def get_latest(self, coin: str = None) -> Dict:
        """Get latest data for a specific coin or all coins."""
        if coin:
            return self.latest_data.get(coin.lower())
        return self.latest_data


if __name__ == "__main__":
    # Test WebSocket client
    client = AggregatedDataClient(['btc', 'eth', 'sol'])
    client.start()
