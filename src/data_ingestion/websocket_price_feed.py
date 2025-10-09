#!/usr/bin/env python3
"""
WebSocket Real-Time Price Feed
Connects to Coinbase WebSocket for live price updates
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Callable, Optional, List
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class WebSocketPriceFeed:
    """Real-time price feed using Coinbase WebSocket"""
    
    def __init__(self, symbols: list = None, update_callback: Callable = None):
        self.symbols = symbols or ['BTC-USD']
        self.update_callback = update_callback
        self.running = False
        self.websocket = None
        self.latest_prices = {}
        self.connection_thread = None
        
    async def connect(self):
        """Connect to Coinbase WebSocket"""
        uri = "wss://ws-feed.exchange.coinbase.com"
        
        try:
            self.websocket = await websockets.connect(uri)
            logger.info("Connected to Coinbase WebSocket")
            
            # Subscribe to ticker updates
            subscribe_message = {
                "type": "subscribe",
                "product_ids": self.symbols,
                "channels": ["ticker"]
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to {self.symbols}")
            
            # Listen for messages
            async for message in self.websocket:
                await self._handle_message(message)
                
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            raise
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'ticker':
                symbol = data.get('product_id', '').replace('-USD', '')
                price = float(data.get('price', 0))
                timestamp = datetime.now(timezone.utc)
                
                # Update latest prices
                self.latest_prices[symbol] = {
                    'price': price,
                    'timestamp': timestamp,
                    'bid': float(data.get('best_bid', price)),
                    'ask': float(data.get('best_ask', price)),
                    'volume_24h': float(data.get('volume_24h', 0))
                }
                
                # Call update callback if provided
                if self.update_callback:
                    self.update_callback(symbol, self.latest_prices[symbol])
                
                logger.debug(f"Updated {symbol}: ${price:.2f}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def start(self):
        """Start WebSocket connection in background thread"""
        if self.running:
            return
        
        self.running = True
        self.connection_thread = threading.Thread(target=self._run_async, daemon=True)
        self.connection_thread.start()
        logger.info("Started WebSocket price feed")
    
    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.websocket:
            asyncio.run_coroutine_threadsafe(self.websocket.close(), asyncio.get_event_loop())
        logger.info("Stopped WebSocket price feed")
    
    def _run_async(self):
        """Run async WebSocket connection in thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.connect())
        except Exception as e:
            logger.error(f"WebSocket thread error: {e}")
        finally:
            loop.close()
    
    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """Get latest price for a symbol"""
        return self.latest_prices.get(symbol)
    
    def get_all_prices(self) -> Dict[str, Dict]:
        """Get all latest prices"""
        return self.latest_prices.copy()
    
    def collect_and_save_prices(self, symbols: List[str], data_dir: Path) -> Dict[str, Dict]:
        """Collect and save real-time prices (compatibility method)"""
        try:
            results = {}
            for symbol in symbols:
                if symbol in self.latest_prices:
                    results[symbol] = self.latest_prices[symbol]
                else:
                    # Fallback to spot price if not in WebSocket data
                    results[symbol] = self.get_spot_price(f"{symbol}-USD")
            return results
        except Exception as e:
            logger.error(f"Error collecting real-time prices: {e}")
            return {}
    
    def get_spot_price(self, currency_pair: str) -> Optional[Dict]:
        """Get current spot price for a currency pair"""
        try:
            import requests
            response = requests.get(f"https://api.coinbase.com/v2/prices/{currency_pair}/spot")
            if response.status_code == 200:
                data = response.json()
                return {
                    'price': float(data['data']['amount']),
                    'timestamp': datetime.now(timezone.utc),
                    'currency': currency_pair
                }
            return None
        except Exception as e:
            logger.error(f"Error getting spot price for {currency_pair}: {e}")
            return None

def main():
    """Test WebSocket feed"""
    def price_callback(symbol, price_data):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: ${price_data['price']:.2f}")
    
    feed = WebSocketPriceFeed(['BTC-USD'], price_callback)
    
    try:
        feed.start()
        time.sleep(30)  # Run for 30 seconds
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()

if __name__ == "__main__":
    main()

