"""
Coinbase Market Adapter
=======================

Market data adapter for Coinbase (crypto).
Uses UnifiedStorage for Parquet operations.
Follows existing CryptoDataCollector patterns for gap detection and incremental fetching.
"""

import asyncio
import os
import sys
import json
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Optional, Dict, Any, List, Tuple
import pandas as pd
import logging
from pathlib import Path
try:
    import websockets
except ImportError:
    websockets = None

# Import existing Coinbase clients
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_ingestion.coinbase_advanced_client import CoinbaseAdvancedClient
from data_ingestion.websocket_price_feed import WebSocketPriceFeed
from data_ingestion.unified_data_storage import UnifiedDataStorage

# Fix relative import issue
try:
    from ..core_engine.market_adapter import (
        MarketAdapter, OHLCVBar, Order, OrderType, OrderSide, OrderStatus
    )
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core_engine.market_adapter import (
        MarketAdapter, OHLCVBar, Order, OrderType, OrderSide, OrderStatus
    )

logger = logging.getLogger(__name__)


class CoinbaseMarketAdapter(MarketAdapter):
    """
    Coinbase adapter for crypto trading.
    Wraps existing Coinbase WebSocket and REST clients for unified interface.
    
    Features:
    - Uses existing Coinbase WebSocket and REST clients
    - 24/7 continuous trading
    - No session boundaries
    
    Data streams:
    - stream_data(): Yields completed 1-minute bars (lagging by ~1 minute)
    - get_current_bar(): Returns the live in-progress bar (current minute)
    - Combined: Historical + Completed + In-Progress for real-time signals
    """
    
    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        
        # Store current in-progress bar for live streaming
        self._current_in_progress_bar = None
        self._last_completed_timestamp = None
        
        # Use existing Coinbase client
        try:
            self.client = CoinbaseAdvancedClient()
            self.product_id = f"{symbol}-USD"
        except Exception as e:
            logger.warning(f"Could not initialize Coinbase client: {e}")
            self.client = None
        
        # Unified storage for Parquet operations
        self.storage = UnifiedDataStorage(market_type='crypto')
        
        # Bar aggregation state for live streaming
        self.current_bar = None
        self.last_bar_time = None
        
        # Rate limiting (like CryptoDataCollector)
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
    async def connect(self) -> bool:
        """Connect to Coinbase"""
        try:
            if self.client is None:
                self.client = CoinbaseAdvancedClient()
            
            self.is_connected = True
            logger.info(f"Connected to Coinbase for {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Coinbase: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Coinbase"""
        self.is_connected = False
        logger.info(f"Disconnected from Coinbase for {self.symbol}")
    
    def is_market_open(self) -> bool:
        """
        Check if market is open.
        
        For crypto, this always returns True (24/7 trading).
        
        Returns:
            bool: Always True for crypto
        """
        return True  # Crypto trades 24/7
    
    async def stream_data(self, interval_seconds: int = 60) -> AsyncIterator[OHLCVBar]:
        """
        Stream live bars from Coinbase WebSocket.
        
        Uses WebSocket feed to get real-time ticks and aggregates them into 1-minute bars.
        This is true WebSocket streaming (not REST polling).
        
        Args:
            interval_seconds: Bar interval (currently supports 60s)
            
        Yields:
            OHLCVBar: Aggregated OHLCV bars built from live ticks
        """
        if interval_seconds != 60:
            raise ValueError("Coinbase adapter currently supports 60-second bars only")
        
        if not self.is_connected:
            await self.connect()
        
        # Connect to Coinbase WebSocket for real-time streaming
        uri = "wss://ws-feed.exchange.coinbase.com"
        
        try:
            websocket = await websockets.connect(uri)
            logger.info(f"Connected to Coinbase WebSocket for {self.symbol}")
            
            # Subscribe to ticker channel for real-time updates
            subscribe_message = {
                "type": "subscribe",
                "product_ids": [self.product_id],
                "channels": ["ticker"]
            }
            await websocket.send(json.dumps(subscribe_message))
            
            # Track current minute bar being built
            current_minute_data = None
            last_completed_minute = None
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'ticker':
                        price = float(data.get('price', 0))
                        current_time = datetime.now(timezone.utc)
                        
                        # Round to current minute
                        minute_time = current_time.replace(second=0, microsecond=0)
                        
                        # Initialize current minute if needed
                        if current_minute_data is None:
                            current_minute_data = {
                                'timestamp': minute_time,
                                'open': price,
                                'high': price,
                                'low': price,
                                'close': price,
                                'volume': 0.0,
                                'tick_count': 1
                            }
                            # Track in-progress bar immediately
                            self._current_in_progress_bar = current_minute_data
                        elif minute_time != current_minute_data['timestamp']:
                            # New minute started - yield completed bar
                            if last_completed_minute is None or minute_time > last_completed_minute:
                                bar = OHLCVBar(
                                    timestamp=current_minute_data['timestamp'],
                                    symbol=self.symbol,
                                    open=current_minute_data['open'],
                                    high=current_minute_data['high'],
                                    low=current_minute_data['low'],
                                    close=current_minute_data['close'],
                                    volume=current_minute_data['volume'],
                                    source='coinbase'
                                )
                                
                                if self.validate_bar(bar):
                                    # Save completed bar to storage
                                    bar_df = pd.DataFrame([bar.to_dict()], index=[bar.timestamp])
                                    self.storage.save_live_bar(self.symbol, bar_df)
                                    
                                    yield bar
                                
                                last_completed_minute = current_minute_data['timestamp']
                            
                            # Start new minute
                            current_minute_data = {
                                'timestamp': minute_time,
                                'open': price,
                                'high': price,
                                'low': price,
                                'close': price,
                                'volume': 0.0,
                                'tick_count': 1
                            }
                            
                            # Update in-progress bar NOW (the NEW minute being built)
                            self._current_in_progress_bar = current_minute_data
                        else:
                            # Update current minute (within same minute)
                            current_minute_data['high'] = max(current_minute_data['high'], price)
                            current_minute_data['low'] = min(current_minute_data['low'], price)
                            current_minute_data['close'] = price
                            current_minute_data['tick_count'] += 1
                            # Estimate volume based on tick count
                            current_minute_data['volume'] = current_minute_data['tick_count'] * 0.1
                            
                            # Update in-progress bar (current minute being built)
                            self._current_in_progress_bar = current_minute_data
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"WebSocket streaming error: {e}")
            raise
        finally:
            if websocket:
                await websocket.close()
    
    def get_current_in_progress_bar(self) -> Optional[OHLCVBar]:
        """
        Get the current in-progress bar (live data for current minute).
        
        Returns:
            OHLCVBar for the current minute being built, or None if not streaming
        """
        if self._current_in_progress_bar is None:
            return None
        
        try:
            data = self._current_in_progress_bar
            bar = OHLCVBar(
                timestamp=data['timestamp'],
                symbol=self.symbol,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                volume=data['volume'],
                source='coinbase'
            )
            return bar if self.validate_bar(bar) else None
        except Exception as e:
            logger.error(f"Error creating in-progress bar: {e}")
            return None
    
    def get_combined_data_for_signals(self, days: int = 30) -> pd.DataFrame:
        """
        Get combined data for signal generation: Historical + Completed + In-Progress.
        
        This is what should be used for generating live signals (real-time, no 1-minute lag).
        
        Args:
            days: Number of days of historical data to include
            
        Returns:
            DataFrame with combined data (historical + recent completed + current in-progress)
        """
        # Load historical + completed recent bars
        combined_df = self.storage.get_combined_data(self.symbol, days=days)
        
        if combined_df is None or combined_df.empty:
            combined_df = pd.DataFrame()
        
        # Add in-progress bar if available
        in_progress_bar = self.get_current_in_progress_bar()
        if in_progress_bar:
            in_progress_df = pd.DataFrame([in_progress_bar.to_dict()], index=[in_progress_bar.timestamp])
            
            if not combined_df.empty:
                # Append in-progress bar (remove any duplicate)
                combined_df = pd.concat([combined_df, in_progress_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.sort_index()
            else:
                combined_df = in_progress_df
        
        return combined_df
    
    async def place_order(
        self,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """Place order via Coinbase"""
        # Implementation would use existing Coinbase client
        raise NotImplementedError("Order execution not yet implemented in adapter")
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        raise NotImplementedError()
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        raise NotImplementedError()
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if not self.client:
            return {}
        
        try:
            # Use existing client methods
            return {
                'client_initialized': True,
                'symbol': self.symbol,
                'product_id': self.product_id
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    async def _rate_limit_wait(self):
        """Wait to respect rate limits (from CryptoDataCollector pattern)"""
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def find_data_gaps(self, df: pd.DataFrame, start: datetime, end: datetime) -> List[Tuple[datetime, datetime]]:
        """
        Find gaps in historical data (from CryptoDataCollector pattern).
        
        Args:
            df: Existing data
            start: Start of requested range
            end: End of requested range
            
        Returns:
            List of (gap_start, gap_end) tuples
        """
        gaps = []
        
        if df is None or df.empty:
            return [(start, end)]
        
        # Check for gap at the beginning
        if df.index.min() > start:
            gaps.append((start, df.index.min() - timedelta(minutes=1)))
        
        # Check for gaps in the middle (within requested range)
        # For backtesting: need ALL gaps filled in the requested range
        for i in range(len(df.index) - 1):
            current_end = df.index[i]
            next_start = df.index[i + 1]
            
            # Only check gaps within the requested range
            if current_end < start or next_start > end:
                continue
            
            gap_minutes = (next_start - current_end).total_seconds() / 60
            if gap_minutes > 10:  # Gap more than 10 minutes
                gap_start = current_end + timedelta(minutes=1)
                gap_end = next_start - timedelta(minutes=1)
                
                # Clamp to requested range
                gap_start = max(gap_start, start)
                gap_end = min(gap_end, end)
                
                if gap_start < gap_end:
                    gaps.append((gap_start, gap_end))
                    logger.debug(f"Found gap: {gap_start} to {gap_end} ({gap_minutes:.1f} minutes)")
        
        # Check for gap at the end
        latest_data = df.index.max()
        minutes_behind = (end - latest_data).total_seconds() / 60
        
        if minutes_behind > 5:
            gap_start = latest_data + timedelta(minutes=1)
            gaps.append((gap_start, end))
        
        return gaps
    
    async def load_historical_data(self, days: int = 30, fill_gaps: bool = True) -> pd.DataFrame:
        """
        Load historical data with gap detection and incremental fetching.
        
        Follows CryptoDataCollector pattern:
        1. Check existing Parquet files
        2. Find gaps
        3. Fetch only missing data
        4. Append-merge with existing
        
        Args:
            days: Number of days of history
            fill_gaps: If True, fetch missing data from API
            
        Returns:
            DataFrame with OHLCV data
        """
        end = datetime.now(tz=timezone.utc) - timedelta(minutes=1)
        start = end - timedelta(days=days)
        
        logger.info(f"Loading historical data for {self.symbol} (last {days} days)")
        
        # Check if we have existing data
        existing_df = self.storage.load_historical(self.symbol)
        
        if existing_df is None or existing_df.empty:
            logger.info(f"No existing data for {self.symbol}, fetching from API...")
            if fill_gaps and self.client:
                df = await self._fetch_and_save_data(start, end)
            else:
                return pd.DataFrame()
        else:
            # Check for gaps
            gaps = self.find_data_gaps(existing_df, start, end)
            
            if gaps:
                logger.info(f"Found {len(gaps)} gaps for {self.symbol}")
                
                if fill_gaps and self.client:
                    # Fetch missing data
                    for gap_start, gap_end in gaps:
                        logger.info(f"Attempting to fill gap: {gap_start} to {gap_end}")
                        df = await self._fetch_and_save_data(gap_start, gap_end)
                        if df is None or df.empty:
                            logger.warning(f"Could not fetch data for gap: {gap_start} to {gap_end}")
                        else:
                            logger.info(f"Successfully filled gap: {gap_start} to {gap_end} ({len(df)} bars)")
                else:
                    logger.warning(f"Gaps detected but fill_gaps=False")
            
            existing_df = self.storage.load_historical(self.symbol)
            return existing_df
        
        return existing_df if 'existing_df' in locals() else pd.DataFrame()
    
    async def _fetch_and_save_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch data from API and save to storage - CHUNKED to stay under 350 candles"""
        try:
            # Calculate gap size
            gap_minutes = (end - start).total_seconds() / 60
            
            # Calculate number of requests needed (350 candles max per request)
            candles_per_request = 350
            gap_requests = int((gap_minutes + candles_per_request - 1) // candles_per_request)
            
            logger.info(f"Fetching data: {start} to {end} ({gap_minutes:.1f} minutes, {gap_requests} requests)")
            
            all_candles = []
            
            # Process in chunks
            current_end = end
            for request_num in range(gap_requests):
                # Each call goes back exactly 350 minutes
                current_start = current_end - timedelta(minutes=candles_per_request)
                
                # Don't go before gap start
                if current_start < start:
                    current_start = start
                
                # Calculate duration
                duration_minutes = (current_end - current_start).total_seconds() / 60
                logger.info(f"  Request {request_num+1}/{gap_requests}: {current_start} to {current_end} ({duration_minutes:.1f} minutes)")
                
                if duration_minutes > 350:
                    logger.warning(f"  Skipping - window too large ({duration_minutes:.1f} minutes > 350)")
                    continue
                
                # Convert to Unix timestamps
                start_unix = str(int(current_start.timestamp()))
                end_unix = str(int(current_end.timestamp()))
                
                # Get candles for this specific time range - use public candles directly
                try:
                    from coinbase.rest import RESTClient
                    import os
                    
                    api_key = os.getenv('COINBASE_API_KEY')
                    api_secret = os.getenv('COINBASE_API_SECRET')
                    
                    if api_key and api_secret:
                        # Format secret if needed
                        if api_secret.startswith('-----BEGIN EC PRIVATE KEY-----'):
                            formatted_secret = api_secret.replace('\\n', '\n')
                        else:
                            formatted_secret = f"-----BEGIN EC PRIVATE KEY-----\n{api_secret}\n-----END EC PRIVATE KEY-----\n"
                        
                        client = RESTClient(api_key=api_key, api_secret=formatted_secret)
                        response = client.get_public_candles(
                            product_id=self.product_id,
                            start=start_unix,
                            end=end_unix,
                            granularity="ONE_MINUTE"
                        )
                        candles = response.candles if hasattr(response, 'candles') else []
                    else:
                        logger.error("API keys not found")
                        candles = []
                except Exception as e:
                    logger.error(f"API call failed for {start_unix} to {end_unix}: {type(e).__name__}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    candles = []
                
                if candles:
                    all_candles.extend(candles)
                    logger.info(f"  Got {len(candles)} candles, total so far: {len(all_candles)}")
                else:
                    logger.warning(f"  No candles returned for request {request_num+1}")
                
                # Move to next window
                current_end = current_start
                
                # Stop if we've reached gap start
                if current_end <= start:
                    break
                
                # Rate limit
                await self._rate_limit_wait()
            
            logger.info(f"Retrieved {len(all_candles)} candles total from API")
            
            if not all_candles:
                return pd.DataFrame()
            
            # Convert to DataFrame - handle Candle objects
            data = []
            timestamps = []
            
            for candle in all_candles:
                # Handle Candle objects (from REST client)
                if hasattr(candle, 'start'):
                    data.append({
                        'open': float(candle.open),
                        'high': float(candle.high),
                        'low': float(candle.low),
                        'close': float(candle.close),
                        'volume': float(candle.volume)
                    })
                    # Convert Unix timestamp to datetime (start is string)
                    start_str = str(candle.start) if hasattr(candle.start, '__str__') else candle.start
                    timestamps.append(pd.to_datetime(int(start_str), unit='s', utc=True))
                elif isinstance(candle, dict):
                    # Handle dict format
                    data.append({
                        'open': float(candle.get('open', 0)),
                        'high': float(candle.get('high', 0)),
                        'low': float(candle.get('low', 0)),
                        'close': float(candle.get('close', 0)),
                        'volume': float(candle.get('volume', 0))
                    })
                    # Convert Unix timestamp to datetime
                    start_time = candle.get('start', 0)
                    if isinstance(start_time, str):
                        timestamps.append(pd.to_datetime(int(start_time), unit='s', utc=True))
                    else:
                        timestamps.append(pd.to_datetime(start_time, utc=True))
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df.index = timestamps
            
            # Save to storage
            self.storage.save_historical(self.symbol, df, append=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from API: {e}")
            return pd.DataFrame()
    
    def get_market_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Get historical market data from Coinbase.
        
        Uses UnifiedStorage and gap detection pattern.
        
        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        # Calculate days range
        days = max(1, (end - start).days + 1)
        
        # Use load_historical_data which handles gaps
        df = self.load_historical_data(days=days, fill_gaps=True)
        
        # Filter to exact range
        if not df.empty:
            df = df[(df.index >= start) & (df.index <= end)]
        
        return df