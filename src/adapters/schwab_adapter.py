"""
Schwab Market Adapter
=====================

Market data adapter for Charles Schwab (equities trading).
Handles OAuth2 authentication, WebSocket streaming, bar aggregation, and session management.

Features:
- OAuth2 authentication with automatic token refresh
- Historical data fetching via marketdata/history endpoint
- Live 1-minute bar streaming via WebSocket
- Market hours awareness (9:30-16:00 ET)
- Token persistence and secure storage
"""

import asyncio
import os
import sys
import json
import logging
from datetime import datetime, timedelta, time, timezone
from typing import AsyncIterator, Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv, find_dotenv

def _reload_env_variables() -> None:
    """Reload .env variables on each adapter initialization."""
    try:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=True)
        else:
            load_dotenv(override=True)
    except Exception:
        pass

# Handle imports for both package and direct script execution
try:
    from pandas_market_calendars import get_calendar
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False
    get_calendar = None

try:
    from schwabdev.client import Client
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False
    Client = None

# Optional stream helper
try:
    from schwabdev.stream import Stream  # type: ignore
    SCHWAB_STREAM_AVAILABLE = True
except Exception:
    SCHWAB_STREAM_AVAILABLE = False
    Stream = None  # type: ignore

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None

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


class SchwabMarketAdapter(MarketAdapter):
    """
    Schwab adapter for US equities trading.
    
    Features:
    - OAuth2 authentication with automatic token refresh
    - WebSocket streaming for real-time data
    - Trade aggregation into bars
    - Session-aware (9:30-16:00 ET)
    - DST-aware market hours
    - Token persistence and secure storage
    """
    
    def __init__(
        self,
        symbol: str,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        token_path: Optional[str] = None,
        **kwargs
    ):
        if not SCHWAB_AVAILABLE:
            raise ImportError(
                "schwabdev not installed. Install with: pip install schwabdev"
            )
        
        super().__init__(symbol, **kwargs)
        
        # Store current in-progress bar for live streaming
        self._current_in_progress_bar = None
        self._last_completed_timestamp = None
        self.current_bar = None
        
        # Always reload .env on each construction
        _reload_env_variables()
        
        # API credentials (constructor overrides > env)
        client_id = client_id or os.getenv("SCHWAB_CLIENT_ID")
        client_secret = client_secret or os.getenv("SCHWAB_CLIENT_SECRET")
        redirect_uri = redirect_uri or os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1")
        
        if not client_id or not client_secret:
            logger.warning(
                "Schwab credentials not found. Set SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET"
            )
        
        # Token path for persistence
        if token_path is None:
            token_dir = Path("data") / "schwab_tokens"
            token_dir.mkdir(exist_ok=True, parents=True)
            token_path = str(token_dir / f"{symbol}_token.json")
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.token_path = token_path
        
        # Initialize client using official no-arg constructor
        self.client = None
        try:
            self.client = Client(self.client_id, self.client_secret)
            logger.info("Schwab client initialized")
        except Exception as e:
            logger.exception(f"Schwab client init failed: {e}")
        
        # Market calendar for NYSE
        self.calendar = get_calendar('NYSE') if CALENDAR_AVAILABLE else None
        
        # Unified storage for Parquet operations
        self.storage = None
        try:
            from data_ingestion.unified_data_storage import UnifiedDataStorage
            self.storage = UnifiedDataStorage(market_type='equities')
        except ImportError:
            try:
                from src.data_ingestion.unified_data_storage import UnifiedDataStorage
                self.storage = UnifiedDataStorage(market_type='equities')
            except ImportError:
                logger.warning("UnifiedStorage not available")
        
        # Bar aggregation state
        self.current_bar = None
        self.last_bar_time = None
        self.trades_buffer = []
        
        # WebSocket state
        self._ws = None
        self._bar_queue = asyncio.Queue()
        self._stream_task = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        
        # Rate limiting
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
        # Exponential backoff for retries
        self._retry_delays = [1, 2, 4, 8, 16, 32]
    
    async def connect(self) -> bool:
        """Initialize Schwab client with OAuth2 authentication."""
        try:
            # Ensure client exists
            if self.client is None:
                self.client = Client(self.client_id, self.client_secret)

            
            # Ensure tokens are valid/refresh if needed
            await self._ensure_token_valid()
            self.is_connected = True
            logger.info(f"Schwab client initialized for {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Schwab client: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Schwab"""
        try:
            if self._stream_task:
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except asyncio.CancelledError:
                    pass
                self._stream_task = None
            
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None
            
            self.is_connected = False
            logger.info(f"Disconnected from Schwab for {self.symbol}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Schwab: {e}")
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open (9:30-16:00 ET)
        
        Returns:
            bool: True if market is open
        """
        if not self.calendar:
            # Fallback: check using Eastern time explicitly
            now_et = datetime.now(ZoneInfo("America/New_York"))
            # Skip weekends
            if now_et.weekday() >= 5:
                return False
            market_open = time(9, 30)
            market_close = time(16, 0)
            return market_open <= now_et.time() <= market_close
        
        # Use market calendar for accurate check
        now = pd.Timestamp.now(tz='US/Eastern')
        schedule = self.calendar.schedule(start_date=now.date(), end_date=now.date())
        
        if schedule.empty:
            return False
        
        market_open = schedule.iloc[0]['market_open'].to_pydatetime()
        market_close = schedule.iloc[0]['market_close'].to_pydatetime()
        
        return market_open <= now.to_pydatetime() <= market_close
    
    def _should_process_trade(self, trade_time: datetime) -> bool:
        """
        Check if a trade should be processed based on market hours.
        
        Args:
            trade_time: Timestamp of the trade
            
        Returns:
            bool: True if trade is during market hours
        """
        # Convert to Eastern time
        if trade_time.tzinfo is None:
            trade_time = trade_time.replace(tzinfo=timezone.utc)
        
        et = trade_time.astimezone(ZoneInfo("America/New_York"))
        
        # Check if it's a trading day
        if self.calendar:
            if not self.calendar.valid_days(start_date=et.date(), end_date=et.date()).any():
                return False
        
        # Check time (9:30-16:00 ET)
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        return market_open <= et.time() <= market_close
    
    async def _rate_limit_wait(self):
        """Wait to respect rate limits"""
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def _ensure_token_valid(self) -> bool:
        """Ensure OAuth2 token is valid, refresh if needed via Client methods."""
        try:
            if self.client is not None:
                if hasattr(self.client, 'refresh_if_needed'):
                    await asyncio.to_thread(self.client.refresh_if_needed)
                elif hasattr(self.client, 'ensure_tokens'):
                    await asyncio.to_thread(self.client.ensure_tokens)
                elif hasattr(self.client, 'refresh_tokens'):
                    await asyncio.to_thread(self.client.refresh_tokens)
            return True
        except Exception as e:
            logger.error(f"Error ensuring tokens: {e}")
            return False
    
    async def _retry_request(self, func, *args, max_retries: int = 3, **kwargs):
        """
        Retry a request with exponential backoff for rate limits and token errors
        
        Args:
            func: Function to retry
            *args: Positional arguments
            max_retries: Maximum number of retries
            **kwargs: Keyword arguments
            
        Returns:
            Result of func
        """
        for attempt in range(max_retries):
            try:
                await self._rate_limit_wait()
                await self._ensure_token_valid()
                
                # Execute request
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(func, *args, **kwargs)
                
                self._reconnect_attempts = 0  # Reset on success
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for token expiry (401)
                if '401' in error_str or 'unauthorized' in error_str or 'token' in error_str:
                    logger.warning(f"Token expired, refreshing... (attempt {attempt + 1}/{max_retries})")
                    await self._ensure_token_valid()
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self._retry_delays[min(attempt, len(self._retry_delays) - 1)])
                        continue
                
                # Check for rate limit (429)
                elif '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
                    delay = self._retry_delays[min(attempt, len(self._retry_delays) - 1)]
                    logger.warning(f"Rate limited, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        continue
                
                # Other errors
                if attempt == max_retries - 1:
                    logger.error(f"Request failed after {max_retries} attempts: {e}")
                    raise
                else:
                    delay = self._retry_delays[min(attempt, len(self._retry_delays) - 1)]
                    logger.warning(f"Request failed, retrying in {delay}s (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(delay)
        
        raise Exception(f"Request failed after {max_retries} attempts")
    
    def find_data_gaps(self, df: pd.DataFrame, start: datetime, end: datetime) -> List[Tuple[datetime, datetime]]:
        """
        Find gaps in historical data (market-hours aware).
        
        Args:
            df: Existing data
            start: Start of requested range
            end: End of requested range
            
        Returns:
            List of (gap_start, gap_end) tuples (only for trading hours)
        """
        gaps: List[Tuple[datetime, datetime]] = []
        if df is None or df.empty:
            return [(start, end)]
        
        # Normalize/clamp helper
        def clamp_range(a: datetime, b: datetime) -> Optional[Tuple[datetime, datetime]]:
            s = max(start, a)
            e = min(end, b)
            return (s, e) if s < e else None
        
        # Beginning gap
        first_ts = df.index.min()
        begin_gap = clamp_range(start, first_ts - timedelta(minutes=1))
        if begin_gap:
            gaps.append(begin_gap)
        
        # Middle gaps
        for i in range(len(df.index) - 1):
            left = df.index[i]
            right = df.index[i + 1]
            gap_minutes = (right - left).total_seconds() / 60
            if gap_minutes > 1.5:  # missing at least one minute
                gap = clamp_range(left + timedelta(minutes=1), right - timedelta(minutes=1))
                if gap:
                    gaps.append(gap)
        
        # Ending gap
        last_ts = df.index.max()
        end_gap = clamp_range(last_ts + timedelta(minutes=1), end)
        if end_gap:
            gaps.append(end_gap)
        
        return gaps
    
    async def load_historical_data(self, days: int = 30, fill_gaps: bool = True, overwrite: bool = False) -> pd.DataFrame:
        """
        Load historical data with gap detection (market-hours aware).
        
        Args:
            days: Number of trading days of history
            fill_gaps: If True, fetch missing data from API
            
        Returns:
            DataFrame with OHLCV data (only trading hours)
        """
        if self.storage is None:
            logger.error("UnifiedStorage not initialized")
            return pd.DataFrame()
        
        # Clamp end time (no delay needed for Schwab, but respect market hours)
        now_utc = datetime.now(tz=timezone.utc)
        end = now_utc
        start = end - timedelta(days=days)
        
        logger.info(f"Loading historical data for {self.symbol} (last {days} days)")
        
        # Check if we have existing data
        existing_df = self.storage.load_historical(self.symbol)
        
        if overwrite:
            logger.info(f"Overwrite enabled; fetching full window {start} -> {end}")
            await self._fetch_and_save_data(start, end, append=False)
            existing_df = self.storage.load_historical(self.symbol)
            return existing_df if existing_df is not None else pd.DataFrame()
        
        if existing_df is None or existing_df.empty:
            logger.info(f"No existing data for {self.symbol}, fetching from API...")
            if fill_gaps and self.client:
                await self._fetch_and_save_data(start, end, append=False)
            else:
                return pd.DataFrame()
        else:
            # Fetch only missing ranges
            existing_min = existing_df.index.min()
            existing_max = existing_df.index.max()
            
            # Earlier missing range: [start, existing_min)
            early_start = start
            early_end = min(existing_min - timedelta(minutes=1), end)
            if fill_gaps and self.client and early_start < early_end:
                logger.info(f"Filling earlier missing range: {early_start} to {early_end}")
                await self._fetch_and_save_data(early_start, early_end, append=True)
            
            # Later missing range: (existing_max, end]
            late_start = max(existing_max + timedelta(minutes=1), start)
            late_end = end
            if fill_gaps and self.client and late_start < late_end:
                logger.info(f"Filling later missing range: {late_start} to {late_end}")
                await self._fetch_and_save_data(late_start, late_end, append=True)
            
            # Reload after fills
            existing_df = self.storage.load_historical(self.symbol)
            return existing_df
        
        return existing_df if 'existing_df' in locals() else pd.DataFrame()
    
    async def _fetch_and_save_data(self, start: datetime, end: datetime, append: bool = True) -> pd.DataFrame:
        """Fetch data from Schwab API and save to storage."""
        try:
            if self.client is None:
                logger.error("Schwab client not initialized")
                return pd.DataFrame()
            
            logger.info(f"Fetching {self.symbol} from {start} to {end}")
            
            def fetch_data():
                """Synchronous fetch function using Schwabdev's price_history endpoint (final working version)."""
                try:
                    if hasattr(self.client, "price_history"):
                        # Schwab requires ms timestamps and camelCase param names
                        start_ms = int(start.timestamp() * 1000)
                        end_ms = int(end.timestamp() * 1000)

                        response = self.client.price_history(
                            symbol=self.symbol,
                            startDate=start_ms,
                            endDate=end_ms,
                            frequencyType="minute",
                            frequency=1,
                            needExtendedHoursData=False,
                        )

                        # --- Decode JSON safely ---
                        if hasattr(response, "json"):
                            try:
                                data = response.json()
                            except Exception:
                                logger.warning("price_history(): response not JSON decodable; using raw text")
                                data = json.loads(response.text)
                        else:
                            data = response

                        return data

                    raise AttributeError("No supported price history method found on this client.")
                except Exception as e:
                    logger.error(f"Error in fetch_data: {e}")
                    raise

            # Fetch with retry
            response = await self._retry_request(fetch_data)
            
            # Parse response to DataFrame
            if response:
                # Extract candles from response
                candles = []
                if isinstance(response, dict):
                    # Common keys: 'candles', 'data', or nested under 'result'
                    if 'candles' in response:
                        candles = response['candles']
                    elif 'data' in response and isinstance(response['data'], dict) and 'candles' in response['data']:
                        candles = response['data']['candles']
                    else:
                        candles = response.get('data', [])
                elif hasattr(response, 'candles'):
                    candles = getattr(response, 'candles')
                elif hasattr(response, 'data'):
                    candles = getattr(response, 'data')
                elif isinstance(response, list):
                    candles = response

                rows: List[Dict[str, Any]] = []
                for candle in candles or []:
                    if isinstance(candle, dict):
                        ts = candle.get('datetime', candle.get('time', candle.get('timestamp')))
                        if isinstance(ts, (int, float)):
                            # Convert milliseconds/seconds to datetime (assume ms if large)
                            ts = datetime.fromtimestamp((ts / 1000) if ts > 1e12 else ts, tz=timezone.utc)
                        else:
                            ts = pd.to_datetime(ts)
                            if ts.tzinfo is None:
                                ts = ts.tz_localize('UTC')
                            else:
                                ts = ts.tz_convert('UTC')
                        rows.append({
                            'timestamp': ts,
                            'open': float(candle.get('open', candle.get('o', 0) or 0)),
                            'high': float(candle.get('high', candle.get('h', 0) or 0)),
                            'low': float(candle.get('low', candle.get('l', 0) or 0)),
                            'close': float(candle.get('close', candle.get('c', 0) or 0)),
                            'volume': float(candle.get('volume', candle.get('v', 0) or 0)),
                        })
                    else:
                        # Object-like candle
                        ts = getattr(candle, 'datetime', getattr(candle, 'time', getattr(candle, 'timestamp', None)))
                        if ts is not None:
                            if isinstance(ts, (int, float)):
                                ts = datetime.fromtimestamp((ts / 1000) if ts > 1e12 else ts, tz=timezone.utc)
                            else:
                                ts = pd.to_datetime(ts)
                                if ts.tzinfo is None:
                                    ts = ts.tz_localize('UTC')
                                else:
                                    ts = ts.tz_convert('UTC')
                            rows.append({
                                'timestamp': ts,
                                'open': float(getattr(candle, 'open', getattr(candle, 'o', 0) or 0)),
                                'high': float(getattr(candle, 'high', getattr(candle, 'h', 0) or 0)),
                                'low': float(getattr(candle, 'low', getattr(candle, 'l', 0) or 0)),
                                'close': float(getattr(candle, 'close', getattr(candle, 'c', 0) or 0)),
                                'volume': float(getattr(candle, 'volume', getattr(candle, 'v', 0) or 0)),
                            })

                if rows:
                    df = pd.DataFrame(rows)
                    df.set_index('timestamp', inplace=True)
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                    else:
                        df.index = df.index.tz_convert('UTC')
                    df = df.sort_index()
                    df = df[~df.index.duplicated(keep='last')][['open', 'high', 'low', 'close', 'volume']]
                    logger.info(f"Fetched {len(df)} bars for {self.symbol}")
                    if self.storage:
                        self.storage.save_historical(self.symbol, df, append=append)
                    return df

            logger.warning(f"No data fetched for {self.symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching data from Schwab API: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
    
    async def stream_data(self, interval_seconds: int = 60) -> AsyncIterator[OHLCVBar]:
        """
        Stream 1-minute bars from Schwab.
        
        Trades are aggregated into bars during market hours only.
        
        Args:
            interval_seconds: Bar interval (currently supports 60s = 1-minute)
            
        Yields:
            OHLCVBar: Aggregated OHLCV bars
        """
        if interval_seconds != 60:
            raise ValueError("Schwab adapter currently supports 60-second bars only")
        
        if not self.is_connected:
            await self.connect()
        
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library not installed. Install with: pip install websockets")
        
        # Start streaming task
        if self._stream_task is None or self._stream_task.done():
            logger.info("Starting Schwab WebSocket stream...")
            self._stream_task = asyncio.create_task(self._stream_worker())
            await asyncio.sleep(1)  # Allow connection to establish
        
        try:
            while self.is_connected:
                # Yield bars from queue
                try:
                    bar = await asyncio.wait_for(
                        self._bar_queue.get(),
                        timeout=0.1
                    )
                    yield bar
                    
                except asyncio.TimeoutError:
                    # No bar available yet, continue
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"Error streaming data: {e}")
            raise
    
    async def _stream_worker(self):
        """Prefer library stream; fallback to quote polling."""
        # Library stream path
        if SCHWAB_STREAM_AVAILABLE:
            try:
                await self._ensure_token_valid()
                stream = Stream(self.client)  # type: ignore

                async def on_quote(msg: dict):
                    try:
                        price = float(msg.get("last", msg.get("price", 0.0)))
                        size = float(msg.get("size", msg.get("volume", 0.0)))
                        ts = msg.get("timestamp") or msg.get("time") or msg.get("datetime")
                        trade_time = pd.to_datetime(ts, utc=True).to_pydatetime() if ts else datetime.now(tz=timezone.utc)
                        await self._process_trade(trade_time, price, size)
                    except Exception as e:
                        logger.error(f"quote handler error: {e}")

                await stream.subscribe_quotes([self.symbol], on_quote)  # type: ignore
                await stream.run_forever()  # type: ignore
                return
            except Exception as e:
                logger.warning(f"Stream module failed, falling back to polling: {e}")

        # Polling fallback at ~1Hz
        logger.info("Starting polling fallback for live bars (1Hz).")
        try:
            while self.is_connected:
                try:
                    await self._ensure_token_valid()

                    def _get_quote():
                        if hasattr(self.client, "marketdata") and hasattr(self.client.marketdata, "quote"):
                            return self.client.marketdata.quote(symbol=self.symbol)
                        return self.client.get(f"/marketdata/v1/quotes/{self.symbol}")

                    quote = await self._retry_request(_get_quote)

                    if isinstance(quote, dict):
                        px = quote.get("last") or quote.get("price") or quote.get("close") or quote.get("mark") or 0.0
                        sz = quote.get("size") or quote.get("lastSize") or quote.get("volume") or 0.0
                        ts = quote.get("timestamp") or quote.get("time") or quote.get("datetime")
                    else:
                        px = getattr(quote, "last", None) or getattr(quote, "price", 0.0)
                        sz = getattr(quote, "size", None) or getattr(quote, "volume", 0.0)
                        ts = getattr(quote, "timestamp", None) or getattr(quote, "time", None) or getattr(quote, "datetime", None)

                    price = float(px or 0.0)
                    size = float(sz or 0.0)
                    trade_time = pd.to_datetime(ts, utc=True).to_pydatetime() if ts else datetime.now(tz=timezone.utc)

                    await self._process_trade(trade_time, price, size)
                except Exception as e:
                    logger.warning(f"Polling error: {e}")
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Polling fatal: {e}")
    
    async def _process_trade(self, trade_time: datetime, price: float, volume: float):
        """Process incoming trade and update current bar"""
        try:
            # Check if trade is during market hours
            if not self._should_process_trade(trade_time):
                return
            
            # Align trade to minute boundary
            aligned_time = trade_time.replace(second=0, microsecond=0)
            
            # Initialize bar if needed
            if self.current_bar is None or aligned_time != self.current_bar['timestamp']:
                # Save previous bar if exists
                if self.current_bar is not None:
                    bar = OHLCVBar(
                        timestamp=self.current_bar['timestamp'],
                        symbol=self.symbol,
                        open=self.current_bar['open'],
                        high=self.current_bar['high'],
                        low=self.current_bar['low'],
                        close=self.current_bar['close'],
                        volume=self.current_bar['volume'],
                        source='schwab'
                    )
                    
                    if self.validate_bar(bar):
                        # Save live bar to storage
                        bar_df = pd.DataFrame([bar.to_dict()], index=[bar.timestamp])
                        if self.storage:
                            self.storage.save_live_bar(self.symbol, bar_df)
                        await self._bar_queue.put(bar)
                
                # Start new bar
                self.current_bar = {
                    'timestamp': aligned_time,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume
                }
            else:
                # Update current bar
                self.current_bar['high'] = max(self.current_bar['high'], price)
                self.current_bar['low'] = min(self.current_bar['low'], price)
                self.current_bar['close'] = price
                self.current_bar['volume'] += volume
            
            # Update in-progress bar (for real-time signal generation)
            if self.current_bar:
                self._current_in_progress_bar = self.current_bar.copy()
        
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
    
    async def _process_completed_bar(self, data: dict):
        """Process a completed bar from WebSocket"""
        try:
            symbol = data.get('symbol', self.symbol)
            timestamp_str = data.get('timestamp', data.get('datetime', data.get('time', '')))
            
            if timestamp_str:
                if isinstance(timestamp_str, (int, float)):
                    ts = datetime.fromtimestamp(timestamp_str / 1000, tz=timezone.utc)
                else:
                    ts = pd.to_datetime(timestamp_str)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize('UTC')
                    else:
                        ts = ts.tz_convert('UTC')
            else:
                ts = datetime.now(tz=timezone.utc)
            
            ts = ts.replace(second=0, microsecond=0)
            
            bar = OHLCVBar(
                timestamp=ts,
                symbol=symbol,
                open=float(data.get('open', 0)),
                high=float(data.get('high', 0)),
                low=float(data.get('low', 0)),
                close=float(data.get('close', 0)),
                volume=float(data.get('volume', 0)),
                source='schwab'
            )
            
            if self.validate_bar(bar):
                bar_df = pd.DataFrame([bar.to_dict()], index=[bar.timestamp])
                if self.storage:
                    self.storage.save_live_bar(self.symbol, bar_df)
                await self._bar_queue.put(bar)
                
                # Update in-progress bar baseline
                self._current_in_progress_bar = {
                    'timestamp': ts,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }
        
        except Exception as e:
            logger.error(f"Error processing completed bar: {e}")
    
    def get_current_in_progress_bar(self):
        """Get the current in-progress bar (live data for current minute)"""
        if self._current_in_progress_bar is None:
            return None
        
        try:
            data = self._current_in_progress_bar
            
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in data:
                timestamp = data['timestamp']
                if not isinstance(timestamp, datetime):
                    timestamp = pd.to_datetime(timestamp, utc=True)
                
                bar = OHLCVBar(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    volume=data['volume'],
                    source='schwab'
                )
                return bar if self.validate_bar(bar) else None
        except Exception as e:
            logger.error(f"Error creating in-progress bar: {e}")
            return None
    
    def get_combined_data_for_signals(self, days: int = 30):
        """
        Get combined data for signal generation: Historical + Completed + In-Progress.
        
        For real-time signals with zero lag.
        """
        if self.storage is None:
            return pd.DataFrame()
        
        # Load historical + completed recent bars
        combined_df = self.storage.get_combined_data(self.symbol, days=days)
        
        if combined_df is None or combined_df.empty:
            combined_df = pd.DataFrame()
        
        # Add in-progress bar if available
        in_progress_bar = self.get_current_in_progress_bar()
        if in_progress_bar:
            in_progress_df = pd.DataFrame([in_progress_bar.to_dict()], index=[in_progress_bar.timestamp])
            
            if not combined_df.empty:
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
        """
        Place an order via Schwab.
        
        Note: This adapter focuses on market data. For order execution,
        use SchwabBroker (if implemented).
        """
        raise NotImplementedError(
            "Use SchwabBroker for order execution. "
            "This adapter is for market data only."
        )
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        raise NotImplementedError("Use SchwabBroker for order management")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        raise NotImplementedError("Use SchwabBroker for order management")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if not self.client:
            return {}
        
        try:
            await self._ensure_token_valid()
            
            def fetch_account():
                if hasattr(self.client, 'get_account'):
                    return self.client.get_account()
                elif hasattr(self.client, 'accounts'):
                    accounts = self.client.accounts.get_all_accounts()
                    return accounts[0] if accounts else None
                else:
                    # Fallback API call
                    response = self.client.get("/trader/v1/accounts")
                    return response.json() if hasattr(response, 'json') else response
            
            account = await self._retry_request(fetch_account)
            
            if account:
                if isinstance(account, dict):
                    return {
                        'buying_power': float(account.get('buyingPower', account.get('buying_power', 0))),
                        'equity': float(account.get('equity', 0)),
                        'cash': float(account.get('cash', 0)),
                        'day_trading_buying_power': float(account.get('dayTradingBuyingPower', account.get('day_trading_buying_power', 0)))
                    }
                else:
                    return {
                        'buying_power': float(getattr(account, 'buying_power', getattr(account, 'buyingPower', 0))),
                        'equity': float(getattr(account, 'equity', 0)),
                        'cash': float(getattr(account, 'cash', 0)),
                        'day_trading_buying_power': float(getattr(account, 'day_trading_buying_power', getattr(account, 'dayTradingBuyingPower', 0)))
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_market_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Get historical market data from Schwab.
        
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
        df = asyncio.run(self.load_historical_data(days=days, fill_gaps=True))
        
        # Filter to exact range
        if not df.empty:
            df = df[(df.index >= start) & (df.index <= end)]
        
        return df

