"""
Alpaca Market Adapter
=====================

Market data adapter for Alpaca (equities trading).
Handles WebSocket streaming, bar aggregation, and session management.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta, time, timezone
from typing import AsyncIterator, Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv, find_dotenv

def _reload_env_variables() -> None:
    """Reload .env variables on each adapter initialization (override in-memory)."""
    try:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=True)
        else:
            load_dotenv(override=True)
    except Exception:
        # Non-fatal if dotenv is unavailable or file not found
        pass

# Handle imports for both package and direct script execution
try:
    from pandas_market_calendars import get_calendar
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False
    get_calendar = None

try:
    from alpaca.data.live import StockDataStream
    from alpaca.data import StockHistoricalDataClient, TimeFrame
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.enums import DataFeed
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide, OrderType as AlpacaOrderType, TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

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


class AlpacaMarketAdapter(MarketAdapter):
    """
    Alpaca adapter for US equities trading.
    
    Features:
    - WebSocket streaming (v2/sip or v2/iex)
    - Trade aggregation into bars
    - Session-aware (9:30-16:00 ET)
    - DST-aware market hours
    """
    
    def __init__(self, symbol: str, paper: bool = True, api_key: Optional[str] = None, secret_key: Optional[str] = None, **kwargs):
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py not installed. Install with: pip install alpaca-py"
            )
        
        super().__init__(symbol, **kwargs)
        self.paper = paper
        
        # Store current in-progress bar for live streaming
        self._current_in_progress_bar = None
        self._last_completed_timestamp = None
        self.current_bar = None  # For bar aggregation
        
        # Always reload .env on each construction to pick up latest values
        _reload_env_variables()

        # API credentials (constructor overrides > env)
        api_key = api_key or os.getenv("ALPACA_API_KEY")
        secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            logger.warning(
                "Alpaca credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY"
            )
        else:
            # Masked logging for verification
            try:
                masked_api = f"{api_key[:4]}...{api_key[-3:]}"
                masked_secret = f"{secret_key[:4]}...{secret_key[-3:]}"
                logger.info(f"Alpaca credentials loaded: key={masked_api} secret={masked_secret}")
            except Exception:
                pass
        
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Initialize clients
        self.data_stream = None
        self.historical_client = None
        self.trading_client = None
        
        # Market calendar for NYSE
        self.calendar = get_calendar('NYSE') if CALENDAR_AVAILABLE else None
        
        # Unified storage for Parquet operations
        self.storage = None  # Will initialize with import workaround
        try:
            from data_ingestion.unified_data_storage import UnifiedDataStorage
            self.storage = UnifiedDataStorage(market_type='equities')
        except ImportError:
            logger.warning("UnifiedStorage not available")
        
        # Bar aggregation state
        self.current_bar = None
        self.last_bar_time = None
        self.trades_buffer = []
        
        # Connection state
        self._ws = None
        self._bar_queue = asyncio.Queue()
        
        # Rate limiting (like CryptoDataCollector)
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
        # Historical feed selection (affects time clamping)
        self.historical_feed: str = 'sip'  # 'sip' (15-min delay on Basic) or 'iex'
        
    async def connect(self) -> bool:
        """Initialize Alpaca clients. Streaming started in stream_data."""
        try:
            # Historical client (auth handled by SDK)
            if not self.historical_client:
                self.historical_client = StockHistoricalDataClient(self.api_key, self.secret_key)
                logger.info("Historical client initialized")

            # Data stream client (do not run here)
            if self.data_stream is None:
                # Use IEX feed for free-plan real-time eligibility
                self.data_stream = StockDataStream(self.api_key, self.secret_key, feed=DataFeed.IEX)
                # Diagnostics: errors/disconnects
                try:
                    self.data_stream.on_error(lambda e: logger.error(f"Stream error: {e}"))
                    self.data_stream.on_disconnect(lambda: logger.warning("Stream disconnected"))
                except Exception as _e:
                    logger.debug(f"Could not attach stream diagnostics: {_e}")

            # Trading client
            if not self.trading_client:
                self.trading_client = TradingClient(api_key=self.api_key, secret_key=self.secret_key, paper=self.paper)

            self.is_connected = True
            logger.info(f"Clients initialized for {self.symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Alpaca clients: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Alpaca"""
        try:
            if getattr(self, "_stream_task", None):
                try:
                    # Stop stream if running
                    await self.data_stream.stop()
                except Exception:
                    pass
                self._stream_task.cancel()
                self._stream_task = None

            self.is_connected = False
            logger.info(f"Disconnected from Alpaca for {self.symbol}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Alpaca: {e}")
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open (9:30-16:00 ET)
        
        Returns:
            bool: True if market is open
        """
        if not self.calendar:
            # Fallback: check using Eastern time explicitly
            now_et = datetime.now(ZoneInfo("America/New_York"))
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
        if not self.calendar.valid_days(start_date=et.date(), end_date=et.date()).any():
            return False
        
        # Check time (9:30-16:00 ET)
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        return market_open <= et.time() <= market_close
    
    async def _rate_limit_wait(self):
        """Wait to respect rate limits (from CryptoDataCollector pattern)"""
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
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
        
        # Clamp end for SIP (Basic plan enforces 15-min delay)
        now_utc = datetime.now(tz=timezone.utc)
        end = now_utc - timedelta(minutes=15) if getattr(self, 'historical_feed', 'sip') == 'sip' else now_utc
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
            if fill_gaps and self.historical_client:
                await self._fetch_and_save_data(start, end, append=False)
            else:
                return pd.DataFrame()
        else:
            # Fetch only truly missing disjoint ranges against requested window
            existing_min = existing_df.index.min()
            existing_max = existing_df.index.max()

            # Earlier missing range: [start, existing_min)
            early_start = start
            early_end = min(existing_min - timedelta(minutes=1), end)
            if fill_gaps and self.historical_client and early_start < early_end:
                logger.info(f"Filling earlier missing range: {early_start} to {early_end}")
                await self._fetch_and_save_data(early_start, early_end, append=True)

            # Later missing range: (existing_max, end]
            late_start = max(existing_max + timedelta(minutes=1), start)
            late_end = end
            if fill_gaps and self.historical_client and late_start < late_end:
                logger.info(f"Filling later missing range: {late_start} to {late_end}")
                await self._fetch_and_save_data(late_start, late_end, append=True)

            # Reload after fills
            existing_df = self.storage.load_historical(self.symbol)
            return existing_df
        
        return existing_df if 'existing_df' in locals() else pd.DataFrame()
    
    async def _fetch_and_save_data(self, start: datetime, end: datetime, append: bool = True) -> pd.DataFrame:
        """Fetch data from Alpaca API and save to storage (paged via next_page_token)."""
        try:
            if self.historical_client is None:
                logger.error("Alpaca historical client not initialized")
                return pd.DataFrame()
            
            # Alpaca's get_stock_bars is synchronous, so we run it in executor
            import asyncio
            loop = asyncio.get_event_loop()
            logger.info(f"Fetching {self.symbol} from {start} to {end}")

            # Chunked fetching independent of next_page_token (handles any timeframe)
            frames_chunked: List[pd.DataFrame] = []
            minutes_per_bar = 1  # TimeFrame.Minute; extend mapping if other timeframes are added
            max_minutes = 10000 * minutes_per_bar
            seg_start = start

            def fetch_segment(seg_s: datetime, seg_e: datetime):
                req = StockBarsRequest(
                    symbol_or_symbols=[self.symbol],
                    timeframe=TimeFrame.Minute,
                    start=seg_s,
                    end=seg_e,
                    limit=10000,
                    feed=DataFeed.SIP if getattr(self, 'historical_feed', 'sip') == 'sip' else DataFeed.IEX,
                )
                return self.historical_client.get_stock_bars(req)

            while seg_start < end:
                seg_end = min(seg_start + timedelta(minutes=max_minutes - 1), end)
                resp = await loop.run_in_executor(None, fetch_segment, seg_start, seg_end)
                df_page = getattr(resp, 'df', None)
                frame: Optional[pd.DataFrame] = None
                if isinstance(df_page, pd.DataFrame) and not df_page.empty:
                    try:
                        if 'symbol' in df_page.index.names:
                            frame = df_page.xs(self.symbol, level='symbol')
                        else:
                            frame = df_page
                    except Exception:
                        frame = df_page
                else:
                    rows = []
                    data = getattr(resp, 'data', {})
                    bars = data.get(self.symbol, []) if isinstance(data, dict) else []
                    for bar in bars:
                        rows.append({
                            'timestamp': bar.timestamp,
                            'open': float(bar.open),
                            'high': float(bar.high),
                            'low': float(bar.low),
                            'close': float(bar.close),
                            'volume': float(bar.volume)
                        })
                    frame = pd.DataFrame(rows)
                    if not frame.empty:
                        frame.index = pd.to_datetime(frame['timestamp'])
                        frame.drop(columns=['timestamp'], inplace=True, errors='ignore')

                if frame is not None and not frame.empty:
                    if frame.index.tz is None:
                        frame.index = frame.index.tz_localize('UTC')
                    else:
                        frame.index = frame.index.tz_convert('UTC')
                    frames_chunked.append(frame[['open','high','low','close','volume']])
                    logger.info(f"Segment {seg_start} -> {seg_end}: {len(frame)} bars, total so far: {sum(len(f) for f in frames_chunked)}")
                else:
                    logger.info(f"Segment {seg_start} -> {seg_end}: 0 bars")

                seg_start = seg_end + timedelta(minutes=1)

            if frames_chunked:
                df = pd.concat(frames_chunked).sort_index()
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                logger.info(f"Fetched {len(df)} bars for {self.symbol} via chunked fetching")
                if self.storage:
                    self.storage.save_historical(self.symbol, df, append=append)
                return df

            def fetch_page(token: Optional[str]):
                req = StockBarsRequest(
                    symbol_or_symbols=[self.symbol],
                    timeframe=TimeFrame.Minute,
                    start=start,
                    end=end,
                    limit=10000,
                    feed=DataFeed.SIP if getattr(self, 'historical_feed', 'sip') == 'sip' else DataFeed.IEX,
                    page_token=token
                )
                resp = self.historical_client.get_stock_bars(req)
                logger.debug(f"Response type: {type(resp)}, dir: {[x for x in dir(resp) if not x.startswith('_')][:20]}")
                if hasattr(resp, 'data'):
                    logger.debug(f"Response.data keys: {list(resp.data.keys()) if isinstance(resp.data, dict) else type(resp.data)}")
                return resp

            frames: List[pd.DataFrame] = []
            page_token: Optional[str] = None
            page_num = 0
            while True:
                page_num += 1
                resp = await loop.run_in_executor(None, fetch_page, page_token)
                if resp is None:
                    break

                # Get bars from response for this symbol
                bars = resp[self.symbol] if self.symbol in resp else []
                
                df_page = getattr(resp, 'df', None)
                frame: Optional[pd.DataFrame] = None
                if isinstance(df_page, pd.DataFrame) and not df_page.empty:
                    try:
                        if 'symbol' in df_page.index.names:
                            frame = df_page.xs(self.symbol, level='symbol')
                        else:
                            frame = df_page
                    except Exception:
                        frame = df_page
                else:
                    rows = []
                    for bar in bars:
                        rows.append({
                            'timestamp': bar.timestamp,
                            'open': float(bar.open),
                            'high': float(bar.high),
                            'low': float(bar.low),
                            'close': float(bar.close),
                            'volume': float(bar.volume)
                        })
                    frame = pd.DataFrame(rows)
                    if not frame.empty:
                        frame.index = pd.to_datetime(frame['timestamp'])
                        frame.drop(columns=['timestamp'], inplace=True, errors='ignore')

                if frame is not None and not frame.empty:
                    if frame.index.tz is None:
                        frame.index = frame.index.tz_localize('UTC')
                    else:
                        frame.index = frame.index.tz_convert('UTC')
                    frames.append(frame[['open','high','low','close','volume']])
                    logger.info(f"Page {page_num}: {len(frame)} bars, total so far: {sum(len(f) for f in frames)}")

                page_token = getattr(resp, 'next_page_token', None)
                logger.info(f"Page {page_num} complete: bars={len(frame) if frame is not None and not frame.empty else 0}, token={page_token[:20] if page_token else None}, has_token={bool(page_token)}")
                if not page_token:
                    logger.info(f"Pagination complete: {page_num} pages, {sum(len(f) for f in frames)} total bars")
                    break

            if not frames:
                logger.warning(f"No data fetched for {self.symbol}")
                return pd.DataFrame()

            df = pd.concat(frames).sort_index()
            
            # Ensure timezone-aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            logger.info(f"Fetched {len(df)} bars for {self.symbol}")
            
            # Save to storage
            if self.storage:
                self.storage.save_historical(self.symbol, df, append=append)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Alpaca API: {e}")
            return pd.DataFrame()
    
    async def stream_data(self, interval_seconds: int = 60) -> AsyncIterator[OHLCVBar]:
        """
        Stream 1-minute bars from Alpaca.
        
        Trades are aggregated into bars during market hours only.
        
        Args:
            interval_seconds: Bar interval (currently supports 60s = 1-minute)
            
        Yields:
            OHLCVBar: Aggregated OHLCV bars
        """
        if interval_seconds != 60:
            raise ValueError("Alpaca adapter currently supports 60-second bars only")
        
        if not self.is_connected:
            await self.connect()

        # Register handlers via explicit subscriptions (SDK version without decorators)
        async def _on_trade(trade):
            """Handle incoming trade data"""
            try:
                logger.info(f"on_trade trade received at {trade.timestamp}")
                # Check if trade is during market hours
                trade_time = trade.timestamp
                
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
                            source='alpaca'
                        )
                        
                        if self.validate_bar(bar):
                            # Save live bar to storage (accumulates)
                            bar_df = pd.DataFrame([bar.to_dict()], index=[bar.timestamp])
                            if self.storage:
                                self.storage.save_live_bar(self.symbol, bar_df)
                            await self._bar_queue.put(bar)
                    
                    # Start new bar
                    self.current_bar = {
                        'timestamp': aligned_time,
                        'open': trade.price,
                        'high': trade.price,
                        'low': trade.price,
                        'close': trade.price,
                        'volume': trade.size
                    }
                else:
                    # Update current bar
                    self.current_bar['high'] = max(self.current_bar['high'], trade.price)
                    self.current_bar['low'] = min(self.current_bar['low'], trade.price)
                    self.current_bar['close'] = trade.price
                    self.current_bar['volume'] += trade.size
                
                # Update in-progress bar (for real-time signal generation)
                if self.current_bar:
                    self._current_in_progress_bar = self.current_bar
                    
            except Exception as e:
                logger.error(f"Error handling trade: {e}")

        async def _on_bar(bar):
            """Handle completed 1-minute bar events (IEX bars)."""
            try:
                ts = bar.timestamp.replace(second=0, microsecond=0)
                ohlcv = {
                    'timestamp': ts,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': float(bar.volume),
                }
                bar_obj = OHLCVBar(
                    timestamp=ts,
                    symbol=self.symbol,
                    open=ohlcv['open'],
                    high=ohlcv['high'],
                    low=ohlcv['low'],
                    close=ohlcv['close'],
                    volume=ohlcv['volume'],
                    source='alpaca'
                )
                if self.validate_bar(bar_obj):
                    bar_df = pd.DataFrame([bar_obj.to_dict()], index=[bar_obj.timestamp])
                    if self.storage:
                        self.storage.save_live_bar(self.symbol, bar_df)
                    await self._bar_queue.put(bar_obj)
                # Baseline for in-progress if trades sparse
                self._current_in_progress_bar = ohlcv
            except Exception as e:
                logger.error(f"Error handling bar: {e}")

        # Subscribe before starting the stream (SDK variants)
        try:
            # Newer SDK: set_handlers + subscribe_* with symbol lists
            if hasattr(self.data_stream, 'set_handlers'):
                try:
                    self.data_stream.set_handlers(bars=_on_bar, trades=_on_trade)
                    logger.info("Handlers set via set_handlers")
                except Exception as e:
                    logger.debug(f"set_handlers failed: {e}")
            # Subscribe symbols (list-based)
            if hasattr(self.data_stream, 'subscribe_bars'):
                try:
                    self.data_stream.subscribe_bars([self.symbol])
                    logger.info(f"Subscribed bars for {self.symbol}")
                except Exception as e:
                    logger.debug(f"subscribe_bars failed: {e}")
            if hasattr(self.data_stream, 'subscribe_trades'):
                try:
                    self.data_stream.subscribe_trades([self.symbol])
                    logger.info(f"Subscribed trades for {self.symbol}")
                except Exception as e:
                    logger.debug(f"subscribe_trades failed: {e}")
            # Older SDK variant: function-based subscribe
            if hasattr(self.data_stream, 'subscribe'):
                try:
                    self.data_stream.subscribe(trades=[self.symbol], bars=[self.symbol])
                    logger.info(f"Subscribed via generic subscribe() for {self.symbol}")
                except Exception as e:
                    logger.debug(f"generic subscribe() failed: {e}")
        except Exception as e:
            logger.error(f"Error subscribing to IEX stream: {e}")

        # Start the stream in background (non-blocking)
        if getattr(self, "_stream_task", None) is None or self._stream_task.done():
            logger.info("Starting Alpaca WebSocket stream (IEX)...")
            self._stream_task = asyncio.create_task(self.data_stream.run())
            await asyncio.sleep(1)
            logger.info(f"Stream task started? {not self._stream_task.done()}")
        
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
    
    def get_current_in_progress_bar(self):
        """Get the current in-progress bar (live data for current minute)"""
        if self._current_in_progress_bar is None:
            return None
        
        try:
            data = self._current_in_progress_bar
            
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in data:
                timestamp = data['timestamp']
                if not hasattr(timestamp, 'timestamp'):
                    timestamp = pd.to_datetime(timestamp, utc=True)
                
                bar = OHLCVBar(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    volume=data['volume'],
                    source='alpaca'
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
        Place an order via Alpaca.
        
        Note: This adapter focuses on market data. For order execution,
        use AlpacaBroker.
        """
        raise NotImplementedError(
            "Use AlpacaBroker for order execution. "
            "This adapter is for market data only."
        )
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        raise NotImplementedError("Use AlpacaBroker for order management")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        raise NotImplementedError("Use AlpacaBroker for order management")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if not self.trading_client:
            return {}
        
        try:
            account = self.trading_client.get_account()
            return {
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'cash': float(account.cash),
                'day_trading_buying_power': float(account.daytrading_buying_power)
            }
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
        Get historical market data from Alpaca.
        
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
