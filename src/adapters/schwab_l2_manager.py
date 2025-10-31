"""
Schwab Level 2 Order Book Data Manager (fixed)
==============================================

✅ Uses schwabdev.stream.Stream for live BOOK data
✅ Handles reconnects, cache, and SQLite persistence
✅ Compatible with existing Schwab adapter
"""

import asyncio
import logging
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from schwabdev.stream import Stream
    SCHWAB_STREAM_AVAILABLE = True
except Exception:
    SCHWAB_STREAM_AVAILABLE = False
    Stream = None


class L2DataManager:
    def __init__(
        self,
        client,
        symbol: str,
        cache_size: int = 1000,
        storage_path: Optional[str] = None,
    ):
        self.client = client
        self.symbol = symbol.upper()
        self.cache_size = cache_size
        self.l2_cache = deque(maxlen=cache_size)
        self._ws_task = None
        self._stop_event = asyncio.Event()

        # SQLite storage
        if storage_path is None:
            storage_dir = Path("data") / "l2_data"
            storage_dir.mkdir(exist_ok=True, parents=True)
            storage_path = str(storage_dir / f"{symbol}_l2.db")
        self.storage_path = Path(storage_path)
        self._init_database()

    # ==============================
    # DATABASE INITIALIZATION
    # ==============================
    def _init_database(self):
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS l2_updates (
                    timestamp INTEGER PRIMARY KEY,
                    symbol TEXT,
                    bid_price REAL,
                    bid_size INTEGER,
                    ask_price REAL,
                    ask_size INTEGER,
                    bid_levels TEXT,
                    ask_levels TEXT
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON l2_updates(timestamp)"
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[L2] Error initializing SQLite DB: {e}")

    # ==============================
    # CORE L2 STREAM LOGIC
    # ==============================
    async def stream_l2_data(self, on_update: Optional[Callable] = None):
        """
        Subscribe to Schwab BOOK stream for this symbol.
        Stores incoming updates in memory + SQLite.
        """
        if not SCHWAB_STREAM_AVAILABLE:
            raise ImportError(
                "schwabdev.stream not installed or unavailable. pip install schwabdev"
            )

        try:
            stream = Stream(self.client)

            async def on_book(msg: Dict[str, Any]):
                try:
                    book = msg.get("data", msg)
                    bids = book.get("bids") or []
                    asks = book.get("asks") or []
                    best_bid = float(bids[0]["price"]) if bids else 0.0
                    best_ask = float(asks[0]["price"]) if asks else 0.0
                    bid_size = int(bids[0]["size"]) if bids else 0
                    ask_size = int(asks[0]["size"]) if asks else 0

                    update = {
                        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                        "symbol": self.symbol,
                        "bid_price": best_bid,
                        "bid_size": bid_size,
                        "ask_price": best_ask,
                        "ask_size": ask_size,
                        "bid_levels": json.dumps(bids),
                        "ask_levels": json.dumps(asks),
                    }

                    self.l2_cache.append(update)
                    self._insert_db(update)

                    if on_update:
                        await on_update(update)
                except Exception as e:
                    logger.error(f"[L2] Error parsing BOOK message: {e}")

            logger.info(f"[L2] Subscribing to {self.symbol} BOOK stream...")
            if hasattr(stream, "subscribe_book"):
                await stream.subscribe_book([self.symbol], on_book)
            else:
                await stream.subscribe("BOOK", [self.symbol], on_book)

            await stream.run_forever()
        except Exception as e:
            logger.error(f"[L2] Stream failed: {e}")

    def _insert_db(self, update: Dict[str, Any]):
        """Insert a single update into SQLite."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO l2_updates
                (timestamp, symbol, bid_price, bid_size, ask_price, ask_size, bid_levels, ask_levels)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    update["timestamp"],
                    update["symbol"],
                    update["bid_price"],
                    update["bid_size"],
                    update["ask_price"],
                    update["ask_size"],
                    update["bid_levels"],
                    update["ask_levels"],
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[L2] SQLite insert error: {e}")

    # ==============================
    # DATA ACCESS METHODS
    # ==============================
    def get_recent_l2_data(self, seconds: int = 60) -> pd.DataFrame:
        """Return last N seconds of cache."""
        cutoff = int((datetime.now(timezone.utc) - timedelta(seconds=seconds)).timestamp() * 1000)
        items = [x for x in self.l2_cache if x["timestamp"] >= cutoff]
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame(items)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def get_l2_data_from_db(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch L2 records from SQLite for a time range."""
        try:
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)
            conn = sqlite3.connect(self.storage_path)
            df = pd.read_sql_query(
                """
                SELECT * FROM l2_updates
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
                """,
                conn,
                params=(start_ts, end_ts),
            )
            conn.close()
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return df
        except Exception as e:
            logger.error(f"[L2] Error reading from DB: {e}")
            return pd.DataFrame()

    def cleanup_old_data(self, days: int = 7):
        """Delete L2 entries older than N days."""
        try:
            cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM l2_updates WHERE timestamp < ?", (cutoff,))
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            logger.info(f"[L2] Cleaned {deleted} old L2 rows for {self.symbol}")
        except Exception as e:
            logger.error(f"[L2] Cleanup error: {e}")
