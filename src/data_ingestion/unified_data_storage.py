#!/usr/bin/env python3
"""
Unified Data Storage Manager
============================

Single class for managing data storage for ANY market (crypto, equities, etc.).
Handles:
- Parquet file management
- Append-only storage
- Live bar accumulation
- Gap detection
- Market-agnostic (works for both)

API calls are handled by ADAPTERS.
Storage is handled HERE.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class UnifiedDataStorage:
    """
    Unified storage manager for all markets.
    
    Works for:
    - Crypto (24/7 data)
    - Equities (market hours only)
    - Future markets (commodities, forex, etc.)
    
    Storage pattern:
    - {market_type}_db/{symbol}_historical.parquet  (bulk data)
    - {market_type}_db/{symbol}_1m.parquet          (live bars)
    """
    
    def __init__(self, market_type: str = "crypto"):
        """
        Initialize storage for a specific market type.
        
        Args:
            market_type: 'crypto', 'equities', etc.
        """
        self.market_type = market_type
        self.db_dir = Path(f"data/{market_type}_db")
        self.db_dir.mkdir(exist_ok=True, parents=True)
    
    def load_historical(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load historical data for a symbol.
        
        Returns: DataFrame with OHLCV data, or None if not found
        """
        try:
            db_file = self.db_dir / f"{symbol}_historical.parquet"
            if not db_file.exists():
                logger.info(f"No historical data found for {symbol}")
                return None
            
            df = pd.read_parquet(db_file)
            if df.empty:
                return None
            
            # Ensure proper datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    df = df.set_index('timestamp')
                else:
                    df.index = pd.to_datetime(df.index, utc=True)
            
            # Ensure timezone-aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            logger.info(f"Loaded {len(df)} historical points for {symbol} ({self.market_type})")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return None
    
    def save_historical(self, symbol: str, df: pd.DataFrame, append: bool = True):
        """
        Save historical data (append-only).
        
        Args:
            symbol: Trading symbol
            df: DataFrame with OHLCV data
            append: If True, merge with existing data; if False, replace
        """
        try:
            db_file = self.db_dir / f"{symbol}_historical.parquet"
            
            if db_file.exists() and append:
                # Load existing data and merge
                existing_df = pd.read_parquet(db_file)
                
                if not existing_df.empty:
                    # Ensure proper datetime index for both
                    for d in [existing_df, df]:
                        if not isinstance(d.index, pd.DatetimeIndex):
                            if 'timestamp' in d.columns:
                                d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True)
                                d = d.set_index('timestamp')
                            else:
                                d.index = pd.to_datetime(d.index, utc=True)
                        if d.index.tz is None:
                            d.index = d.index.tz_localize('UTC')
                    
                    # Combine and remove duplicates
                    combined_df = pd.concat([existing_df, df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df = combined_df.sort_index()
                    
                    logger.info(f"Merged {symbol}: {len(existing_df)} existing + {len(df)} new = {len(combined_df)} total")
                    df = combined_df
            
            # Save to database
            df.to_parquet(db_file)
            logger.info(f"Saved {len(df)} data points for {symbol} ({self.market_type})")
            
        except Exception as e:
            logger.error(f"Error saving historical data for {symbol}: {e}")
    
    def load_live_bars(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load accumulated live 1-minute bars"""
        try:
            live_file = self.db_dir / f"{symbol}_1m.parquet"
            if not live_file.exists():
                return None
            
            df = pd.read_parquet(live_file)
            
            # Ensure proper datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    df = df.set_index('timestamp')
                else:
                    df.index = pd.to_datetime(df.index, utc=True)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading live bars for {symbol}: {e}")
            return None
    
    def save_live_bar(self, symbol: str, bar: pd.DataFrame, merge_threshold: int = 60):
        """
        Save a live 1-minute bar.
        
        Args:
            symbol: Trading symbol
            bar: Single 1-minute bar as DataFrame
            merge_threshold: Merge to historical after N bars accumulated
        """
        try:
            live_file = self.db_dir / f"{symbol}_1m.parquet"
            
            if live_file.exists():
                # Append to existing live data
                existing_df = pd.read_parquet(live_file)
                combined = pd.concat([existing_df, bar])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
            else:
                combined = bar
            
            # Save live data
            combined.to_parquet(live_file)
            
            # Periodically merge to historical (to avoid live file growing too large)
            if len(combined) >= merge_threshold:
                self._merge_live_to_historical(symbol, combined)
                # Keep only recent bars (last hour)
                cutoff = datetime.now() - timedelta(hours=1)
                cutoff = pd.Timestamp(cutoff, tz='UTC')
                recent = combined[combined.index >= cutoff]
                recent.to_parquet(live_file)
                
        except Exception as e:
            logger.error(f"Error saving live bar for {symbol}: {e}")
    
    def _merge_live_to_historical(self, symbol: str, live_df: pd.DataFrame):
        """Merge accumulated live data to historical database"""
        try:
            logger.info(f"Merging live data to historical for {symbol}")
            
            # Save to historical (append-only)
            self.save_historical(symbol, live_df, append=True)
            
        except Exception as e:
            logger.error(f"Error merging live to historical for {symbol}: {e}")
    
    def get_combined_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get combined historical + live data.
        
        Returns: Full dataset (historical + any recent live bars)
        """
        # Load historical
        historical_df = self.load_historical(symbol)
        if historical_df is None or historical_df.empty:
            historical_df = pd.DataFrame()
        
        # Load live bars
        live_df = self.load_live_bars(symbol)
        
        if live_df is not None and not live_df.empty:
            # Combine
            if not historical_df.empty:
                combined = pd.concat([historical_df, live_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
            else:
                combined = live_df
        else:
            combined = historical_df
        
        # Filter to last N days
        if not combined.empty:
            cutoff = datetime.now() - timedelta(days=days)
            cutoff = pd.Timestamp(cutoff, tz='UTC')
            combined = combined[combined.index >= cutoff]
        
        return combined
    
    def detect_gaps(self, symbol: str, expected_frequency_minutes: int = 1) -> List[tuple]:
        """
        Detect gaps in data.
        
        Args:
            symbol: Trading symbol
            expected_frequency_minutes: Expected bar interval (default 1 minute)
            
        Returns:
            List of (start, end) tuples for gaps
        """
        try:
            df = self.load_historical(symbol)
            if df is None or len(df) < 2:
                return []
            
            # Calculate expected time between bars
            expected_delta = pd.Timedelta(minutes=expected_frequency_minutes)
            
            gaps = []
            for i in range(len(df) - 1):
                time_diff = df.index[i + 1] - df.index[i]
                
                # If gap is more than 2x expected (allow some tolerance)
                if time_diff > expected_delta * 2:
                    gaps.append((df.index[i], df.index[i + 1]))
            
            if gaps:
                logger.warning(f"Detected {len(gaps)} gaps in {symbol}")
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error detecting gaps for {symbol}: {e}")
            return []

