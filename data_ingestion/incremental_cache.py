"""
Incremental Caching System for High-Frequency Data Collection
Handles resume capability, progress tracking, and data merging
"""

import asyncio
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time

logger = logging.getLogger(__name__)

class IncrementalCache:
    """Handles incremental caching for high-frequency data collection"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.raw_dir = data_dir / 'raw'
        self.cache_dir = data_dir / 'cache'
        self.progress_file = self.cache_dir / 'collection_progress.json'
        self.checkpoint_file = self.cache_dir / 'checkpoints.json'
        
        # Create directories
        self.raw_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load progress and checkpoints
        self.progress = self._load_progress()
        self.checkpoints = self._load_checkpoints()
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load collection progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading progress: {e}")
        
        return {
            'symbols_completed': [],
            'symbols_in_progress': {},
            'last_update': None,
            'total_symbols': 0,
            'collection_start_time': None
        }
    
    def _load_checkpoints(self) -> Dict[str, Any]:
        """Load collection checkpoints"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading checkpoints: {e}")
        
        return {}
    
    def _save_progress(self):
        """Save collection progress"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def _save_checkpoints(self):
        """Save collection checkpoints"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoints, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving checkpoints: {e}")
    
    def start_collection(self, symbols: List[str], days_back: int):
        """Start a new collection session"""
        self.progress = {
            'symbols_completed': [],
            'symbols_in_progress': {},
            'last_update': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'collection_start_time': datetime.now().isoformat(),
            'days_back': days_back
        }
        self._save_progress()
        logger.info(f"Started collection for {len(symbols)} symbols, {days_back} days back")
    
    def get_symbols_to_collect(self, all_symbols: List[str]) -> List[str]:
        """Get symbols that still need to be collected"""
        completed = set(self.progress.get('symbols_completed', []))
        in_progress = set(self.progress.get('symbols_in_progress', {}).keys())
        
        # Return symbols that are neither completed nor in progress
        remaining = [s for s in all_symbols if s not in completed and s not in in_progress]
        
        logger.info(f"Symbols to collect: {len(remaining)} remaining out of {len(all_symbols)}")
        logger.info(f"Completed: {len(completed)}, In progress: {len(in_progress)}")
        
        return remaining
    
    def mark_symbol_started(self, symbol: str, days_back: int):
        """Mark a symbol as started"""
        self.progress['symbols_in_progress'][symbol] = {
            'started_at': datetime.now().isoformat(),
            'days_back': days_back,
            'last_checkpoint': None
        }
        self._save_progress()
        logger.info(f"Started collection for {symbol}")
    
    def mark_symbol_completed(self, symbol: str, data_points: int, days_covered: int):
        """Mark a symbol as completed"""
        # Remove from in_progress
        if symbol in self.progress['symbols_in_progress']:
            del self.progress['symbols_in_progress'][symbol]
        
        # Add to completed
        if symbol not in self.progress['symbols_completed']:
            self.progress['symbols_completed'].append(symbol)
        
        self.progress['last_update'] = datetime.now().isoformat()
        self._save_progress()
        
        logger.info(f"Completed {symbol}: {data_points} points, {days_covered} days")
    
    def save_checkpoint(self, symbol: str, last_timestamp: datetime, data_points: int):
        """Save a checkpoint for resuming"""
        self.checkpoints[symbol] = {
            'last_timestamp': last_timestamp.isoformat(),
            'data_points': data_points,
            'checkpoint_time': datetime.now().isoformat()
        }
        self._save_checkpoints()
        
        # Update progress
        if symbol in self.progress['symbols_in_progress']:
            self.progress['symbols_in_progress'][symbol]['last_checkpoint'] = last_timestamp.isoformat()
            self._save_progress()
    
    def get_checkpoint(self, symbol: str) -> Optional[datetime]:
        """Get the last checkpoint for a symbol"""
        if symbol in self.checkpoints:
            try:
                return datetime.fromisoformat(self.checkpoints[symbol]['last_timestamp'])
            except Exception as e:
                logger.warning(f"Error parsing checkpoint for {symbol}: {e}")
        return None
    
    def get_existing_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get existing data for a symbol"""
        file_path = self.raw_dir / f'prices_{symbol}.parquet'
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                logger.info(f"Loaded existing data for {symbol}: {len(df)} points")
                return df
            except Exception as e:
                logger.warning(f"Error loading existing data for {symbol}: {e}")
        return None
    
    def merge_data(self, symbol: str, new_data: pd.DataFrame) -> pd.DataFrame:
        """Merge new data with existing data"""
        existing_data = self.get_existing_data(symbol)
        
        if existing_data is None or existing_data.empty:
            logger.info(f"No existing data for {symbol}, using new data")
            return new_data
        
        # Handle different timestamp column names
        if 'date' in existing_data.columns and 'timestamp' not in existing_data.columns:
            # Rename 'date' to 'timestamp' for consistency
            existing_data = existing_data.rename(columns={'date': 'timestamp'})
        
        # Convert timestamp columns to datetime if needed
        if 'timestamp' in existing_data.columns:
            existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])
        if 'timestamp' in new_data.columns:
            new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
        
        # Ensure both dataframes have the same columns
        if not existing_data.empty and not new_data.empty:
            # Find common columns
            common_cols = list(set(existing_data.columns) & set(new_data.columns))
            if common_cols:
                existing_data = existing_data[common_cols]
                new_data = new_data[common_cols]
        
        # Find the last timestamp in existing data
        last_existing_timestamp = existing_data['timestamp'].max()
        
        # Filter new data to only include data after the last existing timestamp
        new_data_filtered = new_data[new_data['timestamp'] > last_existing_timestamp]
        
        if new_data_filtered.empty:
            logger.info(f"No new data for {symbol} (all data already exists)")
            return existing_data
        
        # Merge the data
        merged_data = pd.concat([existing_data, new_data_filtered], ignore_index=True)
        merged_data = merged_data.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates based on timestamp
        merged_data = merged_data.drop_duplicates(subset=['timestamp'], keep='last')
        
        logger.info(f"Merged data for {symbol}: {len(existing_data)} existing + {len(new_data_filtered)} new = {len(merged_data)} total")
        
        return merged_data
    
    def save_data(self, symbol: str, data: pd.DataFrame):
        """Save data to parquet file"""
        if data.empty:
            logger.warning(f"No data to save for {symbol}")
            return
        
        file_path = self.raw_dir / f'prices_{symbol}.parquet'
        try:
            data.to_parquet(file_path, index=False)
            logger.info(f"Saved {len(data)} data points for {symbol}")
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection status"""
        total = self.progress.get('total_symbols', 0)
        completed = len(self.progress.get('symbols_completed', []))
        in_progress = len(self.progress.get('symbols_in_progress', {}))
        remaining = total - completed - in_progress
        
        return {
            'total_symbols': total,
            'completed': completed,
            'in_progress': in_progress,
            'remaining': remaining,
            'progress_percentage': (completed / total * 100) if total > 0 else 0,
            'last_update': self.progress.get('last_update'),
            'collection_start_time': self.progress.get('collection_start_time')
        }
    
    def cleanup_failed_symbols(self, max_age_hours: int = 24):
        """Clean up symbols that have been in progress for too long (likely failed)"""
        now = datetime.now()
        failed_symbols = []
        
        for symbol, info in self.progress.get('symbols_in_progress', {}).items():
            try:
                started_at = datetime.fromisoformat(info['started_at'])
                age_hours = (now - started_at).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    failed_symbols.append(symbol)
            except Exception as e:
                logger.warning(f"Error checking age for {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            logger.info(f"Cleaning up {len(failed_symbols)} failed symbols: {failed_symbols}")
            for symbol in failed_symbols:
                del self.progress['symbols_in_progress'][symbol]
            
            self._save_progress()
    
    def reset_collection(self, symbols: List[str] = None):
        """Reset collection progress (use with caution)"""
        if symbols:
            # Reset specific symbols
            for symbol in symbols:
                if symbol in self.progress['symbols_completed']:
                    self.progress['symbols_completed'].remove(symbol)
                if symbol in self.progress['symbols_in_progress']:
                    del self.progress['symbols_in_progress'][symbol]
                if symbol in self.checkpoints:
                    del self.checkpoints[symbol]
        else:
            # Reset all
            self.progress = {
                'symbols_completed': [],
                'symbols_in_progress': {},
                'last_update': None,
                'total_symbols': 0,
                'collection_start_time': None
            }
            self.checkpoints = {}
        
        self._save_progress()
        self._save_checkpoints()
        logger.info(f"Reset collection progress for {symbols or 'all symbols'}")
