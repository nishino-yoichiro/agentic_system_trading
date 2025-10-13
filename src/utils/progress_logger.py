#!/usr/bin/env python3
"""
Progress Logger Utility
======================

Centralized logging utility with tqdm progress bars for backtesting and data processing.
This provides a clean way to show progress across all backtesting operations.

Author: Quantitative Strategy Designer
Date: 2025-01-28
"""

import logging
import sys
from typing import List, Optional, Union, Iterable, Any
from tqdm import tqdm
from pathlib import Path
import time
from datetime import datetime

class ProgressLogger:
    """Enhanced logger with progress bar support"""
    
    def __init__(self, name: str = None, level: int = logging.INFO):
        self.logger = logging.getLogger(name or __name__)
        self.logger.setLevel(level)
        
        # Add console handler if not already present
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def progress_bar(self, 
                    iterable: Iterable, 
                    desc: str = "Processing", 
                    total: Optional[int] = None,
                    unit: str = "it",
                    leave: bool = True,
                    ncols: int = 100,
                    mininterval: float = 0.1) -> tqdm:
        """
        Create a progress bar for an iterable
        
        Args:
            iterable: The iterable to wrap with progress bar
            desc: Description for the progress bar
            total: Total number of items (if not inferrable from iterable)
            unit: Unit of measurement
            leave: Whether to leave the progress bar after completion
            ncols: Width of the progress bar
            mininterval: Minimum update interval in seconds
        
        Returns:
            tqdm progress bar object
        """
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            unit=unit,
            leave=leave,
            ncols=ncols,
            mininterval=mininterval,
            file=sys.stdout,
            dynamic_ncols=True
        )
    
    def backtest_progress(self, 
                         symbols: List[str], 
                         strategies: List[str] = None,
                         desc: str = "Backtesting") -> tqdm:
        """
        Create a progress bar specifically for backtesting operations
        
        Args:
            symbols: List of symbols being backtested
            strategies: List of strategies being tested
            desc: Description for the progress bar
        
        Returns:
            tqdm progress bar object
        """
        total_items = len(symbols)
        if strategies:
            total_items *= len(strategies)
        
        strategy_info = f" ({len(strategies)} strategies)" if strategies else ""
        full_desc = f"{desc} {len(symbols)} symbols{strategy_info}"
        
        return tqdm(
            total=total_items,
            desc=full_desc,
            unit="symbol",
            ncols=120,
            mininterval=0.5,
            file=sys.stdout,
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
    
    def signal_generation_progress(self, 
                                  data_points: int,
                                  symbol: str,
                                  strategy: str = None) -> tqdm:
        """
        Create a progress bar for signal generation
        
        Args:
            data_points: Number of data points to process
            symbol: Symbol being processed
            strategy: Strategy being used
        
        Returns:
            tqdm progress bar object
        """
        desc = f"Generating signals for {symbol}"
        if strategy:
            desc += f" ({strategy})"
        
        return tqdm(
            total=data_points,
            desc=desc,
            unit="bar",
            ncols=120,
            mininterval=0.1,
            file=sys.stdout,
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
    
    def parameter_sweep_progress(self, 
                                total_combinations: int,
                                symbols: List[str],
                                strategies: List[str]) -> tqdm:
        """
        Create a progress bar for parameter sweep operations
        
        Args:
            total_combinations: Total number of parameter combinations to test
            symbols: List of symbols
            strategies: List of strategies
        
        Returns:
            tqdm progress bar object
        """
        desc = f"Parameter sweep: {len(symbols)} symbols × {len(strategies)} strategies"
        
        return tqdm(
            total=total_combinations,
            desc=desc,
            unit="combo",
            ncols=120,
            mininterval=0.5,
            file=sys.stdout,
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
    
    def walk_forward_progress(self, 
                             total_periods: int,
                             start_date: datetime,
                             end_date: datetime) -> tqdm:
        """
        Create a progress bar for walk-forward backtesting
        
        Args:
            total_periods: Total number of periods to process
            start_date: Start date of backtest
            end_date: End date of backtest
        
        Returns:
            tqdm progress bar object
        """
        date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        desc = f"Walk-forward backtest ({date_range})"
        
        return tqdm(
            total=total_periods,
            desc=desc,
            unit="period",
            ncols=120,
            mininterval=0.5,
            file=sys.stdout,
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )

# Global instance for easy access
progress_logger = ProgressLogger("crypto_pipeline")

# Convenience functions for common operations
def log_info(message: str):
    """Log info message"""
    progress_logger.info(message)

def log_warning(message: str):
    """Log warning message"""
    progress_logger.warning(message)

def log_error(message: str):
    """Log error message"""
    progress_logger.error(message)

def log_debug(message: str):
    """Log debug message"""
    progress_logger.debug(message)

def create_progress_bar(iterable: Iterable, desc: str = "Processing", **kwargs) -> tqdm:
    """Create a progress bar for any iterable"""
    return progress_logger.progress_bar(iterable, desc, **kwargs)

def create_backtest_progress(symbols: List[str], strategies: List[str] = None, desc: str = "Backtesting") -> tqdm:
    """Create a progress bar for backtesting operations"""
    return progress_logger.backtest_progress(symbols, strategies, desc)

def create_signal_progress(data_points: int, symbol: str, strategy: str = None) -> tqdm:
    """Create a progress bar for signal generation"""
    return progress_logger.signal_generation_progress(data_points, symbol, strategy)

def create_parameter_sweep_progress(total_combinations: int, symbols: List[str], strategies: List[str]) -> tqdm:
    """Create a progress bar for parameter sweep operations"""
    return progress_logger.parameter_sweep_progress(total_combinations, symbols, strategies)

def create_walk_forward_progress(total_periods: int, start_date: datetime, end_date: datetime) -> tqdm:
    """Create a progress bar for walk-forward backtesting"""
    return progress_logger.walk_forward_progress(total_periods, start_date, end_date)

# Example usage
if __name__ == "__main__":
    # Test the progress logger
    logger = ProgressLogger("test")
    
    # Test basic logging
    logger.info("Testing progress logger")
    
    # Test progress bar
    items = list(range(100))
    for item in logger.progress_bar(items, desc="Testing progress"):
        time.sleep(0.01)
    
    # Test backtest progress
    symbols = ["BTC", "ETH", "ADA"]
    strategies = ["reversal", "momentum"]
    pbar = logger.backtest_progress(symbols, strategies)
    for i in range(6):  # 3 symbols × 2 strategies
        time.sleep(0.5)
        pbar.set_postfix({"Current": f"Symbol {i//2 + 1}, Strategy {i%2 + 1}"})
        pbar.update(1)
    pbar.close()
    
    print("Progress logger test completed!")

