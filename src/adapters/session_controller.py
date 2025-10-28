"""
Session Controller
==================

Manages trading sessions for equity markets.
Handles market open/close, DST transitions, and session boundaries.
"""

import sys
from datetime import datetime, time, timedelta
from typing import Optional
from pathlib import Path
import logging

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False

try:
    from pandas_market_calendars import get_calendar
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False
    get_calendar = None

logger = logging.getLogger(__name__)


class SessionController:
    """
    Manages trading sessions for market-aware strategies.
    
    Features:
    - Market hours detection (9:30-16:00 ET for US equities)
    - Pre-market and after-hours awareness
    - DST-aware timezone handling
    - Trading day validation
    - Session boundaries
    """
    
    def __init__(self, exchange: str = 'NYSE'):
        """
        Initialize session controller.
        
        Args:
            exchange: Exchange calendar ('NYSE', 'NASDAQ', etc.)
        """
        self.exchange = exchange
        
        if CALENDAR_AVAILABLE:
            self.calendar = get_calendar(exchange)
        else:
            self.calendar = None
            logger.warning("pandas-market-calendars not available")
        
        if PYTZ_AVAILABLE:
            self.et_tz = pytz.timezone('US/Eastern')
        else:
            self.et_tz = None
            logger.warning("pytz not available")
        
    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if market is currently open.
        
        Args:
            dt: Datetime to check (defaults to now)
            
        Returns:
            bool: True if market is open
        """
        if not PYTZ_AVAILABLE or not CALENDAR_AVAILABLE:
            # Fallback: check time manually
            now = datetime.now()
            market_open = time(9, 30)
            market_close = time(16, 0)
            return market_open <= now.time() <= market_close
        
        if dt is None:
            dt = datetime.now(self.et_tz)
        
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = self.et_tz.localize(dt)
        
        # Convert to Eastern Time
        et = dt.astimezone(self.et_tz)
        date = et.date()
        
        # Check if it's a trading day
        schedule = self.calendar.schedule(start_date=date, end_date=date)
        
        if schedule.empty:
            return False
        
        # Check if current time is within trading hours
        market_open = schedule.iloc[0]['market_open'].to_pydatetime().replace(tzinfo=pytz.UTC).astimezone(self.et_tz)
        market_close = schedule.iloc[0]['market_close'].to_pydatetime().replace(tzinfo=pytz.UTC).astimezone(self.et_tz)
        
        return market_open <= et <= market_close
    
    def get_next_open(self, dt: Optional[datetime] = None) -> datetime:
        """
        Get the next market open time.
        
        Args:
            dt: Reference datetime (defaults to now)
            
        Returns:
            datetime: Next market open
        """
        if dt is None:
            dt = datetime.now(self.et_tz)
        
        # If market is currently open, return current day's open
        if self.is_market_open(dt):
            et = dt.astimezone(self.et_tz)
            date = et.date()
            schedule = self.calendar.schedule(start_date=date, end_date=date)
            return schedule.iloc[0]['market_open'].to_pydatetime().replace(tzinfo=pytz.UTC).astimezone(self.et_tz)
        
        # Find next trading day
        et = dt.astimezone(self.et_tz)
        date = et.date()
        
        # Look ahead up to 7 days
        for days_ahead in range(1, 8):
            check_date = date + timedelta(days=days_ahead)
            schedule = self.calendar.schedule(start_date=check_date, end_date=check_date)
            
            if not schedule.empty:
                market_open = schedule.iloc[0]['market_open'].to_pydatetime().replace(tzinfo=pytz.UTC).astimezone(self.et_tz)
                return market_open
        
        # If no trading day found in next week, return a fallback
        logger.warning("No trading day found in next 7 days")
        return dt + timedelta(days=1)
    
    def get_market_open_time(self, date: datetime) -> Optional[datetime]:
        """
        Get market open time for a specific date.
        
        Args:
            date: Date to check
            
        Returns:
            datetime: Market open time, or None if not a trading day
        """
        schedule = self.calendar.schedule(start_date=date.date(), end_date=date.date())
        
        if schedule.empty:
            return None
        
        return schedule.iloc[0]['market_open'].to_pydatetime().replace(tzinfo=pytz.UTC)
    
    def get_market_close_time(self, date: datetime) -> Optional[datetime]:
        """
        Get market close time for a specific date.
        
        Args:
            date: Date to check
            
        Returns:
            datetime: Market close time, or None if not a trading day
        """
        schedule = self.calendar.schedule(start_date=date.date(), end_date=date.date())
        
        if schedule.empty:
            return None
        
        return schedule.iloc[0]['market_close'].to_pydatetime().replace(tzinfo=pytz.UTC)
    
    def is_pre_market(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if current time is pre-market (before 9:30 ET).
        
        Args:
            dt: Datetime to check (defaults to now)
            
        Returns:
            bool: True if pre-market
        """
        if dt is None:
            dt = datetime.now(self.et_tz)
        
        et = dt.astimezone(self.et_tz)
        market_open = time(9, 30)
        
        return et.time() < market_open and self.is_trading_day(et.date())
    
    def is_after_hours(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if current time is after-hours (after 16:00 ET).
        
        Args:
            dt: Datetime to check (defaults to now)
            
        Returns:
            bool: True if after-hours
        """
        if dt is None:
            dt = datetime.now(self.et_tz)
        
        et = dt.astimezone(self.et_tz)
        market_close = time(16, 0)
        
        return et.time() > market_close and self.is_trading_day(et.date())
    
    def is_trading_day(self, date) -> bool:
        """
        Check if a date is a trading day.
        
        Args:
            date: Date to check (date or datetime)
            
        Returns:
            bool: True if trading day
        """
        if isinstance(date, datetime):
            date = date.date()
        
        schedule = self.calendar.schedule(start_date=date, end_date=date)
        return not schedule.empty
    
    def wait_until_open(self, dt: Optional[datetime] = None) -> float:
        """
        Calculate seconds until market opens.
        
        Args:
            dt: Reference datetime (defaults to now)
            
        Returns:
            float: Seconds until market opens
        """
        next_open = self.get_next_open(dt)
        
        if dt is None:
            dt = datetime.now(self.et_tz)
        
        wait_seconds = (next_open - dt).total_seconds()
        return max(0, wait_seconds)
    
    def format_market_status(self, dt: Optional[datetime] = None) -> str:
        """
        Get human-readable market status.
        
        Args:
            dt: Datetime to check (defaults to now)
            
        Returns:
            str: Market status description
        """
        if dt is None:
            dt = datetime.now(self.et_tz)
        
        if not self.is_trading_day(dt):
            return "Market Closed (Not a trading day)"
        
        if self.is_market_open(dt):
            close_time = self.get_market_close_time(dt)
            if close_time:
                return f"Market Open (Closes at {close_time.strftime('%H:%M ET')})"
            return "Market Open"
        
        if self.is_pre_market(dt):
            open_time = self.get_next_open(dt)
            return f"Pre-Market (Opens at {open_time.strftime('%H:%M ET')})"
        
        if self.is_after_hours(dt):
            return "After-Hours"
        
        return "Market Closed"
