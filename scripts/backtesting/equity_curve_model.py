"""
Professional Equity Curve Data Model
Handles dimensional explosion through unified abstraction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class EquityCurve:
    """
    Unified representation of any backtest result.
    All downstream metrics operate on this normalized structure.
    """
    id: str                    # e.g. "BTC_btc_ny_session"
    symbol: str
    strategy: str
    timestamps: np.ndarray     # datetime array
    equity: np.ndarray         # normalized to start at 1.0
    returns: np.ndarray        # pct changes for Sharpe/Sortino
    trades: List[Dict]         # raw trade data
    meta: Dict                 # backtest metadata
    
    # Computed metrics (lazy-loaded)
    _rolling_sharpe: Dict[int, np.ndarray] = field(default_factory=dict)
    _segmented_performance: Dict[str, Dict] = field(default_factory=dict)
    _regime_performance: Dict[str, Dict] = field(default_factory=dict)
    _optimal_conditions: List[Tuple[str, float, str]] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure data consistency after initialization"""
        if len(self.timestamps) != len(self.equity):
            raise ValueError(f"Timestamps ({len(self.timestamps)}) and equity ({len(self.equity)}) length mismatch")
        
        if len(self.returns) != len(self.equity) - 1:
            raise ValueError(f"Returns ({len(self.returns)}) should be equity length - 1 ({len(self.equity) - 1})")
    
    @property
    def total_return(self) -> float:
        """Total return from start to end"""
        if len(self.equity) == 0:
            return 0.0
        initial_capital = self.meta.get('initial_capital', self.equity[0])
        final_capital = self.equity[-1]
        return float((final_capital - initial_capital) / initial_capital)
    
    @property
    def sharpe_ratio(self) -> float:
        """Overall Sharpe ratio"""
        if len(self.returns) == 0:
            return 0.0
        return float(np.mean(self.returns) / np.std(self.returns)) if np.std(self.returns) > 0 else 0.0
    
    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown"""
        peak = np.maximum.accumulate(self.equity)
        drawdown = (self.equity - peak) / peak
        return float(np.min(drawdown))
    
    @property
    def win_rate(self) -> float:
        """Win rate from trades"""
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
        return winning_trades / len(self.trades) if self.trades else 0.0
    
    @property
    def total_trades(self) -> int:
        """Total number of trades"""
        return len(self.trades)
    
    def normalize_to_one(self) -> 'EquityCurve':
        """Normalize equity curve to start at 1.0"""
        if self.equity[0] == 0:
            logger.warning(f"Zero initial equity for {self.id}, skipping normalization")
            return self
        
        normalized_equity = self.equity / self.equity[0]
        return EquityCurve(
            id=self.id,
            symbol=self.symbol,
            strategy=self.strategy,
            timestamps=self.timestamps,
            equity=normalized_equity,
            returns=self.returns,
            trades=self.trades,
            meta=self.meta
        )
    
    def get_rolling_sharpe(self, window: int) -> np.ndarray:
        """Get rolling Sharpe for specific window (lazy-loaded)"""
        if window not in self._rolling_sharpe:
            if len(self.returns) < window:
                logger.warning(f"Insufficient data for {window}-period rolling Sharpe: {len(self.returns)} < {window}")
                self._rolling_sharpe[window] = np.array([])
            else:
                rolling_mean = pd.Series(self.returns).rolling(window=window).mean()
                rolling_std = pd.Series(self.returns).rolling(window=window).std()
                rolling_sharpe = rolling_mean / rolling_std
                self._rolling_sharpe[window] = rolling_sharpe.fillna(0).values
        
        return self._rolling_sharpe[window]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'strategy': self.strategy,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'meta': self.meta
        }


class RollingWindowManager:
    """
    Dynamic window size selection based on backtest duration
    """
    
    @staticmethod
    def select_windows(curve: EquityCurve, percentages: List[float] = None) -> List[int]:
        """
        Select rolling windows as percentage of total data points
        
        Args:
            curve: EquityCurve to analyze
            percentages: List of percentages (default: [0.01, 0.03, 0.07])
        
        Returns:
            List of window sizes
        """
        if percentages is None:
            percentages = [0.01, 0.03, 0.07]  # short, medium, long
        
        n = len(curve.timestamps)
        windows = [max(5, int(n * p)) for p in percentages]
        
        # Ensure windows don't exceed data length
        windows = [w for w in windows if w <= n]
        
        logger.info(f"Selected windows for {curve.id}: {windows} (from {n} data points)")
        return windows


class NormalizationPolicy:
    """
    Consistent normalization rules for different comparison contexts
    """
    
    @staticmethod
    def normalize_multi_strategy(curves: List[EquityCurve]) -> List[EquityCurve]:
        """Normalize multiple strategies for same symbol comparison"""
        return [curve.normalize_to_one() for curve in curves]
    
    @staticmethod
    def normalize_multi_symbol(curves: List[EquityCurve], weights: Dict[str, float] = None) -> List[EquityCurve]:
        """Normalize multiple symbols for portfolio comparison"""
        if weights is None:
            # Equal weight
            symbols = list(set(curve.symbol for curve in curves))
            weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
        
        normalized = []
        for curve in curves:
            weight = weights.get(curve.symbol, 1.0)
            weighted_equity = curve.equity * weight
            normalized.append(EquityCurve(
                id=f"{curve.id}_weighted",
                symbol=curve.symbol,
                strategy=curve.strategy,
                timestamps=curve.timestamps,
                equity=weighted_equity,
                returns=curve.returns,
                trades=curve.trades,
                meta={**curve.meta, 'weight': weight}
            ))
        
        return normalized


class EquityCurveFactory:
    """
    Factory to create EquityCurve objects from various backtest result formats
    """
    
    @staticmethod
    def from_backtest_result(result, symbol: str, strategy: str) -> EquityCurve:
        """Convert BacktestResult to EquityCurve"""
        # Extract timestamps and equity
        if hasattr(result, 'equity_curve') and hasattr(result.equity_curve, 'index'):
            timestamps = result.equity_curve.index.values
            equity = result.equity_curve.values
        else:
            # Fallback for simple equity series
            timestamps = np.array([result.start_date, result.end_date])
            equity = np.array([result.initial_capital, result.final_capital])
        
        # Calculate returns
        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
        else:
            returns = np.array([])
        
        # Create metadata
        meta = {
            'start_date': result.start_date,
            'end_date': result.end_date,
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'backtest_duration_days': (pd.to_datetime(result.end_date) - pd.to_datetime(result.start_date)).days
        }
        
        curve_id = f"{symbol}_{strategy}"
        
        return EquityCurve(
            id=curve_id,
            symbol=symbol,
            strategy=strategy,
            timestamps=timestamps,
            equity=equity,
            returns=returns,
            trades=getattr(result, 'trades', []),
            meta=meta
        )
    
    @staticmethod
    def from_trades(trades: List[Dict], symbol: str, strategy: str, 
                   initial_capital: float = 100000) -> EquityCurve:
        """Create EquityCurve from raw trade data"""
        if not trades:
            # Empty curve
            timestamps = np.array([datetime.now()])
            equity = np.array([initial_capital])
            returns = np.array([])
        else:
            # Extract timestamps and calculate equity progression
            timestamps = np.array([trade['timestamp'] for trade in trades])
            
            # Calculate cumulative equity
            equity = [initial_capital]
            for trade in trades:
                pnl = trade.get('pnl', 0)
                equity.append(equity[-1] + pnl)
            
            equity = np.array(equity)
            
            # Calculate returns
            if len(equity) > 1:
                returns = np.diff(equity) / equity[:-1]
            else:
                returns = np.array([])
        
        meta = {
            'initial_capital': initial_capital,
            'final_capital': equity[-1] if len(equity) > 0 else initial_capital,
            'trade_count': len(trades)
        }
        
        curve_id = f"{symbol}_{strategy}"
        
        return EquityCurve(
            id=curve_id,
            symbol=symbol,
            strategy=strategy,
            timestamps=timestamps,
            equity=equity,
            returns=returns,
            trades=trades,
            meta=meta
        )

