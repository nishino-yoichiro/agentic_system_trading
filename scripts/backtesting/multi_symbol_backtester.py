#!/usr/bin/env python3
"""
Multi-Symbol Crypto Backtester
Professional backtesting framework with enhanced metrics and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
import argparse
import json
from dataclasses import dataclass
import warnings
from tqdm import tqdm
import bisect

# Import professional architecture
from equity_curve_model import EquityCurve, EquityCurveFactory, RollingWindowManager, NormalizationPolicy
from professional_report_generator import ReportGenerator, ReportConfig

warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from crypto_signal_integration import CryptoSignalIntegration
from crypto_analysis_engine import CryptoAnalysisEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Container for individual symbol backtest results"""
    symbol: str
    strategy: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    equity_curve: pd.Series
    trades: List[Dict]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    # Enhanced metrics
    rolling_sharpe: Dict[int, pd.Series] = None
    segmented_performance: Dict[str, Dict] = None
    regime_performance: Dict[str, Dict] = None
    optimal_conditions: List[Tuple[str, float, str]] = None

@dataclass
class PortfolioResult:
    """Container for portfolio backtest results"""
    symbols: List[str]
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    final_capital: float
    individual_results: Dict[str, BacktestResult]
    portfolio_equity_curve: pd.Series
    rebalance_dates: List[datetime]

class MultiSymbolBacktester:
    """Professional multi-symbol crypto backtester with enhanced metrics"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.analysis_engine = CryptoAnalysisEngine()
        self.signal_integration = CryptoSignalIntegration()
        
        # Get available symbols from data directory
        self.available_symbols = self._get_available_symbols()
        logger.info(f"Available symbols: {self.available_symbols}")
        
    def _get_available_symbols(self) -> List[str]:
        """Get available symbols from data directory"""
        data_dir = Path("data/crypto_db")
        if not data_dir.exists():
            logger.warning("Data directory not found, using default symbols")
            return ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']
        
        symbols = []
        for file_path in data_dir.glob("*_historical.parquet"):
            symbol = file_path.stem.replace("_historical", "").upper()
            symbols.append(symbol)
        
        if not symbols:
            logger.warning("No symbol data found, using default symbols")
            return ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']
        
        return sorted(symbols)
    
    def load_symbol_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Load historical data for a symbol"""
        try:
            df = self.analysis_engine.load_symbol_data(symbol, days=days)
            logger.info(f"Loaded {len(df)} data points for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_signals_with_framework(self, symbol: str, days: int = 30, 
                                       strategies: List[str] = None) -> pd.DataFrame:
        """Generate signals using the professional framework"""
        try:
            logger.info(f"Generating signals for symbol: '{symbol}' (type: {type(symbol)})")
            logger.info(f"Symbols list: {[symbol]} (type: {type([symbol])})")
            
            # Convert single symbol to list as expected by the framework
            signals_list = self.signal_integration.generate_signals([symbol], days=days, strategies=strategies)
            
            if not signals_list:
                logger.warning(f"No signals generated for {symbol}")
                return pd.DataFrame()
            
            # Convert list of signals to DataFrame
            signals_df = pd.DataFrame(signals_list)
            logger.info(f"Generated {len(signals_df)} signals for {symbol} using new framework")
            return signals_df
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_rolling_metrics(self, returns: pd.Series, windows: List[int] = None) -> Dict[int, pd.Series]:
        """Calculate rolling Sharpe ratios for different windows"""
        rolling_sharpe = {}
        
        # Auto-adjust windows based on data length (proportional to backtest period)
        if windows is None:
            data_length = len(returns)
            # Use windows that are proportional to the data length
            # For 30-day backtest with ~43k data points, use much smaller windows
            if data_length < 100:
                windows = [3, 5, 7]  # Very short data
            elif data_length < 1000:
                windows = [5, 10, 15]  # Short data (few days)
            elif data_length < 10000:
                windows = [10, 20, 30]  # Medium data (few weeks)
            elif data_length < 50000:
                windows = [20, 50, 100]  # Longer data (month+)
            else:
                windows = [50, 100, 200]  # Very long data (months+)
        
        logger.info(f"Calculating rolling metrics for {len(returns)} returns with windows: {windows}")
        
        for window in windows:
            if len(returns) < window:
                logger.warning(f"Insufficient data for {window}-period rolling analysis: {len(returns)} < {window}")
                continue
                
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            rolling_sharpe[window] = rolling_mean / rolling_std
            
            logger.info(f"Calculated {window}-period rolling Sharpe: {len(rolling_sharpe[window])} values")
        
        logger.info(f"Rolling Sharpe calculation complete: {len(rolling_sharpe)} windows")
        return rolling_sharpe
    
    def calculate_segmented_performance(self, trades: List[Dict], returns: pd.Series) -> Dict[str, Dict]:
        """Calculate performance segmented by time and volatility"""
        if not trades:
            return {}
        
        logger.info(f"Calculating segmented performance for {len(trades)} trades")
        
        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        segmented_performance = {}
        
        # Time-based segmentation
        segmented_performance.update(self._segment_by_time(trades_df, returns))
        
        # Volatility-based segmentation
        segmented_performance.update(self._segment_by_volatility(trades_df, returns))
        
        logger.info(f"Total segmented performance: {list(segmented_performance.keys())}")
        return segmented_performance
    
    def _segment_by_time(self, trades_df: pd.DataFrame, returns: pd.Series) -> Dict[str, Dict]:
        """Segment performance by time periods"""
        segments = {}
        
        # Hour of day
        if len(trades_df) > 0:
            # Convert timestamp to datetime if it's a string
            if trades_df['timestamp'].dtype == 'object':
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df['hour'] = trades_df['timestamp'].dt.hour
            hourly_performance = {}
            
            for hour in range(24):
                hour_trades = trades_df[trades_df['hour'] == hour]
                if len(hour_trades) > 0:
                    hour_returns = [trade.get('pnl', 0) for trade in hour_trades.to_dict('records')]
                    if hour_returns:
                        sharpe = np.mean(hour_returns) / np.std(hour_returns) if np.std(hour_returns) > 0 else 0
                        hourly_performance[hour] = {
                            'sharpe_ratio': sharpe,
                            'count': len(hour_trades),
                            'avg_return': np.mean(hour_returns)
                        }
            
            if hourly_performance:
                segments['hourly'] = hourly_performance
        
        # Day of week
        if len(trades_df) > 0:
            # Convert timestamp to datetime if it's a string (already done above, but ensure it's done)
            if trades_df['timestamp'].dtype == 'object':
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df['dayofweek'] = trades_df['timestamp'].dt.dayofweek
            dow_performance = {}
            
            for day in range(7):
                day_trades = trades_df[trades_df['dayofweek'] == day]
                if len(day_trades) > 0:
                    day_returns = [trade.get('pnl', 0) for trade in day_trades.to_dict('records')]
                    if day_returns:
                        sharpe = np.mean(day_returns) / np.std(day_returns) if np.std(day_returns) > 0 else 0
                        dow_performance[day] = {
                            'sharpe_ratio': sharpe,
                            'count': len(day_trades),
                            'avg_return': np.mean(day_returns)
                        }
            
            if dow_performance:
                segments['dayofweek'] = dow_performance
        
        logger.info(f"Time segments: {list(segments.keys())}")
        return segments
    
    def _segment_by_volatility(self, trades_df: pd.DataFrame, returns: pd.Series) -> Dict[str, Dict]:
        """Segment performance by volatility regimes"""
        if len(returns) < 10:
            logger.warning("Insufficient data for volatility segmentation")
            return {}
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=min(20, len(returns)//2)).std()
        vol_median = rolling_vol.median()
        
        # Ensure rolling_vol.index is timezone-aware DatetimeIndex
        if hasattr(rolling_vol.index, 'tz') and rolling_vol.index.tz is None:
            rolling_vol.index = rolling_vol.index.tz_localize('UTC')
        elif not hasattr(rolling_vol.index, 'tz'):
            # If it's not a DatetimeIndex, convert it
            rolling_vol.index = pd.to_datetime(rolling_vol.index, utc=True)
        
        volatility_segments = {}
        
        for _, trade in trades_df.iterrows():
            trade_time = trade['timestamp']
            
            # Ensure trade_time is a pandas Timestamp for comparison
            if isinstance(trade_time, str):
                trade_time = pd.to_datetime(trade_time)
            
            # Ensure trade_time is timezone-aware for comparison
            if trade_time.tz is None:
                trade_time = trade_time.tz_localize('UTC')
            
            # Find closest volatility reading
            vol_at_time = rolling_vol.loc[rolling_vol.index <= trade_time]
            if len(vol_at_time) > 0:
                vol_at_time = vol_at_time.iloc[-1]
                
                if pd.notna(vol_at_time):
                    regime = 'High' if vol_at_time > vol_median else 'Low'
                    
                    if regime not in volatility_segments:
                        volatility_segments[regime] = []
                    
                    volatility_segments[regime].append(trade.get('pnl', 0))
        
        # Calculate metrics for each regime
        volatility_performance = {}
        for regime, returns_list in volatility_segments.items():
            if returns_list:
                sharpe = np.mean(returns_list) / np.std(returns_list) if np.std(returns_list) > 0 else 0
                volatility_performance[regime] = {
                    'sharpe_ratio': sharpe,
                    'count': len(returns_list),
                    'avg_return': np.mean(returns_list)
                }
        
        logger.info(f"Volatility segments: {list(volatility_performance.keys())}")
        return {'volatility': volatility_performance} if volatility_performance else {}
    
    def detect_market_regimes(self, returns: pd.Series) -> Dict[str, Dict]:
        """Detect market regimes and calculate performance per regime"""
        if len(returns) < 10:
            logger.warning("Insufficient data for regime detection")
            return {}
        
        logger.info(f"Detecting market regimes for {len(returns)} returns")
        
        # Calculate rolling volatility and trend
        window = min(5, len(returns)//2)
        rolling_vol = returns.rolling(window=window).std()
        rolling_trend = returns.rolling(window=window).mean()
        
        vol_median = rolling_vol.median()
        trend_median = rolling_trend.median()
        
        logger.info(f"Vol median: {vol_median:.6f}, Trend median: {trend_median:.6f}")
        
        # Classify regimes
        regimes = {}
        for i, (vol, trend) in enumerate(zip(rolling_vol, rolling_trend)):
            if pd.notna(vol) and pd.notna(trend):
                vol_regime = 'High' if vol > vol_median else ('Medium' if vol > vol_median * 0.7 else 'Low')
                trend_regime = 'Up' if trend > trend_median else 'Down'
                regime = f"{vol_regime}_Vol_{trend_regime}_Trend"
                
                if regime not in regimes:
                    regimes[regime] = []
                
                if i < len(returns):
                    regimes[regime].append(returns.iloc[i])
        
        # Calculate performance per regime
        regime_performance = {}
        for regime, returns_list in regimes.items():
            if len(returns_list) > 0:
                sharpe = np.mean(returns_list) / np.std(returns_list) if np.std(returns_list) > 0 else 0
                regime_performance[regime] = {
                    'sharpe_ratio': sharpe,
                    'count': len(returns_list),
                    'avg_return': np.mean(returns_list)
                }
                logger.info(f"Regime {regime}: Sharpe={sharpe:.3f}, Count={len(returns_list)}")
        
        logger.info(f"Regime performance calculation complete: {len(regime_performance)} regimes")
        return regime_performance
    
    def find_optimal_conditions(self, segmented_performance: Dict[str, Dict], 
                               regime_performance: Dict[str, Dict]) -> List[Tuple[str, float, str]]:
        """Find optimal trading conditions based on Sharpe ratios"""
        optimal_conditions = []
        
        # Add segmented performance conditions
        for segment_type, segments in segmented_performance.items():
            for segment, metrics in segments.items():
                sharpe = metrics.get('sharpe_ratio', 0)
                if sharpe > 0.5:  # Threshold for "good" performance
                    optimal_conditions.append((str(segment), sharpe, segment_type))
        
        # Add regime performance conditions
        for regime, metrics in regime_performance.items():
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe > 0.5:  # Threshold for "good" performance
                optimal_conditions.append((regime, sharpe, 'regime'))
        
        # Sort by Sharpe ratio (descending)
        optimal_conditions.sort(key=lambda x: x[1], reverse=True)
        
        return optimal_conditions[:10]  # Top 10 conditions
    
    def _execute_trades_for_strategy(self, symbol: str, strategy: str, signals_df: pd.DataFrame, 
                                   df: pd.DataFrame, verbose: bool = False) -> BacktestResult:
        """Execute trades for a specific strategy and return BacktestResult"""
        trades = []
        equity = [self.initial_capital]
        position = 0.0
        position_cost = 0.0  # Track the cost basis of current position
        
        for _, signal in signals_df.iterrows():
            timestamp = signal['timestamp']
            price = signal['price']
            action = signal['action']
            
            if action == 'BUY' and position <= 0:
                # Close short position if any, then go long
                if position < 0:
                    # Close short: profit = (short_price - current_price) * quantity
                    pnl = -position * (position_cost - price)  # position_cost is the short price
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'CLOSE_SHORT',
                        'price': price,
                        'quantity': -position,
                        'pnl': pnl,
                        'capital_used': 0,
                        'capital_gained': -position * price
                    })
                
                # Open long position
                quantity = self.initial_capital * 0.1 / price  # 10% of capital
                position = quantity
                position_cost = price  # Track the buy price
                capital_used = quantity * price
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': price,
                    'quantity': quantity,
                    'pnl': 0,
                    'capital_used': capital_used,
                    'capital_gained': 0
                })
                
                if verbose:
                    logger.info(f"{timestamp} - BUY {quantity:.6f} {symbol} at ${price:.2f}")
                    
            elif action == 'SELL' and position > 0:
                # Close long position only (long-only strategy)
                # Close long: profit = (sell_price - buy_price) * quantity
                pnl = position * (price - position_cost)
                trades.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': price,
                    'quantity': position,
                    'pnl': pnl,
                    'capital_used': 0,
                    'capital_gained': position * price
                })
                
                # Reset position to 0 (no shorting)
                position = 0.0
                position_cost = 0.0
                
                if verbose:
                    logger.info(f"{timestamp} - SELL {position:.6f} {symbol} at ${price:.2f}")
            
            # Update equity
            if trades:
                total_pnl = sum(trade.get('pnl', 0) for trade in trades)
                equity.append(self.initial_capital + total_pnl)
        
        # Calculate metrics
        if len(equity) > 1:
            # Create proper timestamps for equity curve
            timestamps = [signals_df['timestamp'].iloc[0]] + list(signals_df['timestamp'])
            equity_series = pd.Series(equity, index=timestamps[:len(equity)])
            returns = equity_series.pct_change().dropna()
            
            total_return = (equity[-1] - equity[0]) / equity[0]
            annualized_return = (1 + total_return) ** (365 / len(signals_df)) - 1 if len(signals_df) > 0 else 0
            
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = drawdown.min()
            
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            # Calculate profit factor
            buy_trades = [t for t in trades if t.get('capital_used', 0) > 0]
            sell_trades = [t for t in trades if t.get('capital_gained', 0) > 0]
            
            if buy_trades and sell_trades:
                total_buy_value = sum(t['capital_used'] for t in buy_trades)
                total_sell_value = sum(t['capital_gained'] for t in sell_trades)
                profit_factor = total_sell_value / total_buy_value if total_buy_value > 0 else 0
            else:
                profit_factor = 0
        else:
            equity_series = pd.Series([self.initial_capital])
            returns = pd.Series([])
            total_return = 0
            annualized_return = 0
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0
            profit_factor = 0
        
        # Calculate enhanced metrics
        rolling_sharpe = self.calculate_rolling_metrics(returns)
        segmented_performance = self.calculate_segmented_performance(trades, returns)
        regime_performance = self.detect_market_regimes(returns)
        optimal_conditions = self.find_optimal_conditions(segmented_performance, regime_performance)
        
        return BacktestResult(
            symbol=symbol,
            strategy=strategy,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            profit_factor=profit_factor,
            equity_curve=equity_series,
            trades=trades,
            start_date=signals_df['timestamp'].iloc[0],
            end_date=signals_df['timestamp'].iloc[-1],
            initial_capital=self.initial_capital,
            final_capital=equity[-1] if equity else self.initial_capital,
            rolling_sharpe=rolling_sharpe,
            segmented_performance=segmented_performance,
            regime_performance=regime_performance,
            optimal_conditions=optimal_conditions
        )
    
    def backtest_symbol(self, symbol: str, days: int = 30, use_sentiment: bool = False,
                       verbose: bool = False, strategies: List[str] = None) -> Dict[str, BacktestResult]:
        """Backtest a single symbol with multiple strategies"""
        logger.info(f"Backtesting {symbol} with strategies: {strategies}")
        
        # Load data
        df = self.load_symbol_data(symbol, days)
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        results = {}
        
        # If no strategies specified, use all available
        if not strategies:
            strategies = ['btc_ny_session', 'liquidity_sweep_reversal', 'volume_weighted_trend_continuation', 
                         'volatility_expansion_breakout', 'daily_avwap_zscore_reversion', 
                         'opening_range_break_retest', 'keltner_exhaustion_fade', 'fakeout_reversion']
        
        # Backtest each strategy separately
        for strategy in strategies:
            try:
                logger.info(f"Backtesting {symbol} with strategy: {strategy}")
                
                # Generate signals for this specific strategy
                signals_df = self.generate_signals_with_framework(symbol, days=days, strategies=[strategy])
                
                if signals_df is None or (hasattr(signals_df, 'empty') and signals_df.empty) or len(signals_df) == 0:
                    logger.warning(f"No signals generated for {symbol} with strategy {strategy}")
                    continue
                
                # Execute trades for this strategy
                result = self._execute_trades_for_strategy(symbol, strategy, signals_df, df, verbose)
                results[f"{symbol}_{strategy}"] = result
                
            except Exception as e:
                logger.error(f"Error backtesting {symbol} with strategy {strategy}: {e}")
                continue
        
        return results
    
    def backtest_multiple_symbols(self, symbols: List[str], days: int = 30, 
                                 use_sentiment: bool = False, verbose: bool = False,
                                 strategies: List[str] = None) -> Dict[str, BacktestResult]:
        """Backtest multiple symbols with multiple strategies"""
        logger.info(f"Backtesting symbols: {symbols}")
        
        results = {}
        total_symbols = len(symbols)
        
        # Create progress bar for backtesting
        pbar = tqdm(total=total_symbols, desc="Backtesting symbols")
        
        for i, symbol in enumerate(symbols):
            if symbol.upper() not in self.available_symbols:
                logger.warning(f"Symbol {symbol} not available. Available: {self.available_symbols}")
                pbar.update(1)
                continue
            
            # Update progress bar description
            pbar.set_description(f"Backtesting {symbol}")
            
            try:
                symbol_results = self.backtest_symbol(symbol, days, use_sentiment, verbose, strategies)
                # symbol_results is now a dict of {strategy_name: BacktestResult}
                results.update(symbol_results)  # Add all strategy results to main results
                
                # Show summary for this symbol
                if symbol_results:
                    best_result = max(symbol_results.values(), key=lambda r: r.total_return)
                    pbar.set_postfix({
                        "Return": f"{best_result.total_return:.2%}",
                        "Trades": best_result.total_trades,
                        "Sharpe": f"{best_result.sharpe_ratio:.2f}"
                    })
                    logger.info(f"Completed backtest for {symbol}: {best_result.total_return:.2%} return")
                else:
                    pbar.set_postfix({"Return": "0.00%", "Trades": 0, "Sharpe": "0.00"})
                    logger.info(f"No results for {symbol}")
            except Exception as e:
                logger.error(f"Exception in backtest_symbol for {symbol}: {e}")
                # Create a dummy result to continue
                dummy_result = BacktestResult(
                    symbol=symbol,
                    strategy="error",
                    total_return=0.0,
                    annualized_return=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    total_trades=0,
                    profit_factor=0.0,
                    equity_curve=pd.Series([self.initial_capital]),
                    trades=[],
                    start_date=datetime.now(),
                    end_date=datetime.now(),
                    initial_capital=self.initial_capital,
                    final_capital=self.initial_capital,
                    rolling_sharpe=None,
                    segmented_performance=None,
                    regime_performance=None,
                    optimal_conditions=None
                )
                results[f"{symbol}_error"] = dummy_result
                pbar.set_postfix({"Error": str(e)[:20]})
            
            pbar.update(1)
        
        pbar.close()
        logger.info(f"Backtest completed for {len(results)} symbols")
        return results
    
    def create_comparative_report(self, results: Dict[str, BacktestResult], 
                                portfolio_result: Optional[PortfolioResult] = None,
                                output_dir: str = "backtests/results") -> str:
        """Create comprehensive comparative report using professional architecture"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert BacktestResult objects to EquityCurve objects
        curves = []
        for key, result in results.items():
            # Extract symbol and strategy from key
            if '_' in key:
                symbol = key.split('_')[0]
                strategy = '_'.join(key.split('_')[1:])
            else:
                symbol = key
                strategy = getattr(result, 'strategy', 'unknown')
            
            curve = EquityCurveFactory.from_backtest_result(result, symbol, strategy)
            curves.append(curve)
        
        # Add portfolio curve if available
        if portfolio_result:
            portfolio_curve = EquityCurve(
                id="portfolio_equal_weight",
                symbol="PORTFOLIO",
                strategy="equal_weight",
                timestamps=portfolio_result.portfolio_equity_curve.index.values,
                equity=portfolio_result.portfolio_equity_curve.values,
                returns=np.diff(portfolio_result.portfolio_equity_curve.values) / portfolio_result.portfolio_equity_curve.values[:-1],
                trades=[],
                meta={'portfolio': True, 'initial_capital': self.initial_capital}
            )
            curves.append(portfolio_curve)
        
        # Create professional report
        config = ReportConfig(
            equity=True,
            rolling_sharpe=True,
            drawdown=True,
            segmented_performance=True,
            regime_performance=True,
            rolling_window_percentages=[0.01, 0.03, 0.07]
        )
        
        report_generator = ReportGenerator(curves, config)
        report_path = report_generator.generate_report(f"{output_path}/professional_report_{timestamp}.png")
        
        # Also create a summary report
        summary = report_generator.generate_summary_report()
        
        # Save summary as JSON
        summary_path = f"{output_path}/summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Professional report saved to: {report_path}")
        logger.info(f"Summary saved to: {summary_path}")
        
        return report_path

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Multi-Symbol Crypto Backtester')
    parser.add_argument('--symbols', nargs='+', required=True, 
                       help='Symbols to backtest (e.g., BTC ETH ADA)')
    parser.add_argument('--strategies', nargs='*', default=None,
                       help='Strategies to use (default: all available)')
    parser.add_argument('--days', type=int, default=30, 
                       help='Number of days to backtest (default: 30)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital (default: 100000)')
    parser.add_argument('--sentiment', action='store_true', 
                       help='Use sentiment-enhanced strategies')
    parser.add_argument('--verbose', action='store_true',
                       help='Show verbose output with all trades')
    parser.add_argument('--portfolio', action='store_true', 
                       help='Run portfolio backtest with rebalancing')
    parser.add_argument('--output-dir', default='backtests/results', 
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    try:
        # Initialize backtester
        backtester = MultiSymbolBacktester(initial_capital=args.capital)
        
        # Run individual symbol backtests
        results = backtester.backtest_multiple_symbols(
            symbols=args.symbols,
            days=args.days, 
            use_sentiment=args.sentiment,
            verbose=args.verbose,
            strategies=args.strategies
        )
        
        # Run portfolio backtest if requested
        portfolio_result = None
        if args.portfolio and len(args.symbols) > 1:
            logger.info("Running portfolio backtest...")
            portfolio_result = backtester.backtest_portfolio(
                symbols=args.symbols,
                days=args.days,
                use_sentiment=args.sentiment,
                strategies=args.strategies
            )
        
        # Generate report
        report_path = backtester.create_comparative_report(results, portfolio_result, args.output_dir)
        
        print(f"\nBacktest completed! Report saved to: {report_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        
        # Group results by symbol for summary
        symbol_summaries = {}
        for key, result in results.items():
            if '_' in key:
                symbol = key.split('_')[0]
                strategy = '_'.join(key.split('_')[1:])
            else:
                symbol = key
                strategy = "unknown"
            
            if symbol not in symbol_summaries:
                symbol_summaries[symbol] = []
            symbol_summaries[symbol].append((strategy, result))
        
        for symbol, strategy_results in symbol_summaries.items():
            print(f"\n{symbol.upper()}:")
            for strategy, result in strategy_results:
                print(f"  {strategy}: Return={result.total_return:.2%}, Sharpe={result.sharpe_ratio:.2f}, DD={result.max_drawdown:.2%}, Trades={result.total_trades}")
        
        if portfolio_result:
            print(f"\nPORTFOLIO:")
            print(f"  Return: {portfolio_result.total_return:.2%}")
            print(f"  Sharpe: {portfolio_result.sharpe_ratio:.2f}")
            print(f"  Max DD: {portfolio_result.max_drawdown:.2%}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

if __name__ == "__main__":
    main()