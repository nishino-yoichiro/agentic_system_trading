"""
Crypto Signal Integration
========================

Integrates the crypto trading signal framework with the existing crypto pipeline.
Provides real-time signal generation using actual crypto data.

Author: Quantitative Strategy Designer
Date: 2025-09-28
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta, timezone
import sys

# Add src to path for progress logger
sys.path.append(str(Path(__file__).parent))
from utils.progress_logger import progress_logger, create_signal_progress

try:
    from . import crypto_signal_framework
    # Removed crypto_trading_strategies - using new dynamic system
    from . import crypto_analysis_engine
    from . import crypto_signal_generator
    from . import btc_ny_session_strategy
    from . import professional_crypto_strategies
except ImportError:
    # Fallback for when running as script
    import crypto_signal_framework
    # Removed crypto_trading_strategies - using new dynamic system
    import crypto_analysis_engine
    import crypto_signal_generator
    import btc_ny_session_strategy
    import professional_crypto_strategies

# Import new professional framework classes
SignalFramework = crypto_signal_framework.SignalFramework
StrategyConfig = crypto_signal_framework.StrategyConfig
StrategyMetadata = crypto_signal_framework.StrategyMetadata
StrategyType = crypto_signal_framework.StrategyType
BaseStrategy = crypto_signal_framework.BaseStrategy

# Legacy imports for compatibility
CryptoSignalFramework = crypto_signal_framework.CryptoSignalFramework
# Removed CryptoTradingStrategies - using new dynamic system
CryptoAnalysisEngine = crypto_analysis_engine.CryptoAnalysisEngine
CryptoSentimentGenerator = crypto_signal_generator.CryptoSentimentGenerator

logger = logging.getLogger(__name__)

class CryptoSignalIntegration:
    """
    Main integration class that connects signal framework with crypto pipeline
    """
    
    def __init__(self, data_dir: str = "data", selected_strategies: List[str] = None):
        self.data_dir = Path(data_dir)
        # Use new professional framework
        self.framework = SignalFramework(max_lookback=100)
        # Removed old strategies - using new dynamic framework
        self.analysis_engine = CryptoAnalysisEngine()
        self.sentiment_generator = CryptoSentimentGenerator()
        
        # Initialize strategies using new professional approach
        self._setup_strategies(selected_strategies)
        
        # Performance tracking
        self.performance_history = []
        self.signal_log = []
        
    def _setup_strategies(self, selected_strategies: List[str] = None):
        """Setup trading strategies using new professional framework"""
        
        # Add BTC NY Session strategy using new professional approach
        if not selected_strategies or 'btc_ny_session' in selected_strategies:
            btc_strategy = btc_ny_session_strategy.BTCNYSessionStrategy()
            self.framework.add_strategy(btc_strategy)
            logger.info(f"Added professional strategy: {btc_strategy.name}")
        
        # Add other professional strategies
        professional_strategies = [
            'liquidity_sweep_reversal',
            'volume_weighted_trend_continuation', 
            'volatility_expansion_breakout',
            'daily_avwap_zscore_reversion',
            'opening_range_break_retest',
            'keltner_exhaustion_fade',
            'fakeout_reversion'
        ]
        
        # Add test strategy for manual testing
        try:
            from test_every_minute_strategy import TestEveryMinuteStrategy
            test_strategy = TestEveryMinuteStrategy()
            self.framework.add_strategy(test_strategy)
            logger.info(f"Added test strategy: {test_strategy.name}")
        except Exception as e:
            logger.warning(f"Could not add test strategy: {e}")
        
        for strategy_name in professional_strategies:
            # Only add selected strategies if specified
            if selected_strategies and strategy_name not in selected_strategies:
                continue
            
            try:
                strategy = professional_crypto_strategies.create_strategy(strategy_name)
                if strategy:
                    self.framework.add_strategy(strategy)
                    logger.info(f"Added professional strategy: {strategy.name}")
                else:
                    logger.warning(f"Failed to create strategy: {strategy_name}")
            except Exception as e:
                logger.error(f"Error adding strategy {strategy_name}: {e}")
    
    def load_crypto_data(self, symbols: List[str], days: int = 1095, include_live: bool = True) -> Dict[str, pd.DataFrame]:
        """Load crypto data for signal generation (default: 3 years).
        
        Args:
            symbols: List of symbols to load
            days: Number of days of historical data to load
            include_live: Whether to include live data (False for backtests)
        """
        data = {}
        
        for symbol in symbols:
            try:
                # Load historical DB
                hist_df = self.analysis_engine.load_symbol_data(symbol, days=days)
                if hist_df is None or len(hist_df) <= 50:
                    logger.warning(f"Insufficient historical data for {symbol}")
                    continue
                # Try to load live 1m file for latest bars (only if include_live=True)
                if include_live:
                    live_path = self.data_dir / f"{symbol}_1m_historical.parquet"
                    if live_path.exists():
                        try:
                            live_df = pd.read_parquet(live_path)
                            live_df.index = pd.to_datetime(live_df.index)
                            # Ensure UTC tz
                            if live_df.index.tz is None:
                                live_df.index = live_df.index.tz_localize('UTC')
                            # Keep only data newer than the last historical timestamp to avoid large duplicate overlap
                            last_hist = pd.to_datetime(hist_df.index).max()
                            if last_hist.tz is None:
                                last_hist = last_hist.tz_localize('UTC')
                            live_df = live_df[live_df.index > last_hist]
                            if not live_df.empty:
                                merged = pd.concat([hist_df, live_df])
                                merged = merged[~merged.index.duplicated(keep='last')]
                                merged = merged.sort_index()
                                data[symbol] = merged
                                logger.info(f"Loaded {len(hist_df)} hist + {len(live_df)} live = {len(merged)} for {symbol}")
                            else:
                                data[symbol] = hist_df
                                logger.info(f"Loaded {len(hist_df)} data points for {symbol} (no newer live minutes)")
                        except Exception as e:
                            logger.warning(f"Could not merge live minutes for {symbol}: {e}")
                            data[symbol] = hist_df
                    else:
                        data[symbol] = hist_df
                        logger.info(f"Loaded {len(hist_df)} data points for {symbol} (no live data file)")
                else:
                    # Backtest mode - only use historical data
                    data[symbol] = hist_df
                    logger.info(f"Loaded {len(hist_df)} data points for {symbol} (backtest mode - no live data)")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        return data
    
    def generate_signals(self, symbols: List[str], days: int = 1095, strategies: List[str] = None) -> List[Dict]:
        """Generate trading signals for specified symbols"""
        # Determine optimal data window based on strategy requirements
        if days is None:
            # Use strategy metadata to determine optimal lookback
            max_lookback = 0
            for strategy_name, strategy_info in self.framework.strategies.items():
                if strategies and strategy_name not in strategies:
                    continue
                strategy_lookback = strategy_info['config'].lookback
                max_lookback = max(max_lookback, strategy_lookback)
            
            # Add some buffer for technical indicators and ensure minimum data
            optimal_days = max(max_lookback + 50, 30)  # At least 30 days, plus strategy lookback + buffer
            logger.info(f"Using strategy-optimized lookback: {optimal_days} days (max strategy lookback: {max_lookback})")
        else:
            optimal_days = days
            logger.info(f"Using specified lookback: {optimal_days} days")
        
        # Load data (backtest mode - no live data)
        data = self.load_crypto_data(symbols, optimal_days, include_live=False)
        
        if not data:
            logger.error("No data available for signal generation")
            return []
        
        # Generate signals for each symbol's historical data
        all_signals = []
        
        for symbol, symbol_data in data.items():
            logger.info(f"Generating signals for {symbol} with {len(symbol_data)} data points")
            
            # Filter strategies if specified
            if strategies:
                # Check if any of the selected strategies are configured for this symbol
                symbol_strategies = [s for s in strategies if s in self.framework.strategies]
                if not symbol_strategies:
                    continue
            
            # Generate signals for the entire dataset at once (vectorized)
            logger.info(f"Processing {symbol} with {len(symbol_data)} data points")
            
            # Create progress bar for signal generation
            strategy_names = [s for s in strategies] if strategies else list(self.framework.strategies.keys())
            pbar = create_signal_progress(len(symbol_data), symbol, f"{len(strategy_names)} strategies")
            
            # Generate signals for the entire dataset
            try:
                signals = self.framework.generate_signals({symbol: symbol_data}, progress_bar=pbar, strategies=strategies)
            except Exception as e:
                logger.warning(f"Error generating signals for {symbol}: {e}")
                pbar.close()
                continue
            
            pbar.close()
            
            # Process the generated signals
            for signal in signals:
                if signal and signal.signal_type.name != 'FLAT':
                    # Filter by selected strategies if specified
                    if strategies and signal.strategy_name not in strategies:
                        continue
                    
                    signal_dict = {
                        'symbol': symbol,
                        'strategy': signal.strategy_name,
                        'signal_type': signal.signal_type.name,
                        'confidence': signal.confidence,
                        'price': signal.entry_price,  # Map entry_price to price for backtester compatibility
                        'action': 'BUY' if signal.signal_type.name == 'LONG' else 'SELL' if signal.signal_type.name == 'SHORT' else 'HOLD',
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'reason': signal.reason,
                        'timestamp': signal.timestamp.isoformat() if hasattr(signal.timestamp, 'isoformat') else str(signal.timestamp),
                        'risk_size': signal.risk_size
                    }
                    all_signals.append(signal_dict)
        
        logger.info(f"Generated {len(all_signals)} total signals")
        
        # Log signals
        self.signal_log.extend(all_signals)
        
        return all_signals
    
    def generate_live_signals(self, symbols: List[str], strategies: List[str] = None) -> List[Dict]:
        """Generate signals for live trading - only processes the latest data point"""
        # Load data with strategy-optimized lookback
        max_lookback = 0
        for strategy_name, strategy_info in self.framework.strategies.items():
            if strategies and strategy_name not in strategies:
                continue
            if isinstance(strategy_info, dict) and 'config' in strategy_info:
                strategy_lookback = strategy_info['config'].lookback
            else:
                # Handle case where strategy_info is the strategy object itself
                metadata = getattr(strategy_info, 'metadata', None)
                strategy_lookback = metadata.lookback if metadata else 0
            max_lookback = max(max_lookback, strategy_lookback)
        
        optimal_days = max(max_lookback + 50, 30)
        data = self.load_crypto_data(symbols, optimal_days)
        
        if not data:
            logger.error("No data available for live signal generation")
            return []
        
        all_signals = []
        
        for symbol, symbol_data in data.items():
            logger.info(f"Generating live signals for {symbol} with {len(symbol_data)} data points")
            
            # Only process the latest data point for live trading
            latest_row = symbol_data.iloc[-1]
            latest_timestamp = symbol_data.index[-1]
            
            # Generate signals for each strategy
            for strategy_name, strategy_info in self.framework.strategies.items():
                if strategies and strategy_name not in strategies:
                    continue
                
                try:
                    if isinstance(strategy_info, dict) and 'strategy' in strategy_info:
                        strategy = strategy_info['strategy']
                        metadata = strategy_info['config']
                    else:
                        # Handle case where strategy_info is the strategy object itself
                        strategy = strategy_info
                        metadata = getattr(strategy, 'metadata', None)
                    
                    # For CONSTANT_TIME strategies, check if current time is relevant
                    if metadata.strategy_type.name == 'CONSTANT_TIME':
                        if strategy_name == 'btc_ny_session':
                            # Check if current time is in NY session signal window
                            if not self._is_in_ny_signal_window(latest_timestamp):
                                continue
                    
                    # Generate signal for latest data point
                    if metadata.strategy_type.name == 'CONSTANT_TIME':
                        signal = strategy.generate_signal(latest_row, None)
                    else:
                        # For other strategies, provide recent history
                        history = symbol_data.tail(metadata.lookback + 10)
                        signal = strategy.generate_signal(latest_row, history)
                    
                    if signal and signal.confidence >= metadata.min_confidence:
                        signal_dict = {
                            'symbol': symbol,
                            'strategy': strategy_name,
                            'signal_type': signal.signal_type.name,
                            'confidence': signal.confidence,
                            'price': signal.entry_price,
                            'action': 'BUY' if signal.signal_type.name == 'LONG' else 'SELL' if signal.signal_type.name == 'SHORT' else 'HOLD',
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'reason': signal.reason,
                            'timestamp': latest_timestamp.isoformat() if hasattr(latest_timestamp, 'isoformat') else str(latest_timestamp),
                            'risk_size': signal.risk_size
                        }
                        all_signals.append(signal_dict)
                        logger.info(f"Generated live signal: {strategy_name} {signal.signal_type.name} @ ${signal.entry_price:.2f}")
                        
                except Exception as e:
                    logger.error(f"Error generating live signal for {strategy_name}: {e}")
                    continue
        
        logger.info(f"Generated {len(all_signals)} live signals")
        return all_signals
    
    def _is_in_ny_signal_window(self, timestamp) -> bool:
        """Check if timestamp is in NY session signal generation window"""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        elif timestamp.tzinfo != timezone.utc:
            timestamp = timestamp.astimezone(timezone.utc)
        
        # NY open: 14:30-14:35 UTC, NY close: 21:00-21:05 UTC
        if (timestamp.hour == 14 and 30 <= timestamp.minute <= 35) or \
           (timestamp.hour == 21 and 0 <= timestamp.minute <= 5):
            return True
        return False
    
    def backtest_signals(self, symbols: List[str], days: int = 90, initial_capital: float = 100000, step: int = 10) -> Dict:
        """Backtest the signal framework"""
        logger.info(f"Starting backtest for {symbols} over {days} days")
        
        # Load data (backtest mode - no live data)
        data = self.load_crypto_data(symbols, days, include_live=False)
        
        if not data:
            return {'error': 'No data available for backtesting'}
        
        # Initialize backtest
        capital = initial_capital
        positions = {}
        trades = []
        equity_curve = [capital]
        timestamps = []
        
        # Get all unique timestamps
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index)
        all_timestamps = sorted(list(all_timestamps))
        
        # Run backtest with stride to speed up (evaluate every 'step' bars)
        if step <= 0:
            step = 1
        for i in range(0, len(all_timestamps), step):
            timestamp = all_timestamps[i]
            # Get data up to current timestamp
            current_data = {}
            for symbol, df in data.items():
                current_df = df[df.index <= timestamp]
                if len(current_df) > 50:  # Need minimum data
                    current_data[symbol] = current_df
            
            if not current_data:
                continue
            
            # Generate signals
            signals = self.framework.generate_signals(current_data)
            
            # Process signals
            for signal in signals:
                symbol = signal.strategy_name.split('_')[0] if '_' in signal.strategy_name else 'BTC'
                
                # Calculate position size
                position_size = capital * signal.risk_size
                
                if signal.signal_type.name == 'LONG':
                    if symbol not in positions or positions[symbol] <= 0:
                        # Open long position
                        shares = position_size / signal.entry_price
                        positions[symbol] = shares
                        
                        trades.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'strategy': signal.strategy_name,
                            'action': 'BUY',
                            'price': signal.entry_price,
                            'shares': shares,
                            'value': position_size,
                            'reason': signal.reason
                        })
                
                elif signal.signal_type.name == 'SHORT':
                    if symbol not in positions or positions[symbol] >= 0:
                        # Open short position (simplified - close long first)
                        if symbol in positions and positions[symbol] > 0:
                            # Close long position
                            close_value = positions[symbol] * signal.entry_price
                            capital += close_value
                            
                            trades.append({
                                'timestamp': timestamp,
                                'symbol': symbol,
                                'strategy': signal.strategy_name,
                                'action': 'SELL',
                                'price': signal.entry_price,
                                'shares': positions[symbol],
                                'value': close_value,
                                'reason': 'Close long for short'
                            })
                        
                        # Open short position
                        shares = -position_size / signal.entry_price
                        positions[symbol] = shares
                        
                        trades.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'strategy': signal.strategy_name,
                            'action': 'SHORT',
                            'price': signal.entry_price,
                            'shares': shares,
                            'value': position_size,
                            'reason': signal.reason
                        })
            
            # Update portfolio value
            portfolio_value = capital
            for symbol, shares in positions.items():
                if symbol in current_data and len(current_data[symbol]) > 0:
                    current_price = current_data[symbol]['close'].iloc[-1]
                    portfolio_value += shares * current_price
            
            equity_curve.append(portfolio_value)
            timestamps.append(timestamp)
        
        # Calculate performance metrics
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            
            total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
            annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            cumulative = pd.Series(equity_curve)
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate win rate
            winning_trades = [t for t in trades if t['action'] in ['SELL', 'SHORT'] and 
                            (t['value'] - trades[trades.index(t)-1]['value'] if trades.index(t) > 0 else 0) > 0]
            win_rate = len(winning_trades) / len([t for t in trades if t['action'] in ['SELL', 'SHORT']]) if trades else 0
            
            results = {
                'initial_capital': initial_capital,
                'final_capital': equity_curve[-1],
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'equity_curve': equity_curve,
                'timestamps': [t.isoformat() for t in timestamps],
                'trades': trades[-50:],  # Last 50 trades
                'strategy_performance': self._calculate_strategy_performance(trades)
            }
        else:
            results = {'error': 'Insufficient data for backtesting'}
        
        return results
    
    def _calculate_strategy_performance(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Calculate performance metrics for each strategy"""
        strategy_trades = {}
        
        # Group trades by strategy
        for trade in trades:
            strategy = trade['strategy']
            if strategy not in strategy_trades:
                strategy_trades[strategy] = []
            strategy_trades[strategy].append(trade)
        
        # Calculate metrics for each strategy
        strategy_performance = {}
        for strategy, strategy_trade_list in strategy_trades.items():
            if len(strategy_trade_list) < 2:
                continue
            
            # Calculate P&L for each trade
            pnl_list = []
            for i in range(1, len(strategy_trade_list)):
                if strategy_trade_list[i]['action'] in ['SELL', 'SHORT']:
                    prev_trade = strategy_trade_list[i-1]
                    curr_trade = strategy_trade_list[i]
                    
                    if prev_trade['action'] == 'BUY' and curr_trade['action'] == 'SELL':
                        pnl = curr_trade['value'] - prev_trade['value']
                        pnl_list.append(pnl)
            
            if pnl_list:
                total_pnl = sum(pnl_list)
                avg_pnl = np.mean(pnl_list)
                win_rate = len([p for p in pnl_list if p > 0]) / len(pnl_list)
                
                strategy_performance[strategy] = {
                    'total_pnl': total_pnl,
                    'avg_pnl': avg_pnl,
                    'win_rate': win_rate,
                    'total_trades': len(pnl_list)
                }
        
        return strategy_performance
    
    def get_live_signals(self, symbols: List[str]) -> Dict:
        """Get current live signals for dashboard"""
        signals = self.generate_signals(symbols, days=30)  # Use last 30 days for context
        
        # Group signals by symbol
        symbol_signals = {}
        for signal in signals:
            symbol = signal['symbol']
            if symbol not in symbol_signals:
                symbol_signals[symbol] = []
            symbol_signals[symbol].append(signal)
        
        # Get current prices
        current_prices = {}
        for symbol in symbols:
            try:
                data = self.analysis_engine.load_symbol_data(symbol, days=1)
                if data is not None and len(data) > 0:
                    current_prices[symbol] = data['close'].iloc[-1]
            except:
                current_prices[symbol] = 0
        
        return {
            'signals': symbol_signals,
            'current_prices': current_prices,
            'timestamp': datetime.now().isoformat(),
            'total_signals': len(signals)
        }
    
    def save_signal_log(self, filename: str = None):
        """Save signal log to file"""
        if filename is None:
            filename = f"signal_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        log_path = self.data_dir / "logs" / filename
        log_path.parent.mkdir(exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump(self.signal_log, f, indent=2)
        
        logger.info(f"Signal log saved to {log_path}")
    
    def get_strategy_summary(self) -> Dict:
        """Get summary of all strategies"""
        summary = {}
        
        for name, strategy in self.framework.strategies.items():
            config = strategy['config']
            summary[name] = {
                'symbol': config.symbol,
                'mechanism': config.mechanism,
                'horizon': config.horizon,
                'session': config.session,
                'sharpe': strategy['sharpe'],
                'max_drawdown': strategy['max_dd'],
                'hit_rate': strategy['hit_rate'],
                'total_trades': len(strategy['returns'])
            }
        
        return summary

def main():
    """Demo the crypto signal integration"""
    logger.info("Starting Crypto Signal Integration Demo")
    
    # Initialize integration
    integration = CryptoSignalIntegration()
    
    # Demo symbols
    symbols = ['BTC', 'ETH']
    
    # Generate live signals
    logger.info("Generating live signals...")
    live_signals = integration.get_live_signals(symbols)
    
    print(f"\n=== LIVE SIGNALS ===")
    print(f"Total signals: {live_signals['total_signals']}")
    print(f"Timestamp: {live_signals['timestamp']}")
    
    for symbol, signals in live_signals['signals'].items():
        print(f"\n{symbol} Signals:")
        for signal in signals:
            print(f"  {signal['strategy']}: {signal['signal_type']} @ {signal['entry_price']:.2f} "
                  f"(Confidence: {signal['confidence']:.2f})")
            print(f"    Reason: {signal['reason']}")
    
    # Run backtest
    logger.info("Running backtest...")
    backtest_results = integration.backtest_signals(symbols, days=30)
    
    if 'error' not in backtest_results:
        print(f"\n=== BACKTEST RESULTS ===")
        print(f"Initial Capital: ${backtest_results['initial_capital']:,.2f}")
        print(f"Final Capital: ${backtest_results['final_capital']:,.2f}")
        print(f"Total Return: {backtest_results['total_return']:.2%}")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        print(f"Win Rate: {backtest_results['win_rate']:.2%}")
        print(f"Total Trades: {backtest_results['total_trades']}")
        
        print(f"\n=== STRATEGY PERFORMANCE ===")
        for strategy, perf in backtest_results['strategy_performance'].items():
            print(f"{strategy}:")
            print(f"  Total P&L: ${perf['total_pnl']:,.2f}")
            print(f"  Win Rate: {perf['win_rate']:.2%}")
            print(f"  Total Trades: {perf['total_trades']}")
    
    # Get strategy summary
    strategy_summary = integration.get_strategy_summary()
    print(f"\n=== STRATEGY SUMMARY ===")
    for name, summary in strategy_summary.items():
        print(f"{name}:")
        print(f"  Symbol: {summary['symbol']}")
        print(f"  Mechanism: {summary['mechanism']}")
        print(f"  Horizon: {summary['horizon']}")
        print(f"  Session: {summary['session']}")
        print(f"  Sharpe: {summary['sharpe']:.2f}")
        print(f"  Max DD: {summary['max_drawdown']:.2%}")
        print(f"  Hit Rate: {summary['hit_rate']:.2%}")

if __name__ == "__main__":
    main()
