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
from datetime import datetime, timedelta

import crypto_signal_framework
import crypto_trading_strategies
import crypto_analysis_engine
import crypto_signal_generator

CryptoSignalFramework = crypto_signal_framework.CryptoSignalFramework
StrategyConfig = crypto_signal_framework.StrategyConfig
CryptoTradingStrategies = crypto_trading_strategies.CryptoTradingStrategies
CryptoAnalysisEngine = crypto_analysis_engine.CryptoAnalysisEngine
CryptoSentimentGenerator = crypto_signal_generator.CryptoSentimentGenerator

logger = logging.getLogger(__name__)

class CryptoSignalIntegration:
    """
    Main integration class that connects signal framework with crypto pipeline
    """
    
    def __init__(self, data_dir: str = "data", selected_strategies: List[str] = None):
        self.data_dir = Path(data_dir)
        self.framework = CryptoSignalFramework()
        self.strategies = CryptoTradingStrategies()
        self.analysis_engine = CryptoAnalysisEngine()
        self.sentiment_generator = CryptoSentimentGenerator()
        
        # Initialize strategies
        self._setup_strategies(selected_strategies)
        
        # Performance tracking
        self.performance_history = []
        self.signal_log = []
        
    def _setup_strategies(self, selected_strategies: List[str] = None):
        """Setup trading strategies in the framework"""
        strategy_configs = self.strategies.get_strategy_configs()
        
        for config in strategy_configs:
            # Only add selected strategies if specified
            if selected_strategies and config.name not in selected_strategies:
                continue
                
            strategy_function = self.strategies.strategies[config.name]['function']
            self.framework.add_strategy(config, strategy_function)
            logger.info(f"Added strategy: {config.name}")
    
    def load_crypto_data(self, symbols: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
        """Load crypto data for signal generation"""
        data = {}
        
        for symbol in symbols:
            try:
                # Load data using existing analysis engine
                df = self.analysis_engine.load_symbol_data(symbol, days=days)
                if df is not None and len(df) > 50:
                    data[symbol] = df
                    logger.info(f"Loaded {len(df)} data points for {symbol}")
                else:
                    logger.warning(f"Insufficient data for {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        return data
    
    def generate_signals(self, symbols: List[str], days: int = 30, strategies: List[str] = None) -> List[Dict]:
        """Generate trading signals for specified symbols"""
        # Load data
        data = self.load_crypto_data(symbols, days)
        
        if not data:
            logger.error("No data available for signal generation")
            return []
        
        # Generate signals for each symbol's historical data
        all_signals = []
        
        for symbol, symbol_data in data.items():
            logger.info(f"Generating signals for {symbol} with {len(symbol_data)} data points")
            
            # Filter strategies if specified
            if strategies:
                # Only process selected strategies for this symbol
                symbol_strategies = [s for s in strategies if s.startswith(symbol.lower())]
                if not symbol_strategies:
                    continue
            
            # Generate signals for each historical timestamp
            for i in range(50, len(symbol_data)):  # Check every data point
                current_data = symbol_data.iloc[:i+1].copy()
                
                # Debug: Check if this is a NY session time
                current_time = current_data.index[-1]
                if hasattr(current_time, 'tz'):
                    if current_time.tz is None:
                        ny_time = current_time.tz_localize('UTC').tz_convert('America/New_York')
                    else:
                        ny_time = current_time.tz_convert('America/New_York')
                else:
                    ny_time = current_time
                
                hour = ny_time.hour
                minute = ny_time.minute
                current_minutes = hour * 60 + minute
                ny_open_time = 9 * 60 + 30  # 9:30 AM
                ny_close_time = 16 * 60    # 4:00 PM
                
                
                
                # Generate signals for this historical point
                signals = self.framework.generate_signals({symbol: current_data})
                
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
                            'entry_price': signal.entry_price,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'reason': signal.reason,
                            'timestamp': current_data.index[-1].isoformat() if hasattr(current_data.index[-1], 'isoformat') else str(current_data.index[-1]),
                            'risk_size': signal.risk_size
                        }
                        all_signals.append(signal_dict)
        
        logger.info(f"Generated {len(all_signals)} total signals")
        
        # Log signals
        self.signal_log.extend(all_signals)
        
        return all_signals
    
    def backtest_signals(self, symbols: List[str], days: int = 90, initial_capital: float = 100000, step: int = 10) -> Dict:
        """Backtest the signal framework"""
        logger.info(f"Starting backtest for {symbols} over {days} days")
        
        # Load data
        data = self.load_crypto_data(symbols, days)
        
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
        signals = self.generate_signals(symbols, days=7)  # Use last 7 days for context
        
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
