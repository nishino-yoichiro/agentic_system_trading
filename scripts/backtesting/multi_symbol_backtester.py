#!/usr/bin/env python3
"""
Multi-Symbol Crypto Backtester
Universal backtesting framework for all crypto assets with portfolio analysis
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
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from crypto_analysis_engine import CryptoAnalysisEngine
from crypto_signal_generator import CryptoSentimentGenerator
from crypto_signal_integration import CryptoSignalIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Container for individual symbol backtest results"""
    symbol: str
    strategy: str  # 'original' or 'sentiment_enhanced'
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

@dataclass
class PortfolioResult:
    """Container for portfolio-level backtest results"""
    symbols: List[str]
    strategy: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_matrix: pd.DataFrame
    individual_results: Dict[str, BacktestResult]
    portfolio_equity_curve: pd.Series
    rebalance_dates: List[datetime]

class MultiSymbolBacktester:
    """Universal backtester for multiple crypto symbols with portfolio analysis"""
    
    def __init__(self, initial_capital: float = 10000.0, alpha: float = 0.5, 
                 rebalance_frequency: str = 'daily'):
        self.initial_capital = initial_capital
        self.alpha = alpha
        self.rebalance_frequency = rebalance_frequency
        self.analysis_engine = CryptoAnalysisEngine()
        self.sentiment_generator = CryptoSentimentGenerator(alpha=alpha)
        self.signal_integration = None  # Will be initialized with selected strategies
        
        # Available symbols from data directory
        self.available_symbols = self._get_available_symbols()
        
    def _get_available_symbols(self) -> List[str]:
        """Get list of available symbols from data directory"""
        data_dir = Path("data/crypto_db")
        symbols = []
        for file in data_dir.glob("*_historical.parquet"):
            symbol = file.stem.replace("_historical", "").upper()
            symbols.append(symbol)
        return sorted(symbols)
    
    def load_symbol_data(self, symbol: str, days: int = 3) -> pd.DataFrame:
        """Load historical data for a specific symbol"""
        try:
            df = self.analysis_engine.load_symbol_data(symbol, days=days)
            logger.info(f"Loaded {len(df)} data points for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_signals_for_symbol(self, df: pd.DataFrame, symbol: str, 
                                  lookback_hours: int = 24, use_sentiment: bool = False) -> pd.DataFrame:
        """Generate trading signals for a specific symbol"""
        signals = []
        
        for i in range(lookback_hours, len(df)):
            # Get data up to current point (no lookahead bias)
            current_data = df.iloc[:i+1].copy()
            
            try:
                if use_sentiment:
                    # Use sentiment-enhanced generator
                    signal_data = self.sentiment_generator.generate_enhanced_signals(
                        current_data, symbol=symbol
                    )
                else:
                    # Use base analysis engine
                    signal_data = self.analysis_engine.generate_signals(current_data, symbol)
                
                signals.append({
                    'timestamp': current_data.index[-1],
                    'price': current_data['close'].iloc[-1],
                    'action': signal_data.get('enhanced_signal_type', signal_data.get('signal_type', 'HOLD')),
                    'confidence': signal_data.get('enhanced_confidence', signal_data.get('confidence', 0)),
                    'signal_score': signal_data.get('enhanced_signal_strength', signal_data.get('signal_strength', 0)),
                    'sentiment_score': signal_data.get('sentiment_score', 0) if use_sentiment else 0
                })
                
            except Exception as e:
                logger.warning(f"Error generating signal for {symbol} at {current_data.index[-1]}: {e}")
                signals.append({
                    'timestamp': current_data.index[-1],
                    'price': current_data['close'].iloc[-1],
                    'action': 'HOLD',
                    'confidence': 0,
                    'signal_score': 0,
                    'sentiment_score': 0
                })
        
        return pd.DataFrame(signals)
    
    def generate_signals_with_framework(self, symbol: str, days: int = 7, strategies: List[str] = None) -> pd.DataFrame:
        """Generate signals using the new crypto signal framework with historical data"""
        try:
            # Initialize signal integration with selected strategies
            if self.signal_integration is None or strategies:
                self.signal_integration = CryptoSignalIntegration(selected_strategies=strategies)
                logger.info(f"Initialized signal integration with strategies: {strategies}")
            
            # Generate signals once for the entire period
            signal_data = self.signal_integration.generate_signals([symbol], days=days, strategies=strategies)
            
            signals = []
            for signal_dict in signal_data:
                if signal_dict['symbol'] == symbol and signal_dict['signal_type'] != 'HOLD':
                    # Convert signal types to backtester format
                    action = 'BUY' if signal_dict['signal_type'] == 'LONG' else 'SELL' if signal_dict['signal_type'] == 'SHORT' else 'HOLD'
                    
                    signals.append({
                        'timestamp': pd.Timestamp(signal_dict['timestamp']),
                        'price': signal_dict['entry_price'],
                        'action': action,
                        'confidence': signal_dict['confidence'],
                        'signal_score': signal_dict['confidence'],
                        'sentiment_score': 0,
                        'reason': signal_dict['reason']
                    })
            
            logger.info(f"Generated {len(signals)} signals for {symbol} using new framework")
            return pd.DataFrame(signals)
            
        except Exception as e:
            logger.error(f"Error generating signals with framework: {e}")
            print(f"DEBUG: Exception in generate_signals_with_framework: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def backtest_symbol(self, symbol: str, days: int = 3, use_sentiment: bool = False,
                       verbose: bool = False, strategies: List[str] = None) -> BacktestResult:
        """Backtest a single symbol"""
        logger.info(f"Backtesting {symbol} ({'sentiment-enhanced' if use_sentiment else 'original'} strategy)")
        
        # Load data
        df = self.load_symbol_data(symbol, days)
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Generate signals using new framework
        signals_df = self.generate_signals_with_framework(symbol, days=days, strategies=strategies)
        
        # If no signals from new framework, fall back to old method
        if signals_df.empty:
            logger.info(f"No signals from new framework, using fallback for {symbol}")
            signals_df = self.generate_signals_for_symbol(df, symbol, use_sentiment=use_sentiment)
        
        # Execute trades
        trades = []
        capital = self.initial_capital
        position = 0  # Net position (positive = long, negative = short)
        equity_curve = []
        
        for _, signal in signals_df.iterrows():
            current_price = signal['price']
            action = signal['action']
            confidence = signal['confidence']
            
            # Execute trade
            if action == 'BUY':
                # Buy with available capital
                shares = capital / current_price
                position += shares
                capital = 0
                
                trade = {
                    'timestamp': signal['timestamp'],
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'capital_used': shares * current_price,
                    'confidence': confidence,
                    'signal_score': signal['signal_score'],
                    'sentiment_score': signal['sentiment_score']
                }
                trades.append(trade)
                
                if verbose:
                    timestamp_str = signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    print(f"BUY {shares:.6f} {symbol} at ${current_price:.2f} (Net position: {position:.6f}) - {timestamp_str}")
                    
            elif action == 'SELL':
                # Sell all available capital (short if needed)
                shares = capital / current_price
                position -= shares
                capital = 0
                
                trade = {
                    'timestamp': signal['timestamp'],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'capital_gained': shares * current_price,
                    'confidence': confidence,
                    'signal_score': signal['signal_score'],
                    'sentiment_score': signal['sentiment_score']
                }
                trades.append(trade)
                
                if verbose:
                    timestamp_str = signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    print(f"SELL {shares:.6f} {symbol} at ${current_price:.2f} (Net position: {position:.6f}) - {timestamp_str}")
            
            # Update equity curve
            current_equity = capital + (position * current_price)
            equity_curve.append({'timestamp': signal['timestamp'], 'equity': current_equity})
        
        # Final liquidation at end of backtest
        final_price = df['close'].iloc[-1]
        if position != 0:
            # Liquidate all remaining position
            capital = position * final_price
            if verbose:
                final_timestamp = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                print(f"Final liquidation: {position:.6f} {symbol} at ${final_price:.2f} = ${capital:.2f} - {final_timestamp}")
            position = 0
        
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        days_traded = (signals_df['timestamp'].iloc[-1] - signals_df['timestamp'].iloc[0]).days
        annualized_return = (1 + total_return) ** (365 / days_traded) - 1 if days_traded > 0 else 0
        
        # Calculate equity curve
        equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        current_position = 0
        
        for _, signal in signals_df.iterrows():
            current_price = signal['price']
            action = signal['action']
            
            if action == 'BUY' and current_position == 0 and current_capital > current_price:
                current_position = current_capital / current_price
                current_capital = 0
            elif action == 'SELL' and current_position > 0:
                current_capital = current_position * current_price
                current_position = 0
            
            # Current portfolio value
            portfolio_value = current_capital + (current_position * current_price)
            equity_curve.append(portfolio_value)
        
        equity_series = pd.Series(equity_curve, index=signals_df['timestamp'].tolist() + [signals_df['timestamp'].iloc[-1]])
        
        # Calculate additional metrics
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = [t for t in trades if t['action'] == 'SELL' and len(trades) > 1]
        if len(winning_trades) > 1:
            # Calculate P&L for each trade pair
            trade_pairs = []
            for i in range(0, len(trades)-1, 2):
                if i+1 < len(trades) and trades[i]['action'] == 'BUY' and trades[i+1]['action'] == 'SELL':
                    pnl = trades[i+1]['capital_gained'] - trades[i]['capital_used']
                    trade_pairs.append(pnl)
            
            win_rate = len([p for p in trade_pairs if p > 0]) / len(trade_pairs) if trade_pairs else 0
        else:
            win_rate = 0
        
        # Profit factor
        if trades:
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            if buy_trades and sell_trades:
                total_buy_value = sum(t['capital_used'] for t in buy_trades)
                total_sell_value = sum(t['capital_gained'] for t in sell_trades)
                profit_factor = total_sell_value / total_buy_value if total_buy_value > 0 else 0
            else:
                profit_factor = 0
        else:
            profit_factor = 0
        
        return BacktestResult(
            symbol=symbol,
            strategy='sentiment_enhanced' if use_sentiment else 'original',
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
            final_capital=capital
        )
    
    def backtest_multiple_symbols(self, symbols: List[str], days: int = 3, 
                                use_sentiment: bool = False, verbose: bool = False, strategies: List[str] = None) -> Dict[str, BacktestResult]:
        """Backtest multiple symbols individually"""
        results = {}
        total_symbols = len(symbols)
        
        print(f"Starting backtest for {total_symbols} symbols...")
        
        for i, symbol in enumerate(symbols):
            if symbol.upper() not in self.available_symbols:
                logger.warning(f"Symbol {symbol} not available. Available: {self.available_symbols}")
                continue
            
            print(f"[{i+1}/{total_symbols}] Backtesting {symbol}...")
            
            try:
                result = self.backtest_symbol(symbol, days, use_sentiment, verbose, strategies)
                results[symbol] = result
                print(f"[OK] {symbol}: {result.total_return:.2%} return, {result.total_trades} trades")
                logger.info(f"Completed backtest for {symbol}: {result.total_return:.2%} return")
            except Exception as e:
                print(f"[ERROR] {symbol}: Error - {e}")
                logger.error(f"Exception in backtest_symbol for {symbol}: {e}")
                # Create a dummy result to continue
                from dataclasses import dataclass
                @dataclass
                class DummyResult:
                    start_date = None
                    end_date = None
                    initial_capital = 10000
                    final_capital = 10000
                    total_return = 0.0
                    sharpe_ratio = 0.0
                    max_drawdown = 0.0
                    total_trades = 0
                    win_rate = 0.0
                    equity_curve = None
                results[symbol] = DummyResult()
                continue
        
        print(f"Backtest completed for {len(results)} symbols")
        return results
    
    def backtest_portfolio(self, symbols: List[str], days: int = 3, 
                          use_sentiment: bool = False, equal_weight: bool = True, strategies: List[str] = None) -> PortfolioResult:
        """Backtest a portfolio of symbols with rebalancing"""
        logger.info(f"Backtesting portfolio: {symbols} ({'sentiment-enhanced' if use_sentiment else 'original'} strategy)")
        
        # Load data for all symbols
        symbol_data = {}
        for symbol in symbols:
            if symbol.upper() not in self.available_symbols:
                logger.warning(f"Symbol {symbol} not available")
                continue
            df = self.load_symbol_data(symbol, days)
            if not df.empty:
                symbol_data[symbol.upper()] = df
        
        if not symbol_data:
            raise ValueError("No valid symbol data available")
        
        # Find common date range
        start_dates = [df.index.min() for df in symbol_data.values()]
        end_dates = [df.index.max() for df in symbol_data.values()]
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        # Filter all data to common range
        for symbol in symbol_data:
            symbol_data[symbol] = symbol_data[symbol][
                (symbol_data[symbol].index >= common_start) & 
                (symbol_data[symbol].index <= common_end)
            ]
        
        # Generate signals for all symbols
        symbol_signals = {}
        for symbol, df in symbol_data.items():
            signals = self.generate_signals_for_symbol(df, symbol, use_sentiment=use_sentiment)
            symbol_signals[symbol] = signals
        
        # Portfolio execution with rebalancing
        portfolio_equity = []
        rebalance_dates = []
        individual_results = {}
        
        # Initialize portfolio
        capital_per_symbol = self.initial_capital / len(symbol_data)
        symbol_positions = {symbol: 0 for symbol in symbol_data.keys()}
        symbol_capital = {symbol: capital_per_symbol for symbol in symbol_data.keys()}
        
        # Get all unique timestamps
        all_timestamps = set()
        for signals in symbol_signals.values():
            all_timestamps.update(signals['timestamp'].tolist())
        all_timestamps = sorted(list(all_timestamps))
        
        # Add initial portfolio value
        portfolio_equity.append(self.initial_capital)
        
        for timestamp in all_timestamps:
            # Check if we should rebalance (daily)
            if len(portfolio_equity) == 1 or timestamp.date() != all_timestamps[all_timestamps.index(timestamp)-1].date():
                rebalance_dates.append(timestamp)
                
                # Rebalance portfolio
                total_value = sum(
                    symbol_capital[symbol] + (symbol_positions[symbol] * 
                    symbol_data[symbol][symbol_data[symbol].index <= timestamp]['close'].iloc[-1])
                    for symbol in symbol_data.keys()
                )
                
                if equal_weight:
                    target_value_per_symbol = total_value / len(symbol_data)
                    for symbol in symbol_data.keys():
                        current_price = symbol_data[symbol][symbol_data[symbol].index <= timestamp]['close'].iloc[-1]
                        current_value = symbol_capital[symbol] + (symbol_positions[symbol] * current_price)
                        
                        if current_value < target_value_per_symbol:
                            # Buy more
                            additional_capital = target_value_per_symbol - current_value
                            if additional_capital > 0:
                                shares_to_buy = additional_capital / current_price
                                symbol_positions[symbol] += shares_to_buy
                                symbol_capital[symbol] -= additional_capital
                        elif current_value > target_value_per_symbol:
                            # Sell some
                            excess_value = current_value - target_value_per_symbol
                            shares_to_sell = excess_value / current_price
                            symbol_positions[symbol] = max(0, symbol_positions[symbol] - shares_to_sell)
                            symbol_capital[symbol] += excess_value
            
            # Execute individual symbol trades based on signals
            for symbol, signals in symbol_signals.items():
                symbol_signals_at_time = signals[signals['timestamp'] == timestamp]
                if not symbol_signals_at_time.empty:
                    signal = symbol_signals_at_time.iloc[0]
                    current_price = signal['price']
                    action = signal['action']
                    
                    if action == 'BUY' and symbol_positions[symbol] == 0 and symbol_capital[symbol] > current_price:
                        shares = symbol_capital[symbol] / current_price
                        symbol_positions[symbol] = shares
                        symbol_capital[symbol] = 0
                    elif action == 'SELL' and symbol_positions[symbol] > 0:
                        symbol_capital[symbol] = symbol_positions[symbol] * current_price
                        symbol_positions[symbol] = 0
            
            # Calculate total portfolio value
            total_value = sum(
                symbol_capital[symbol] + (symbol_positions[symbol] * 
                symbol_data[symbol][symbol_data[symbol].index <= timestamp]['close'].iloc[-1])
                for symbol in symbol_data.keys()
            )
            portfolio_equity.append(total_value)
        
        # Calculate portfolio metrics
        # Ensure lengths match by trimming portfolio_equity to match timestamps
        min_length = min(len(portfolio_equity), len(all_timestamps))
        portfolio_equity_series = pd.Series(portfolio_equity[:min_length], index=all_timestamps[:min_length])
        total_return = (portfolio_equity_series.iloc[-1] - self.initial_capital) / self.initial_capital
        days_traded = (all_timestamps[-1] - all_timestamps[0]).days
        annualized_return = (1 + total_return) ** (365 / days_traded) - 1 if days_traded > 0 else 0
        
        returns = portfolio_equity_series.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        rolling_max = portfolio_equity_series.expanding().max()
        drawdown = (portfolio_equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate correlation matrix
        symbol_returns = {}
        for symbol, df in symbol_data.items():
            symbol_returns[symbol] = df['close'].pct_change().dropna()
        
        correlation_matrix = pd.DataFrame(symbol_returns).corr()
        
        return PortfolioResult(
            symbols=list(symbol_data.keys()),
            strategy='sentiment_enhanced' if use_sentiment else 'original',
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            correlation_matrix=correlation_matrix,
            individual_results={},  # Will be populated separately
            portfolio_equity_curve=portfolio_equity_series,
            rebalance_dates=rebalance_dates
        )
    
    def create_comparative_report(self, results: Dict[str, BacktestResult], 
                                portfolio_result: Optional[PortfolioResult] = None,
                                output_dir: str = "backtests/results") -> str:
        """Create comprehensive comparative report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create summary table
        summary_data = []
        for symbol, result in results.items():
            summary_data.append({
                'Symbol': symbol,
                'Strategy': result.strategy,
                'Total Return': f"{result.total_return:.2%}",
                'Annualized Return': f"{result.annualized_return:.2%}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                'Max Drawdown': f"{result.max_drawdown:.2%}",
                'Win Rate': f"{result.win_rate:.2%}",
                'Total Trades': result.total_trades,
                'Profit Factor': f"{result.profit_factor:.2f}",
                'Final Capital': f"${result.final_capital:,.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multi-Symbol Backtest Results', fontsize=16, fontweight='bold')
        
        # 1. Equity curves comparison
        ax1 = axes[0, 0]
        for symbol, result in results.items():
            normalized_equity = result.equity_curve / result.initial_capital
            ax1.plot(normalized_equity.index, normalized_equity.values, 
                    label=f"{symbol} ({result.strategy})", linewidth=2)
        
        if portfolio_result:
            normalized_portfolio = portfolio_result.portfolio_equity_curve / self.initial_capital
            ax1.plot(normalized_portfolio.index, normalized_portfolio.values, 
                    label="Portfolio (Equal Weight)", linewidth=3, linestyle='--', alpha=0.8)
        
        ax1.set_title('Equity Curves Comparison')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Normalized Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns comparison
        ax2 = axes[0, 1]
        symbols = list(results.keys())
        returns = [results[symbol].total_return for symbol in symbols]
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        bars = ax2.bar(symbols, returns, color=colors, alpha=0.7)
        ax2.set_title('Total Returns by Symbol')
        ax2.set_ylabel('Total Return')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                    f'{ret:.1%}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 3. Risk metrics
        ax3 = axes[1, 0]
        sharpe_ratios = [results[symbol].sharpe_ratio for symbol in symbols]
        max_drawdowns = [abs(results[symbol].max_drawdown) for symbol in symbols]
        
        x = np.arange(len(symbols))
        width = 0.35
        
        ax3.bar(x - width/2, sharpe_ratios, width, label='Sharpe Ratio', alpha=0.7)
        ax3_twin = ax3.twinx()
        ax3_twin.bar(x + width/2, max_drawdowns, width, label='Max Drawdown', alpha=0.7, color='red')
        
        ax3.set_title('Risk Metrics')
        ax3.set_xlabel('Symbol')
        ax3.set_ylabel('Sharpe Ratio')
        ax3_twin.set_ylabel('Max Drawdown')
        ax3.set_xticks(x)
        ax3.set_xticklabels(symbols)
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        # 4. Correlation heatmap (if portfolio result available)
        ax4 = axes[1, 1]
        if portfolio_result and len(portfolio_result.correlation_matrix) > 1:
            sns.heatmap(portfolio_result.correlation_matrix, annot=True, cmap='coolwarm', 
                       center=0, ax=ax4, cbar_kws={'shrink': 0.8})
            ax4.set_title('Symbol Correlation Matrix')
        else:
            # Show win rates instead
            win_rates = [results[symbol].win_rate for symbol in symbols]
            ax4.bar(symbols, win_rates, alpha=0.7, color='skyblue')
            ax4.set_title('Win Rates by Symbol')
            ax4.set_ylabel('Win Rate')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / f"multi_symbol_backtest_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create markdown report
        report_path = output_path / f"multi_symbol_backtest_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Multi-Symbol Crypto Backtest Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Symbols:** {', '.join(symbols)}\n\n")
            f.write(f"**Initial Capital:** ${self.initial_capital:,.2f}\n\n")
            f.write(f"**Strategy:** {'Sentiment-Enhanced' if any(r.strategy == 'sentiment_enhanced' for r in results.values()) else 'Original'}\n\n")
            
            f.write("## Summary Table\n\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("## Key Insights\n\n")
            
            # Best performing symbol
            best_symbol = max(results.keys(), key=lambda s: results[s].total_return)
            f.write(f"- **Best Performer:** {best_symbol} ({results[best_symbol].total_return:.2%} return)\n")
            
            # Worst performing symbol
            worst_symbol = min(results.keys(), key=lambda s: results[s].total_return)
            f.write(f"- **Worst Performer:** {worst_symbol} ({results[worst_symbol].total_return:.2%} return)\n")
            
            # Average metrics
            avg_return = np.mean([r.total_return for r in results.values()])
            avg_sharpe = np.mean([r.sharpe_ratio for r in results.values()])
            avg_drawdown = np.mean([abs(r.max_drawdown) for r in results.values()])
            
            f.write(f"- **Average Return:** {avg_return:.2%}\n")
            f.write(f"- **Average Sharpe Ratio:** {avg_sharpe:.2f}\n")
            f.write(f"- **Average Max Drawdown:** {avg_drawdown:.2%}\n\n")
            
            if portfolio_result:
                f.write("## Portfolio Analysis\n\n")
                f.write(f"- **Portfolio Return:** {portfolio_result.total_return:.2%}\n")
                f.write(f"- **Portfolio Sharpe Ratio:** {portfolio_result.sharpe_ratio:.2f}\n")
                f.write(f"- **Portfolio Max Drawdown:** {portfolio_result.max_drawdown:.2%}\n\n")
            
            f.write("## Files Generated\n\n")
            f.write(f"- **Plot:** `{plot_path.name}`\n")
            f.write(f"- **Report:** `{report_path.name}`\n\n")
        
        logger.info(f"Comparative report saved to: {report_path}")
        return str(report_path)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Multi-Symbol Crypto Backtester')
    parser.add_argument('--symbols', nargs='+', required=True, 
                       help='Symbols to backtest (e.g., BTC ETH ADA)')
    parser.add_argument('--strategies', nargs='*', default=None,
                       help='Strategies to use (e.g., btc_asia_sweep eth_breakout_continuation). Default: all')
    parser.add_argument('--days', type=int, default=3, 
                       help='Number of days to backtest (default: 3)')
    parser.add_argument('--capital', type=float, default=10000.0, 
                       help='Initial capital (default: 10000)')
    parser.add_argument('--alpha', type=float, default=0.5, 
                       help='Sentiment alpha parameter (default: 0.5)')
    parser.add_argument('--sentiment', action='store_true', 
                       help='Use sentiment-enhanced strategy')
    parser.add_argument('--portfolio', action='store_true', 
                       help='Run portfolio backtest with rebalancing')
    parser.add_argument('--verbose', action='store_true', 
                       help='Verbose output showing all trades')
    parser.add_argument('--output-dir', default='backtests/results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create backtester
    backtester = MultiSymbolBacktester(
        initial_capital=args.capital,
        alpha=args.alpha
    )
    
    print(f"Available symbols: {backtester.available_symbols}")
    print(f"Requested symbols: {args.symbols}")
    
    # Validate symbols
    valid_symbols = [s.upper() for s in args.symbols if s.upper() in backtester.available_symbols]
    if not valid_symbols:
        print("No valid symbols found!")
        return
    
    print(f"Backtesting symbols: {valid_symbols}")
    
    try:
        # Run individual symbol backtests
        results = backtester.backtest_multiple_symbols(
            valid_symbols, 
            days=args.days, 
            use_sentiment=args.sentiment,
            verbose=args.verbose,
            strategies=args.strategies
        )
        
        # Run portfolio backtest if requested
        portfolio_result = None
        if args.portfolio and len(valid_symbols) > 1:
            portfolio_result = backtester.backtest_portfolio(
                valid_symbols,
                days=args.days,
                use_sentiment=args.sentiment,
                strategies=args.strategies
            )
        
        # Generate report
        report_path = backtester.create_comparative_report(
            results, 
            portfolio_result,
            args.output_dir
        )
        
        print(f"\nBacktest completed! Report saved to: {report_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        
        for symbol, result in results.items():
            print(f"\n{symbol.upper()}:")
            print(f"  Return: {result.total_return:.2%}")
            print(f"  Sharpe: {result.sharpe_ratio:.2f}")
            print(f"  Max DD: {result.max_drawdown:.2%}")
            print(f"  Trades: {result.total_trades}")
        
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
