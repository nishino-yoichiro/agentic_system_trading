#!/usr/bin/env python3
"""
Parameter Sweep Backtester
Runs sensitivity analysis on strategy parameters to optimize signal density vs PnL trade-off
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import itertools
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from crypto_analysis_engine import CryptoAnalysisEngine
from crypto_signal_integration import CryptoSignalIntegration
# Removed old CryptoTradingStrategies import - using new dynamic system
from crypto_signal_framework import SignalType
from utils.progress_logger import progress_logger, create_parameter_sweep_progress, create_signal_progress

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ParameterSweepResult:
    """Container for parameter sweep results"""
    strategy_name: str
    symbol: str
    parameters: Dict[str, Any]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    signal_count_per_day: float
    avg_pnl_per_trade: float
    hit_rate: float  # win_rate * signal_count_per_day
    score: float  # hit_rate * avg_pnl_per_trade

class ParameterSweepBacktester:
    """Parameter sweep backtester for strategy optimization"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.analysis_engine = CryptoAnalysisEngine()
        # Use new dynamic signal integration instead of old strategies
        self.signal_integration = CryptoSignalIntegration()
        
        # Available symbols from data directory
        self.available_symbols = self._get_available_symbols()
        
        # Parameter ranges for each strategy (reduced for testing)
        self.parameter_ranges = {
            'liquidity_sweep_reversal': {
                'swing_lookback': [15, 20],  # Reduced from 3 to 2
                'volume_multiplier': [1.2, 1.5],  # Reduced from 3 to 2
                'body_size_multiplier': [0.6, 0.8],  # Reduced from 3 to 2
                'stop_buffer': [0.003, 0.005]  # Reduced from 3 to 2
            },
            'volume_weighted_trend_continuation': {
                'trend_lookback': [15, 20],  # Reduced from 3 to 2
                'volume_lookback': [15, 20],  # Reduced from 3 to 2
                'slope_threshold': [0.00005, 0.0001],  # Reduced from 3 to 2
                'volume_multiplier': [1.0, 1.2]  # Reduced from 3 to 2
            },
            'volatility_expansion_breakout': {
                'atr_period': [10, 14],  # Reduced from 3 to 2
                'compression_percentile': [20, 30],  # Reduced from 3 to 2
                'body_multiplier': [1.2, 1.5],  # Reduced from 3 to 2
                'volume_multiplier': [1.1, 1.3],  # Reduced from 3 to 2
                'trailing_atr_multiplier': [0.3, 0.5]  # Reduced from 3 to 2
            },
            'fakeout_reversion': {
                'compression_days': [5, 7],  # Days for compression detection
                'vol_z_threshold': [1.0, 1.5],  # Volume z-score threshold (relaxed)
                'vol_reversion_threshold': [0.8, 1.0],  # Volume reversion threshold
                'atr_compression_threshold': [0.5, 0.8],  # ATR compression threshold (relaxed)
                'range_compression_threshold': [0.03, 0.05],  # Range compression threshold (relaxed)
                'rsi_extreme_threshold': [60, 70]  # RSI extreme threshold (relaxed)
            },
            'fakeout_reversion_relaxed': {
                'compression_days': [5, 7],
                'vol_z_threshold': [0.8, 1.2],  # More relaxed
                'vol_reversion_threshold': [0.6, 0.8],  # More relaxed
                'atr_compression_threshold': [0.7, 1.0],  # More relaxed
                'range_compression_threshold': [0.05, 0.08],  # More relaxed
                'rsi_extreme_threshold': [55, 65]  # More relaxed
            },
            'fakeout_reversion_very_relaxed': {
                'compression_days': [3, 5],  # Shorter compression period
                'vol_z_threshold': [0.5, 1.0],  # Very relaxed
                'vol_reversion_threshold': [0.4, 0.6],  # Very relaxed
                'atr_compression_threshold': [0.8, 1.2],  # Very relaxed
                'range_compression_threshold': [0.08, 0.12],  # Very relaxed
                'rsi_extreme_threshold': [50, 60]  # Very relaxed
            }
        }
    
    def _get_available_symbols(self) -> List[str]:
        """Get list of available symbols from historical data directory"""
        data_dir = Path("data/historical")
        symbols = []
        for file in data_dir.glob("*_historical.parquet"):
            symbol = file.stem.replace("_historical", "").upper()
            symbols.append(symbol)
        return sorted(symbols)
    
    def load_symbol_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Load historical data for a specific symbol"""
        try:
            # Load from historical data directory
            file_path = Path(f"data/historical/{symbol}_historical.parquet")
            if not file_path.exists():
                logger.error(f"Historical data file not found: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path)
            
            # Only limit if we have more data than requested
            if days and len(df) > 0:
                minutes_per_day = 1440
                max_rows = days * minutes_per_day
                if len(df) > max_rows:
                    df = df.tail(max_rows)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_signals_with_parameters(self, symbol: str, strategy_name: str, 
                                        parameters: Dict[str, Any], days: int = 30) -> pd.DataFrame:
        """Generate signals using modified strategy parameters"""
        try:
            # Load data
            df = self.load_symbol_data(symbol, days)
            if df.empty:
                return pd.DataFrame()
            
            # Get strategy function
            if strategy_name not in self.strategies.strategies:
                logger.error(f"Strategy {strategy_name} not found")
                return pd.DataFrame()
            
            strategy_func = self.strategies.strategies[strategy_name]['function']
            
            # Generate signals with modified parameters
            signals = []
            regime = {regime_type: True for regime_type in self.strategies.strategies[strategy_name]['config'].regime_filters or []}
            
            logger.info(f"Processing {len(df)} data points for {symbol}")
            
            # Process data in chunks to simulate real-time
            # Start after minimum data requirements (max of all strategy lookbacks)
            min_required_bars = max(
                parameters.get('swing_lookback', 25),
                parameters.get('trend_lookback', 25), 
                parameters.get('volume_lookback', 25),
                parameters.get('atr_period', 20),
                25  # Minimum for volume/body averages
            )
            
            # Process every 10th bar to speed up backtesting (still realistic)
            step_size = 10
            total_iterations = len(range(min_required_bars, len(df), step_size))
            
            # Create progress bar for signal generation
            pbar = create_signal_progress(total_iterations, symbol, strategy_name)
            
            # OPTIMIZATION: Pre-slice data to avoid repeated copying
            data_slices = []
            for i in range(min_required_bars, len(df), step_size):
                data_slices.append(df.iloc[:i+1])
            
            for current_data in data_slices:
                
                try:
                    # Generate signal with parameters
                    signal = strategy_func(current_data, regime, parameters)
                    
                    if signal:
                        logger.debug(f"Signal generated at {current_data.index[-1]}: {signal.signal_type}, confidence: {signal.confidence}")
                        if signal.signal_type != SignalType.FLAT:
                            signals.append({
                                'timestamp': current_data.index[-1],
                                'price': signal.entry_price,
                                'action': 'BUY' if signal.signal_type == SignalType.LONG else 'SELL',
                                'confidence': signal.confidence,
                                'reason': signal.reason,
                                'stop_loss': signal.stop_loss,
                                'take_profit': signal.take_profit
                            })
                    else:
                        logger.debug(f"No signal at {current_data.index[-1]}")
                        
                except Exception as e:
                    logger.warning(f"Error generating signal for {strategy_name} at {current_data.index[-1]}: {e}")
                    continue
                
                # Update progress bar
                pbar.set_postfix({"Signals": len(signals)})
                pbar.update(1)
            
            pbar.close()
            logger.info(f"Generated {len(signals)} signals for {symbol} using {strategy_name}")
            return pd.DataFrame(signals)
            
        except Exception as e:
            logger.error(f"Error generating signals with parameters: {e}")
            return pd.DataFrame()
    
    
    def backtest_with_parameters(self, symbol: str, strategy_name: str, 
                               parameters: Dict[str, Any], days: int = 30) -> ParameterSweepResult:
        """Backtest a single strategy with specific parameters"""
        logger.info(f"Backtesting {strategy_name} on {symbol} with parameters: {parameters}")
        
        # Generate signals
        signals_df = self.generate_signals_with_parameters(symbol, strategy_name, parameters, days)
        
        if signals_df.empty:
            return ParameterSweepResult(
                strategy_name=strategy_name,
                symbol=symbol,
                parameters=parameters,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                signal_count_per_day=0.0,
                avg_pnl_per_trade=0.0,
                hit_rate=0.0,
                score=0.0
            )
        
        # Execute trades
        trades = []
        capital = self.initial_capital
        position = 0
        equity_curve = [self.initial_capital]
        
        for _, signal in signals_df.iterrows():
            current_price = signal['price']
            action = signal['action']
            
            if action == 'BUY' and capital > current_price:
                shares = capital / current_price
                position = shares
                capital = 0
                
                trades.append({
                    'timestamp': signal['timestamp'],
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'capital_used': shares * current_price
                })
                
            elif action == 'SELL' and position > 0:
                capital = position * current_price
                
                trades.append({
                    'timestamp': signal['timestamp'],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'capital_gained': capital
                })
                
                position = 0
            
            # Update equity curve
            current_equity = capital + (position * current_price)
            equity_curve.append(current_equity)
        
        # Final liquidation
        if position > 0:
            final_price = signals_df['price'].iloc[-1]
            capital = position * final_price
            position = 0
        
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        # Calculate additional metrics
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate and trade analysis
        trade_pairs = []
        for i in range(0, len(trades)-1, 2):
            if i+1 < len(trades) and trades[i]['action'] == 'BUY' and trades[i+1]['action'] == 'SELL':
                pnl = trades[i+1]['capital_gained'] - trades[i]['capital_used']
                trade_pairs.append(pnl)
        
        win_rate = len([p for p in trade_pairs if p > 0]) / len(trade_pairs) if trade_pairs else 0
        avg_pnl_per_trade = np.mean(trade_pairs) if trade_pairs else 0
        
        # Signal density
        days_traded = (signals_df['timestamp'].iloc[-1] - signals_df['timestamp'].iloc[0]).days
        signal_count_per_day = len(signals_df) / days_traded if days_traded > 0 else 0
        
        # Hit rate (win_rate * signal_count_per_day)
        hit_rate = win_rate * signal_count_per_day
        
        # Score (hit_rate * avg_pnl_per_trade)
        score = hit_rate * avg_pnl_per_trade
        
        return ParameterSweepResult(
            strategy_name=strategy_name,
            symbol=symbol,
            parameters=parameters,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            signal_count_per_day=signal_count_per_day,
            avg_pnl_per_trade=avg_pnl_per_trade,
            hit_rate=hit_rate,
            score=score
        )
    
    def run_parameter_sweep(self, symbols: List[str], strategies: List[str] = None, 
                          days: int = 30) -> List[ParameterSweepResult]:
        """Run parameter sweep for specified strategies and symbols"""
        if strategies is None:
            strategies = list(self.parameter_ranges.keys())
        
        results = []
        total_combinations = 0
        
        # Calculate total combinations
        for strategy in strategies:
            if strategy in self.parameter_ranges:
                param_combinations = 1
                for param_values in self.parameter_ranges[strategy].values():
                    param_combinations *= len(param_values)
                total_combinations += param_combinations * len(symbols)
        
        logger.info(f"Running parameter sweep: {total_combinations} total combinations")
        
        # Create progress bar for parameter sweep
        pbar = create_parameter_sweep_progress(total_combinations, symbols, strategies)
        
        for symbol in symbols:
            if symbol not in self.available_symbols:
                logger.warning(f"Symbol {symbol} not available")
                continue
            
            for strategy in strategies:
                if strategy not in self.parameter_ranges:
                    logger.warning(f"Strategy {strategy} not in parameter ranges")
                    continue
                
                # Generate all parameter combinations
                param_names = list(self.parameter_ranges[strategy].keys())
                param_values = list(self.parameter_ranges[strategy].values())
                
                for param_combination in itertools.product(*param_values):
                    parameters = dict(zip(param_names, param_combination))
                    
                    # Update progress bar description
                    pbar.set_description(f"Testing {strategy} on {symbol}")
                    
                    try:
                        result = self.backtest_with_parameters(symbol, strategy, parameters, days)
                        results.append(result)
                        
                        # Update progress bar with results
                        if result.total_trades > 0:
                            pbar.set_postfix({
                                "Trades": result.total_trades,
                                "Return": f"{result.total_return:.2%}",
                                "Score": f"{result.score:.3f}"
                            })
                        else:
                            pbar.set_postfix({"Trades": 0, "Return": "0%", "Score": "0.000"})
                        
                    except Exception as e:
                        logger.error(f"Error in parameter sweep for {strategy} on {symbol}: {e}")
                        pbar.set_postfix({"Error": str(e)[:20]})
                    
                    pbar.update(1)
        
        pbar.close()
        logger.info(f"Parameter sweep completed: {len(results)} results")
        return results
    
    def analyze_results(self, results: List[ParameterSweepResult], 
                       output_dir: str = "backtests/results") -> str:
        """Analyze parameter sweep results and create report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame([asdict(result) for result in results])
        
        if results_df.empty:
            logger.warning("No results to analyze")
            return ""
        
        # Create summary statistics
        summary_stats = results_df.groupby(['strategy_name', 'symbol']).agg({
            'total_return': ['mean', 'std', 'min', 'max'],
            'sharpe_ratio': ['mean', 'std'],
            'max_drawdown': ['mean', 'std'],
            'win_rate': ['mean', 'std'],
            'total_trades': ['mean', 'std'],
            'signal_count_per_day': ['mean', 'std'],
            'avg_pnl_per_trade': ['mean', 'std'],
            'hit_rate': ['mean', 'std'],
            'score': ['mean', 'std']
        }).round(4)
        
        # Find best parameters for each strategy
        best_params = {}
        for strategy in results_df['strategy_name'].unique():
            strategy_results = results_df[results_df['strategy_name'] == strategy]
            
            # Sort by score (hit_rate * avg_pnl_per_trade)
            best_result = strategy_results.loc[strategy_results['score'].idxmax()]
            best_params[strategy] = {
                'parameters': best_result['parameters'],
                'score': best_result['score'],
                'hit_rate': best_result['hit_rate'],
                'avg_pnl_per_trade': best_result['avg_pnl_per_trade'],
                'signal_count_per_day': best_result['signal_count_per_day'],
                'total_return': best_result['total_return'],
                'sharpe_ratio': best_result['sharpe_ratio']
            }
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Parameter Sweep Analysis: Signal Count vs PnL Trade-off', fontsize=16, fontweight='bold')
        
        # 1. Signal Count vs PnL scatter plot
        ax1 = axes[0, 0]
        for strategy in results_df['strategy_name'].unique():
            strategy_data = results_df[results_df['strategy_name'] == strategy]
            ax1.scatter(strategy_data['signal_count_per_day'], 
                       strategy_data['avg_pnl_per_trade'],
                       label=strategy, alpha=0.6, s=50)
        
        ax1.set_xlabel('Signals per Day')
        ax1.set_ylabel('Average PnL per Trade')
        ax1.set_title('Signal Density vs PnL Trade-off')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Hit Rate vs Score
        ax2 = axes[0, 1]
        for strategy in results_df['strategy_name'].unique():
            strategy_data = results_df[results_df['strategy_name'] == strategy]
            ax2.scatter(strategy_data['hit_rate'], 
                       strategy_data['score'],
                       label=strategy, alpha=0.6, s=50)
        
        ax2.set_xlabel('Hit Rate (Win Rate × Signal Count)')
        ax2.set_ylabel('Score (Hit Rate × Avg PnL)')
        ax2.set_title('Hit Rate vs Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Return vs Signal Count
        ax3 = axes[0, 2]
        for strategy in results_df['strategy_name'].unique():
            strategy_data = results_df[results_df['strategy_name'] == strategy]
            ax3.scatter(strategy_data['signal_count_per_day'], 
                       strategy_data['total_return'],
                       label=strategy, alpha=0.6, s=50)
        
        ax3.set_xlabel('Signals per Day')
        ax3.set_ylabel('Total Return')
        ax3.set_title('Return vs Signal Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Best parameters heatmap
        ax4 = axes[1, 0]
        best_scores = [best_params[strategy]['score'] for strategy in best_params.keys()]
        strategy_names = list(best_params.keys())
        
        bars = ax4.bar(strategy_names, best_scores, alpha=0.7)
        ax4.set_title('Best Score by Strategy')
        ax4.set_ylabel('Score')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, best_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 5. Parameter sensitivity heatmap
        ax5 = axes[1, 1]
        if len(results_df) > 0:
            # Create correlation matrix for numeric parameters
            numeric_cols = ['total_return', 'sharpe_ratio', 'win_rate', 'signal_count_per_day', 'avg_pnl_per_trade']
            corr_matrix = results_df[numeric_cols].corr()
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax5)
            ax5.set_title('Parameter Correlations')
        
        # 6. Signal count distribution
        ax6 = axes[1, 2]
        for strategy in results_df['strategy_name'].unique():
            strategy_data = results_df[results_df['strategy_name'] == strategy]
            ax6.hist(strategy_data['signal_count_per_day'], alpha=0.6, 
                    label=strategy, bins=20)
        
        ax6.set_xlabel('Signals per Day')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Signal Count Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / f"parameter_sweep_analysis_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed report
        report_path = output_path / f"parameter_sweep_report_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Parameter Sweep Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Combinations Tested:** {len(results_df)}\n\n")
            
            f.write("## Best Parameters by Strategy\n\n")
            for strategy, params in best_params.items():
                f.write(f"### {strategy}\n\n")
                f.write(f"- **Score:** {params['score']:.4f}\n")
                f.write(f"- **Hit Rate:** {params['hit_rate']:.4f}\n")
                f.write(f"- **Avg PnL per Trade:** ${params['avg_pnl_per_trade']:.2f}\n")
                f.write(f"- **Signals per Day:** {params['signal_count_per_day']:.2f}\n")
                f.write(f"- **Total Return:** {params['total_return']:.2%}\n")
                f.write(f"- **Sharpe Ratio:** {params['sharpe_ratio']:.2f}\n\n")
                f.write("**Optimal Parameters:**\n")
                for param, value in params['parameters'].items():
                    f.write(f"- {param}: {value}\n")
                f.write("\n")
            
            f.write("## Summary Statistics\n\n")
            f.write(summary_stats.to_markdown())
            f.write("\n\n")
            
            f.write("## Key Insights\n\n")
            
            # Find strategies that achieve target signal density (3-5 per day)
            target_strategies = results_df[
                (results_df['signal_count_per_day'] >= 3) & 
                (results_df['signal_count_per_day'] <= 5)
            ]
            
            if not target_strategies.empty:
                f.write("### Strategies Achieving Target Signal Density (3-5 per day)\n\n")
                best_target = target_strategies.loc[target_strategies['score'].idxmax()]
                f.write(f"- **Best:** {best_target['strategy_name']} on {best_target['symbol']}\n")
                f.write(f"  - Score: {best_target['score']:.4f}\n")
                f.write(f"  - Signals/day: {best_target['signal_count_per_day']:.2f}\n")
                f.write(f"  - Avg PnL: ${best_target['avg_pnl_per_trade']:.2f}\n")
                f.write(f"  - Parameters: {best_target['parameters']}\n\n")
            
            f.write("### Recommendations\n\n")
            f.write("1. **Volume Threshold Optimization:** Lower volume multipliers (1.2-1.5x) increase signal density\n")
            f.write("2. **Confirmation Logic:** Relax confirmation requirements to allow EITHER volume OR sentiment\n")
            f.write("3. **Timeframe Adjustment:** Consider 3-5 minute bars for more context\n")
            f.write("4. **Parameter Sensitivity:** Focus on parameters with highest correlation to score\n\n")
            
            f.write("## Files Generated\n\n")
            f.write(f"- **Analysis Plot:** `{plot_path.name}`\n")
            f.write(f"- **Report:** `{report_path.name}`\n")
            f.write(f"- **Raw Data:** `parameter_sweep_results_{timestamp}.json`\n\n")
        
        # Save raw results
        results_json_path = output_path / f"parameter_sweep_results_{timestamp}.json"
        with open(results_json_path, 'w') as f:
            json.dump([asdict(result) for result in results], f, indent=2, default=str)
        
        logger.info(f"Parameter sweep analysis saved to: {report_path}")
        return str(report_path)

def main():
    """Main function for parameter sweep"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parameter Sweep Backtester')
    parser.add_argument('--symbols', nargs='+', default=['BTC', 'ETH'], 
                       help='Symbols to test (default: BTC ETH)')
    parser.add_argument('--strategies', nargs='*', default=None,
                       help='Strategies to test. Default: all except btc_ny_session')
    parser.add_argument('--days', type=int, default=30, 
                       help='Days to backtest (default: 30)')
    parser.add_argument('--capital', type=float, default=10000.0, 
                       help='Initial capital (default: 10000)')
    parser.add_argument('--output-dir', default='backtests/results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create parameter sweep backtester
    sweeper = ParameterSweepBacktester(initial_capital=args.capital)
    
    print(f"Available symbols: {sweeper.available_symbols}")
    print(f"Testing symbols: {args.symbols}")
    print(f"Parameter ranges: {sweeper.parameter_ranges}")
    
    # Validate symbols
    valid_symbols = [s.upper() for s in args.symbols if s.upper() in sweeper.available_symbols]
    if not valid_symbols:
        print("No valid symbols found!")
        return
    
    print(f"Running parameter sweep for: {valid_symbols}")
    
    try:
        # Run parameter sweep
        results = sweeper.run_parameter_sweep(
            valid_symbols, 
            strategies=args.strategies,
            days=args.days
        )
        
        # Analyze results
        report_path = sweeper.analyze_results(results, args.output_dir)
        
        print(f"\nParameter sweep completed! Report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("PARAMETER SWEEP SUMMARY")
        print("="*80)
        
        if results:
            # Group by strategy
            strategy_results = {}
            for result in results:
                if result.strategy_name not in strategy_results:
                    strategy_results[result.strategy_name] = []
                strategy_results[result.strategy_name].append(result)
            
            for strategy, strategy_list in strategy_results.items():
                best_result = max(strategy_list, key=lambda x: x.score)
                print(f"\n{strategy.upper()}:")
                print(f"  Best Score: {best_result.score:.4f}")
                print(f"  Signals/Day: {best_result.signal_count_per_day:.2f}")
                print(f"  Avg PnL: ${best_result.avg_pnl_per_trade:.2f}")
                print(f"  Return: {best_result.total_return:.2%}")
                print(f"  Parameters: {best_result.parameters}")
        
    except Exception as e:
        logger.error(f"Parameter sweep failed: {e}")
        raise

if __name__ == "__main__":
    main()
