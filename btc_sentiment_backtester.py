"""
BTC Sentiment-Enhanced Backtester
Compares original strategy vs sentiment-enhanced strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path
import logging
import argparse

from btc_analysis_engine import BTCAnalysisEngine
from btc_signal_generator import BTCSignalGenerator
from btc_sentiment_enhanced_generator import BTCSentimentEnhancedGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class BTCSentimentBacktester:
    """Backtester comparing original vs sentiment-enhanced strategies"""
    
    def __init__(self, initial_capital: float = 10000.0, alpha: float = 0.5):
        self.initial_capital = initial_capital
        self.alpha = alpha
        self.analysis_engine = BTCAnalysisEngine()
        self.original_generator = BTCSignalGenerator()
        self.sentiment_generator = BTCSentimentEnhancedGenerator(alpha=alpha)
        
    def load_historical_data(self, days: int = 3) -> pd.DataFrame:
        """Load historical BTC data for backtesting"""
        try:
            df = self.analysis_engine.load_real_btc_data(days=days)
            if df.empty:
                raise ValueError("No historical data available")
            
            logger.info(f"Loaded {len(df)} data points for backtesting")
            return df
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def generate_signals_for_period(self, df: pd.DataFrame, lookback_hours: int = 24, 
                                  use_sentiment: bool = False) -> pd.DataFrame:
        """Generate trading signals for each point in the historical data"""
        signals = []
        
        # Use rolling window approach
        for i in range(lookback_hours, len(df)):
            historical_data = df.iloc[:i].copy()
            
            if len(historical_data) < 100:
                continue
                
            try:
                # Generate signal using historical data
                indicators = self.analysis_engine.calculate_technical_indicators(historical_data)
                indicators_df = pd.DataFrame([indicators])
                
                if use_sentiment:
                    signal = self.sentiment_generator.generate_enhanced_signals(
                        historical_data, indicators_df, symbol="BTC")
                else:
                    signal = self.original_generator.generate_btc_signals(
                        historical_data, indicators_df)
                
                # Add timestamp and price
                signal['timestamp'] = df.index[i]
                signal['price'] = df.iloc[i]['close']
                signal['volume'] = df.iloc[i]['volume']
                
                signals.append(signal)
                
            except Exception as e:
                logger.warning(f"Error generating signal at index {i}: {e}")
                continue
        
        if not signals:
            logger.error("No signals generated")
            return pd.DataFrame()
            
        signals_df = pd.DataFrame(signals)
        signals_df.set_index('timestamp', inplace=True)
        
        strategy_type = "Sentiment-Enhanced" if use_sentiment else "Original"
        logger.info(f"Generated {len(signals_df)} {strategy_type} trading signals")
        
        # Debug: Show signal distribution
        signal_counts = signals_df['action'].value_counts()
        logger.info(f"{strategy_type} signal distribution: {signal_counts.to_dict()}")
        
        return signals_df
    
    def simulate_trades(self, df: pd.DataFrame, signals_df: pd.DataFrame, 
                       strategy_name: str = "Strategy") -> Dict:
        """Simulate trading based on signals with proper position tracking"""
        
        # Initialize tracking variables
        capital = self.initial_capital
        btc_position = 0.0
        trades = []
        equity_curve = []
        
        # Track performance metrics
        total_trades = 0
        winning_trades = 0
        total_pnl = 0.0
        max_capital = self.initial_capital
        max_drawdown = 0.0
        
        logger.info(f"Starting {strategy_name} trade simulation...")
        
        for i, (timestamp, signal) in enumerate(signals_df.iterrows()):
            current_price = signal['price']
            action = signal['action']
            confidence = signal.get('confidence', 0.5)

            # Skip low confidence or HOLD signals
            if confidence < 0.3 or action == 'HOLD':
                continue

            # Get next price for execution
            next_idx = df.index.get_loc(timestamp) + 1
            if next_idx >= len(df):
                continue
                
            next_price = df.iloc[next_idx]['close']
            
            # Execute trade based on signal
            if action in ['BUY', 'STRONG_BUY'] and btc_position == 0:
                trade_amount = capital * 0.5
                btc_position = trade_amount / next_price
                capital -= trade_amount
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': next_price,
                    'amount': trade_amount,
                    'btc_position': btc_position,
                    'capital': capital
                })
                total_trades += 1
                
            elif action in ['SELL', 'STRONG_SELL'] and btc_position > 0:
                sell_amount = btc_position * next_price
                capital += sell_amount
                
                # Calculate PnL for this trade
                buy_trade = trades[-1] if trades else None
                pnl = 0.0
                if buy_trade and buy_trade['action'] == 'BUY':
                    pnl = sell_amount - buy_trade['amount']
                    total_pnl += pnl
                    
                    if pnl > 0:
                        winning_trades += 1
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': next_price,
                    'amount': sell_amount,
                    'btc_position': 0,
                    'capital': capital
                })
                total_trades += 1
                btc_position = 0
            
            # Calculate current equity
            current_equity = capital + (btc_position * next_price)
            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'capital': capital,
                'position_value': btc_position * next_price
            })
            
            # Track max drawdown
            if current_equity > max_capital:
                max_capital = current_equity
            else:
                drawdown = (max_capital - current_equity) / max_capital
                max_drawdown = max(max_drawdown, drawdown)
        
        # Close any remaining position
        if btc_position > 0:
            final_price = df.iloc[-1]['close']
            final_amount = btc_position * final_price
            capital += final_amount
            trades.append({
                'timestamp': df.index[-1],
                'action': 'SELL',
                'price': final_price,
                'amount': final_amount,
                'btc_position': 0,
                'capital': capital
            })
            total_trades += 1
            btc_position = 0
        
        # Calculate final metrics
        final_equity = capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        win_rate = (winning_trades / max(1, total_trades)) * 100
        
        results = {
            'strategy_name': strategy_name,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'final_equity': final_equity,
            'trades': trades,
            'equity_curve': equity_curve
        }
        
        logger.info(f"{strategy_name} simulation complete: {total_trades} trades, "
                   f"{win_rate:.1f}% win rate, {total_return:.2%} return")
        
        return results
    
    def create_comparison_plot(self, original_results: Dict, sentiment_results: Dict, 
                             output_path: str = "btc_sentiment_comparison.png"):
        """Create comparison plot showing both strategies"""
        
        if not original_results['equity_curve'] or not sentiment_results['equity_curve']:
            logger.error("No equity curve data available for plotting")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Equity curves comparison
        orig_curve = pd.DataFrame(original_results['equity_curve'])
        sent_curve = pd.DataFrame(sentiment_results['equity_curve'])
        
        ax1.plot(orig_curve['timestamp'], orig_curve['equity'], 
                label=f"Original ({original_results['total_return']:.2%})", 
                linewidth=2, color='blue')
        ax1.plot(sent_curve['timestamp'], sent_curve['equity'], 
                label=f"Sentiment-Enhanced ({sentiment_results['total_return']:.2%})", 
                linewidth=2, color='red')
        
        ax1.set_title('BTC Trading Strategy Comparison: Original vs Sentiment-Enhanced', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Performance metrics comparison
        metrics = ['Total Return', 'Win Rate', 'Max Drawdown']
        original_values = [
            original_results['total_return'] * 100,
            original_results['win_rate'],
            original_results['max_drawdown'] * 100
        ]
        sentiment_values = [
            sentiment_results['total_return'] * 100,
            sentiment_results['win_rate'],
            sentiment_results['max_drawdown'] * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax2.bar(x - width/2, original_values, width, label='Original', color='blue', alpha=0.7)
        ax2.bar(x + width/2, sentiment_values, width, label='Sentiment-Enhanced', color='red', alpha=0.7)
        
        ax2.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (orig_val, sent_val) in enumerate(zip(original_values, sentiment_values)):
            ax2.text(i - width/2, orig_val + 0.5, f'{orig_val:.1f}%', 
                    ha='center', va='bottom', fontsize=10)
            ax2.text(i + width/2, sent_val + 0.5, f'{sent_val:.1f}%', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved to {output_path}")
    
    def create_comparison_report(self, original_results: Dict, sentiment_results: Dict,
                               output_path: str = "btc_sentiment_comparison.md"):
        """Create markdown comparison report"""
        
        report = f"""# BTC Trading Strategy Comparison Report

## Strategy Parameters
- **Time Period**: {len(original_results['equity_curve'])} data points
- **Initial Capital**: ${self.initial_capital:,.0f}
- **Sentiment Alpha**: {self.alpha}
- **Lookback Window**: 24 hours

## Performance Comparison

| Metric | Original Strategy | Sentiment-Enhanced | Improvement |
|--------|------------------|-------------------|-------------|
| **Total Trades** | {original_results['total_trades']} | {sentiment_results['total_trades']} | {sentiment_results['total_trades'] - original_results['total_trades']:+d} |
| **Win Rate** | {original_results['win_rate']:.1f}% | {sentiment_results['win_rate']:.1f}% | {sentiment_results['win_rate'] - original_results['win_rate']:+.1f}% |
| **Total Return** | {original_results['total_return']:.2%} | {sentiment_results['total_return']:.2%} | {sentiment_results['total_return'] - original_results['total_return']:+.2%} |
| **Max Drawdown** | {original_results['max_drawdown']:.2%} | {sentiment_results['max_drawdown']:.2%} | {sentiment_results['max_drawdown'] - original_results['max_drawdown']:+.2%} |
| **Final Equity** | ${original_results['final_equity']:,.0f} | ${sentiment_results['final_equity']:,.0f} | ${sentiment_results['final_equity'] - original_results['final_equity']:+,.0f} |

## Analysis

### Sentiment Enhancement Impact
The sentiment-enhanced strategy incorporates news sentiment analysis within ±30 minutes of trading decisions. The sentiment multiplier formula is:

**Enhanced Signal = Base Signal × (1 + α × Sentiment)**

Where:
- α = {self.alpha} (tunable parameter)
- Sentiment = normalized sentiment score [-1, 1]

### Key Findings
- **Sentiment Impact**: {'Positive' if sentiment_results['total_return'] > original_results['total_return'] else 'Negative'} impact on overall returns
- **Trade Frequency**: {'Increased' if sentiment_results['total_trades'] > original_results['total_trades'] else 'Decreased'} number of trades
- **Risk Management**: {'Improved' if sentiment_results['max_drawdown'] < original_results['max_drawdown'] else 'Worsened'} drawdown control

### Recommendation
{'The sentiment-enhanced strategy shows promise and should be considered for live trading.' if sentiment_results['total_return'] > original_results['total_return'] else 'The original strategy performs better. Consider adjusting sentiment parameters or improving sentiment analysis.'}

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Comparison report saved to {output_path}")
    
    def run_comparison_backtest(self, days: int = 3, lookback_hours: int = 24) -> Tuple[Dict, Dict]:
        """Run comparison backtest between original and sentiment-enhanced strategies"""
        
        logger.info(f"Starting comparison backtest for {days} days with {lookback_hours}h lookback")
        
        # Create output directories
        backtest_dir = Path("backtests")
        results_dir = backtest_dir / "results"
        backtest_dir.mkdir(exist_ok=True)
        results_dir.mkdir(exist_ok=True)
        
        # Load historical data
        df = self.load_historical_data(days)
        if df.empty:
            logger.error("Failed to load historical data")
            return {}, {}
        
        # Generate signals for both strategies
        original_signals = self.generate_signals_for_period(df, lookback_hours, use_sentiment=False)
        sentiment_signals = self.generate_signals_for_period(df, lookback_hours, use_sentiment=True)
        
        if original_signals.empty or sentiment_signals.empty:
            logger.error("Failed to generate signals")
            return {}, {}
        
        # Simulate trades for both strategies
        original_results = self.simulate_trades(df, original_signals, "Original")
        sentiment_results = self.simulate_trades(df, sentiment_signals, "Sentiment-Enhanced")
        
        if not original_results or not sentiment_results:
            logger.error("Failed to simulate trades")
            return {}, {}
        
        # Generate timestamped filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_filename = f"btc_sentiment_comparison_{days}d_{lookback_hours}h_{timestamp}.png"
        md_filename = f"btc_sentiment_comparison_{days}d_{lookback_hours}h_{timestamp}.md"
        
        # Generate outputs
        self.create_comparison_plot(original_results, sentiment_results, 
                                  str(results_dir / png_filename))
        self.create_comparison_report(original_results, sentiment_results,
                                    str(results_dir / md_filename))
        
        logger.info("Comparison backtest completed successfully")
        logger.info(f"Results saved to: {results_dir}")
        logger.info(f"Chart: {png_filename}")
        logger.info(f"Report: {md_filename}")
        
        return original_results, sentiment_results


def main():
    """Main function to run sentiment comparison backtest"""
    parser = argparse.ArgumentParser(description="BTC Sentiment-Enhanced Strategy Backtester")
    parser.add_argument('--days', type=int, default=3, 
                       help='Number of days to backtest (default: 3)')
    parser.add_argument('--lookback', type=int, default=24, 
                       help='Lookback hours for signal generation (default: 24)')
    parser.add_argument('--capital', type=float, default=10000.0, 
                       help='Initial capital (default: 10000)')
    parser.add_argument('--alpha', type=float, default=0.5, 
                       help='Sentiment multiplier weight (default: 0.5)')
    
    args = parser.parse_args()
    
    # Create backtester
    backtester = BTCSentimentBacktester(initial_capital=args.capital, alpha=args.alpha)
    
    # Run comparison backtest
    original_results, sentiment_results = backtester.run_comparison_backtest(
        days=args.days, lookback_hours=args.lookback)
    
    if original_results and sentiment_results:
        print(f"\n=== SENTIMENT COMPARISON RESULTS ===")
        print(f"Timeframe: {args.days} days, {args.lookback}h lookback")
        print(f"Sentiment Alpha: {args.alpha}")
        print(f"\nOriginal Strategy:")
        print(f"  Trades: {original_results['total_trades']}")
        print(f"  Win Rate: {original_results['win_rate']:.1f}%")
        print(f"  Return: {original_results['total_return']:.2%}")
        print(f"  Final Equity: ${original_results['final_equity']:,.0f}")
        print(f"\nSentiment-Enhanced Strategy:")
        print(f"  Trades: {sentiment_results['total_trades']}")
        print(f"  Win Rate: {sentiment_results['win_rate']:.1f}%")
        print(f"  Return: {sentiment_results['total_return']:.2%}")
        print(f"  Final Equity: ${sentiment_results['final_equity']:,.0f}")
        print(f"\nImprovement:")
        print(f"  Return Delta: {sentiment_results['total_return'] - original_results['total_return']:+.2%}")
        print(f"  Win Rate Delta: {sentiment_results['win_rate'] - original_results['win_rate']:+.1f}%")
        print(f"\nResults saved to: backtests/results/")
    else:
        print("Comparison backtest failed - check logs for details")


if __name__ == "__main__":
    main()
