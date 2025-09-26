#!/usr/bin/env python3
"""
Simple BTC Backtesting Engine
Fixed version with proper position tracking.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from btc_analysis_engine import BTCAnalysisEngine
from btc_signal_generator import BTCSignalGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class SimpleBTCBacktester:
    """Simple backtesting engine for BTC trading strategies"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.analysis_engine = BTCAnalysisEngine()
        self.signal_generator = BTCSignalGenerator()
        
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
    
    def generate_signals_for_period(self, df: pd.DataFrame, lookback_hours: int = 24) -> pd.DataFrame:
        """Generate trading signals for each point in the historical data"""
        signals = []
        
        # Use rolling window approach - for each point, use previous lookback_hours of data
        for i in range(lookback_hours, len(df)):
            # Get data up to current point (exclusive)
            historical_data = df.iloc[:i].copy()
            
            if len(historical_data) < 100:  # Need minimum data for indicators
                continue
                
            try:
                # Generate signal using historical data
                indicators = self.analysis_engine.calculate_technical_indicators(historical_data)
                # Convert indicators dict to DataFrame for signal generator
                indicators_df = pd.DataFrame([indicators])
                signal = self.signal_generator.generate_btc_signals(historical_data, indicators_df)
                
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
        
        logger.info(f"Generated {len(signals_df)} trading signals")
        
        # Debug: Show signal distribution
        signal_counts = signals_df['action'].value_counts()
        logger.info(f"Signal distribution: {signal_counts.to_dict()}")
        
        return signals_df
    
    def simulate_trades(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> Dict:
        """Simulate trading based on signals with proper position tracking"""
        
        # Initialize tracking variables
        capital = self.initial_capital
        btc_position = 0.0  # BTC position in BTC units
        trades = []
        equity_curve = []
        
        # Track performance metrics
        total_trades = 0
        winning_trades = 0
        total_pnl = 0.0
        max_capital = self.initial_capital
        max_drawdown = 0.0
        
        logger.info("Starting trade simulation...")
        
        for i, (timestamp, signal) in enumerate(signals_df.iterrows()):
            current_price = signal['price']
            action = signal['action']
            confidence = signal.get('confidence', 0.5)
            
            # Only trade if confidence is above threshold and action is not HOLD
            if confidence < 0.3 or action == 'HOLD':
                continue
                
            # Find next available price (next minute)
            next_idx = df.index.get_loc(timestamp) + 1
            if next_idx >= len(df):
                continue
                
            next_price = df.iloc[next_idx]['close']
            
            # Execute trade based on signal
            if action in ['BUY', 'STRONG_BUY'] and btc_position == 0:
                # Buy with 50% of available capital
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
                logger.debug(f"BUY: {btc_position:.6f} BTC at ${next_price:.2f} for ${trade_amount:.2f}")
                
            elif action in ['SELL', 'STRONG_SELL'] and btc_position > 0:
                # Sell entire position
                sell_amount = btc_position * next_price
                capital += sell_amount
                
                # Calculate PnL for this trade
                buy_trade = trades[-1] if trades else None
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
                btc_position = 0  # Reset position
                logger.debug(f"SELL: {btc_position:.6f} BTC at ${next_price:.2f} for ${sell_amount:.2f}")
            
            # Calculate current equity (capital + position value)
            current_equity = capital + (btc_position * next_price)
            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'capital': capital,
                'position_value': btc_position * next_price
            })
            
            # Update max drawdown
            if current_equity > max_capital:
                max_capital = current_equity
            
            drawdown = (max_capital - current_equity) / max_capital
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Close any remaining position at the end
        if btc_position > 0:
            final_price = df.iloc[-1]['close']
            final_amount = btc_position * final_price
            capital += final_amount
            
            # Calculate final PnL
            if trades and trades[-1]['action'] == 'BUY':
                pnl = final_amount - trades[-1]['amount']
                total_pnl += pnl
                
                if pnl > 0:
                    winning_trades += 1
            
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
        
        # Debug: Check for unmatched trades
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        logger.info(f"Debug: {len(buy_trades)} buy trades, {len(sell_trades)} sell trades")
        
        # Calculate win rate based on completed trades (buy-sell pairs)
        completed_trades = min(len(buy_trades), len(sell_trades))
        win_rate = (winning_trades / completed_trades * 100) if completed_trades > 0 else 0
        
        results = {
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
        
        logger.info(f"Simulation complete: {total_trades} trades, {win_rate:.1f}% win rate, {total_return:.2%} return")
        return results
    
    def create_equity_curve_plot(self, results: Dict, output_path: str = "backtest_BTC_24h.png"):
        """Create equity curve visualization"""
        
        if not results['equity_curve']:
            logger.error("No equity curve data to plot")
            return
        
        # Convert to DataFrame for easier plotting
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Equity curve
        ax1.plot(equity_df['timestamp'], equity_df['equity'], 'b-', linewidth=2, label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title('BTC Trading Strategy - Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Drawdown chart
        max_equity = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - max_equity) / max_equity * 100
        
        ax2.fill_between(equity_df['timestamp'], drawdown, 0, color='red', alpha=0.3, label='Drawdown')
        ax2.plot(equity_df['timestamp'], drawdown, 'r-', linewidth=1)
        ax2.set_title('Drawdown (%)', fontsize=12)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add performance metrics as text
        metrics_text = f"""Performance Metrics:
Trades: {results['total_trades']}
Win Rate: {results['win_rate']:.1f}%
Total Return: {results['total_return']:.2%}
Max Drawdown: {results['max_drawdown']:.2%}
Final Equity: ${results['final_equity']:,.0f}"""
        
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Equity curve saved to {output_path}")
    
    def create_summary_report(self, results: Dict, output_path: str = "backtest_summary_BTC.md"):
        """Create markdown summary report"""
        
        summary = f"""# BTC Backtest Summary

## Performance Metrics
- **Trades**: {results['total_trades']}
- **PnL**: ${results['total_pnl']:,.2f}
- **Max Drawdown**: {results['max_drawdown']:.2%}

## Detailed Results
- **Win Rate**: {results['win_rate']:.1f}%
- **Total Return**: {results['total_return']:.2%}
- **Final Equity**: ${results['final_equity']:,.0f}
- **Initial Capital**: ${self.initial_capital:,.0f}

## Strategy Performance
The backtest shows how the BTC trading strategy would have performed over the test period.
"""
        
        with open(output_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"Summary report saved to {output_path}")
    
    def run_backtest(self, days: int = 3, lookback_hours: int = 24) -> Dict:
        """Run complete backtest and generate reports"""
        
        logger.info(f"Starting BTC backtest for {days} days with {lookback_hours}h lookback")
        
        # Load historical data
        df = self.load_historical_data(days)
        if df.empty:
            logger.error("Failed to load historical data")
            return {}
        
        # Generate signals
        signals_df = self.generate_signals_for_period(df, lookback_hours)
        if signals_df.empty:
            logger.error("Failed to generate signals")
            return {}
        
        # Simulate trades
        results = self.simulate_trades(df, signals_df)
        if not results:
            logger.error("Failed to simulate trades")
            return {}
        
        # Generate outputs
        self.create_equity_curve_plot(results)
        self.create_summary_report(results)
        
        logger.info("Backtest completed successfully")
        return results

def main():
    """Main function to run backtest"""
    
    # Create backtester
    backtester = SimpleBTCBacktester(initial_capital=10000.0)
    
    # Run backtest for 3 days (72 hours)
    results = backtester.run_backtest(days=3, lookback_hours=24)
    
    if results:
        print(f"\n=== BACKTEST RESULTS ===")
        print(f"Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Final Equity: ${results['final_equity']:,.0f}")
        print(f"\nFiles generated:")
        print(f"- backtest_BTC_24h.png")
        print(f"- backtest_summary_BTC.md")
    else:
        print("Backtest failed - check logs for details")

if __name__ == "__main__":
    main()
