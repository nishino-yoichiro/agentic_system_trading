#!/usr/bin/env python3
"""
Crypto Signal Runner
===================

Simple script to run the crypto signal framework with real data.

Usage:
    python run_crypto_signals.py [symbols] [--days DAYS] [--backtest] [--live]

Examples:
    python run_crypto_signals.py BTC ETH --days 30 --backtest
    python run_crypto_signals.py BTC --live
    python run_crypto_signals.py BTC ETH ADA SOL --days 60 --backtest
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from crypto_signal_integration import CryptoSignalIntegration

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('crypto_signals.log')
        ]
    )

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Crypto Signal Framework Runner')
    parser.add_argument('symbols', nargs='+', help='Crypto symbols to analyze (e.g., BTC ETH)')
    parser.add_argument('--days', type=int, default=30, help='Number of days of data to use')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--live', action='store_true', help='Generate live signals')
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital for backtest')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize integration
    try:
        integration = CryptoSignalIntegration()
        logger.info(f"Initialized crypto signal integration for symbols: {args.symbols}")
    except Exception as e:
        logger.error(f"Failed to initialize integration: {e}")
        return 1
    
    # Generate live signals
    if args.live or not args.backtest:
        logger.info("Generating live signals...")
        try:
            live_signals = integration.get_live_signals(args.symbols)
            
            print(f"\n{'='*50}")
            print(f"LIVE CRYPTO SIGNALS")
            print(f"{'='*50}")
            print(f"Timestamp: {live_signals['timestamp']}")
            print(f"Total Signals: {live_signals['total_signals']}")
            print(f"Symbols: {', '.join(args.symbols)}")
            
            if live_signals['signals']:
                for symbol, signals in live_signals['signals'].items():
                    print(f"\n{symbol} ({live_signals['current_prices'].get(symbol, 'N/A')}):")
                    for signal in signals:
                        print(f"  📊 {signal['strategy']}")
                        print(f"     Signal: {signal['signal_type']}")
                        print(f"     Entry: ${signal['entry_price']:.2f}")
                        print(f"     Confidence: {signal['confidence']:.1%}")
                        print(f"     Risk Size: {signal['risk_size']:.1%}")
                        if signal['stop_loss']:
                            print(f"     Stop Loss: ${signal['stop_loss']:.2f}")
                        if signal['take_profit']:
                            print(f"     Take Profit: ${signal['take_profit']:.2f}")
                        print(f"     Reason: {signal['reason']}")
                        print()
            else:
                print("No signals generated for current market conditions.")
                
        except Exception as e:
            logger.error(f"Error generating live signals: {e}")
            return 1
    
    # Run backtest
    if args.backtest:
        logger.info(f"Running backtest for {args.days} days...")
        try:
            backtest_results = integration.backtest_signals(
                args.symbols, 
                days=args.days, 
                initial_capital=args.initial_capital
            )
            
            if 'error' in backtest_results:
                print(f"Backtest Error: {backtest_results['error']}")
                return 1
            
            print(f"\n{'='*50}")
            print(f"BACKTEST RESULTS")
            print(f"{'='*50}")
            print(f"Period: {args.days} days")
            print(f"Initial Capital: ${backtest_results['initial_capital']:,.2f}")
            print(f"Final Capital: ${backtest_results['final_capital']:,.2f}")
            print(f"Total Return: {backtest_results['total_return']:.2%}")
            print(f"Annualized Return: {backtest_results['annualized_return']:.2%}")
            print(f"Volatility: {backtest_results['volatility']:.2%}")
            print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
            print(f"Win Rate: {backtest_results['win_rate']:.2%}")
            print(f"Total Trades: {backtest_results['total_trades']}")
            
            if backtest_results['strategy_performance']:
                print(f"\n{'='*30}")
                print(f"STRATEGY PERFORMANCE")
                print(f"{'='*30}")
                for strategy, perf in backtest_results['strategy_performance'].items():
                    print(f"{strategy}:")
                    print(f"  Total P&L: ${perf['total_pnl']:,.2f}")
                    print(f"  Win Rate: {perf['win_rate']:.2%}")
                    print(f"  Total Trades: {perf['total_trades']}")
                    print()
            
            # Save results
            integration.save_signal_log()
            logger.info("Backtest completed and results saved")
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return 1
    
    # Get strategy summary
    try:
        strategy_summary = integration.get_strategy_summary()
        print(f"\n{'='*50}")
        print(f"STRATEGY SUMMARY")
        print(f"{'='*50}")
        for name, summary in strategy_summary.items():
            print(f"{name}:")
            print(f"  Symbol: {summary['symbol']}")
            print(f"  Mechanism: {summary['mechanism']}")
            print(f"  Horizon: {summary['horizon']}")
            print(f"  Session: {summary['session']}")
            print(f"  Sharpe: {summary['sharpe']:.2f}")
            print(f"  Max DD: {summary['max_drawdown']:.2%}")
            print(f"  Hit Rate: {summary['hit_rate']:.2%}")
            print(f"  Total Trades: {summary['total_trades']}")
            print()
            
    except Exception as e:
        logger.error(f"Error getting strategy summary: {e}")
    
    logger.info("Crypto signal analysis completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
