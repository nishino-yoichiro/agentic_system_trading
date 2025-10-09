#!/usr/bin/env python3
"""
Multi-Symbol Backtest Runner
Simple wrapper for running multi-symbol backtests
"""

import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

def main():
    """Run multi-symbol backtest with command line arguments"""
    
    # Available symbols
    available_symbols = ['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI']
    
    print("🚀 Multi-Symbol Crypto Backtester")
    print("=" * 50)
    print(f"Available symbols: {', '.join(available_symbols)}")
    print()
    
    # Get user input
    symbols_input = input("Enter symbols to backtest (space-separated, default: all): ").strip()
    if not symbols_input:
        symbols = available_symbols  # Default to all symbols
    else:
        symbols = [s.upper().strip() for s in symbols_input.split()]
    
    # Validate symbols
    invalid_symbols = [s for s in symbols if s not in available_symbols]
    if invalid_symbols:
        print(f"Invalid symbols: {', '.join(invalid_symbols)}")
        print(f"Available: {', '.join(available_symbols)}")
        return
    
    # Get strategy selection - dynamically load from strategies
    from crypto_trading_strategies import CryptoTradingStrategies
    strategies_obj = CryptoTradingStrategies()
    available_strategies = list(strategies_obj.strategies.keys())
    
    print(f"Available strategies: {', '.join(available_strategies)}")
    strategies_input = input("Enter strategies to use (space-separated, default: all): ").strip()
    
    if strategies_input:
        strategies = [s.strip() for s in strategies_input.split()]
        invalid_strategies = [s for s in strategies if s not in available_strategies]
        if invalid_strategies:
            print(f"Invalid strategies: {', '.join(invalid_strategies)}")
            print(f"Available: {', '.join(available_strategies)}")
            return
    else:
        strategies = available_strategies  # Default to all strategies
    
    # Get other parameters
    days = input("Days to backtest (default 30): ").strip()
    days = int(days) if days.isdigit() else 30
    
    capital = input("Initial capital (default 10000): ").strip()
    capital = float(capital) if capital.replace('.', '').isdigit() else 10000.0
    
    use_sentiment = input("Use sentiment-enhanced strategy? (y/n, default n): ").strip().lower() == 'y'
    
    run_portfolio = input("Run portfolio backtest? (y/n, default n): ").strip().lower() == 'y'
    
    verbose = input("Verbose output? (y/n, default n): ").strip().lower() == 'y'
    
    # Build command
    cmd = [
        sys.executable, str(Path(__file__).parent / 'multi_symbol_backtester.py'),
        '--symbols'] + symbols + [
        '--strategies'] + strategies + [
        '--days', str(days),
        '--capital', str(capital)
    ]
    
    if use_sentiment:
        cmd.append('--sentiment')
    
    if run_portfolio:
        cmd.append('--portfolio')
    
    if verbose:
        cmd.append('--verbose')
    
    print(f"\nRunning command: {' '.join(cmd)}")
    print("=" * 50)
    
    # Run the backtest
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Backtest completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Backtest failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\n⏹️ Backtest interrupted by user")

if __name__ == "__main__":
    main()
