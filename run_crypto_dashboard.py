#!/usr/bin/env python3
"""
Crypto Dashboard Runner
Simple script to run the unified multi-symbol crypto dashboard
"""

import sys
from crypto_dashboard import CryptoDashboard
from loguru import logger

def main():
    """Run the crypto dashboard with specified symbols"""
    
    # Default symbols
    default_symbols = ['BTC', 'ETH', 'ADA', 'SOL']
    
    # Get symbols from command line or use default
    if len(sys.argv) > 1:
        symbols = [s.upper().strip() for s in sys.argv[1:]]
        logger.info(f"Using specified symbols: {symbols}")
    else:
        symbols = default_symbols
        logger.info(f"Using default symbols: {symbols}")
    
    # Validate symbols
    available_symbols = ['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI']
    invalid_symbols = [s for s in symbols if s not in available_symbols]
    
    if invalid_symbols:
        logger.error(f"Invalid symbols: {invalid_symbols}")
        logger.info(f"Available symbols: {available_symbols}")
        return
    
    try:
        # Create and run dashboard
        dashboard = CryptoDashboard(symbols=symbols)
        dashboard.run(host='localhost', port=8080, debug=False)
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")

if __name__ == "__main__":
    main()
