"""
Run Trading Log Once
Single execution of trading log update
"""

from live_trading_log import LiveTradingLog

def main():
    print("🚀 Running Trading Log Update")
    print("=" * 40)
    
    trading_log = LiveTradingLog()
    
    # Show current portfolio
    portfolio = trading_log.get_portfolio_summary()
    print(f"💰 Current Portfolio:")
    print(f"   Total Value: ${portfolio['total_value']:,.2f}")
    print(f"   Cash: ${portfolio['cash_balance']:,.2f}")
    print(f"   BTC Position: {portfolio['btc_position']:.6f} BTC")
    print(f"   Cumulative PnL: ${portfolio['cumulative_pnl']:,.2f}")
    print(f"   Last Signal: {portfolio['last_signal']}")
    print()
    
    # Generate and log signals
    print("📊 Generating signals...")
    trades = trading_log.generate_and_log_signals()
    
    if trades:
        print(f"✅ Executed {len(trades)} trades:")
        for trade in trades:
            print(f"   {trade.signal_type} @ ${trade.price:.2f} | PnL: ${trade.simulated_pnl:.2f}")
            print(f"   Reason: {trade.reason}")
    else:
        print("ℹ️  No new trades executed")
    
    print()
    
    # Show recent trades
    recent_trades = trading_log.get_recent_trades(5)
    if not recent_trades.empty:
        print("📈 Recent Trades:")
        for _, trade in recent_trades.iterrows():
            print(f"   {trade['timestamp'][:19]} | {trade['signal_type']} @ ${trade['price']:.2f} | PnL: ${trade['simulated_pnl']:.2f}")
    
    print()
    print("✅ Trading log update complete!")

if __name__ == "__main__":
    main()
