"""
Trading Log Dashboard
Simple Streamlit dashboard to view live trading log
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import json

from live_trading_log import LiveTradingLog

st.set_page_config(
    page_title="Live Trading Log Dashboard",
    page_icon="📊",
    layout="wide"
)

def load_trading_data():
    """Load trading data from log"""
    trading_log = LiveTradingLog()
    
    # Get portfolio summary
    portfolio = trading_log.get_portfolio_summary()
    
    # Get recent trades
    recent_trades = trading_log.get_recent_trades(50)
    
    # Get daily summary for last 7 days
    daily_summaries = []
    for i in range(7):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        summary = trading_log.get_daily_summary(date)
        daily_summaries.append(summary)
    
    return portfolio, recent_trades, daily_summaries

def main():
    st.title("📊 Live Trading Log Dashboard")
    st.markdown("Real-time BTC trading signals and simulated PnL tracking")
    
    # Load data
    portfolio, recent_trades, daily_summaries = load_trading_data()
    
    # Portfolio Overview
    st.header("💰 Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Value",
            f"${portfolio['total_value']:,.2f}",
            delta=f"${portfolio['cumulative_pnl']:,.2f}"
        )
    
    with col2:
        st.metric(
            "Cash Balance",
            f"${portfolio['cash_balance']:,.2f}"
        )
    
    with col3:
        st.metric(
            "BTC Position",
            f"{portfolio['btc_position']:.6f} BTC"
        )
    
    with col4:
        st.metric(
            "Last Signal",
            portfolio['last_signal']
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Recent Trades")
        if not recent_trades.empty:
            # Create trade timeline
            fig = go.Figure()
            
            # Add buy signals
            buys = recent_trades[recent_trades['signal_type'] == 'BUY']
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys['timestamp'],
                    y=buys['price'],
                    mode='markers',
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    name='BUY',
                    text=buys['reason'],
                    hovertemplate='<b>BUY</b><br>Price: $%{y:.2f}<br>%{text}<extra></extra>'
                ))
            
            # Add sell signals
            sells = recent_trades[recent_trades['signal_type'] == 'SELL']
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells['timestamp'],
                    y=sells['price'],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    name='SELL',
                    text=sells['reason'],
                    hovertemplate='<b>SELL</b><br>Price: $%{y:.2f}<br>%{text}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Trade Signals Over Time",
                xaxis_title="Time",
                yaxis_title="BTC Price ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades recorded yet")
    
    with col2:
        st.subheader("📊 Daily PnL")
        if daily_summaries:
            # Create daily PnL chart
            df_daily = pd.DataFrame(daily_summaries)
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily = df_daily.sort_values('date')
            
            fig = px.bar(
                df_daily, 
                x='date', 
                y='pnl',
                title="Daily Simulated PnL",
                color='pnl',
                color_continuous_scale=['red', 'green']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily data available")
    
    # Recent Trades Table
    st.subheader("📋 Recent Trades")
    if not recent_trades.empty:
        # Format the dataframe for display
        display_df = recent_trades.copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['price'] = display_df['price'].round(2)
        display_df['simulated_pnl'] = display_df['simulated_pnl'].round(2)
        display_df['cumulative_pnl'] = display_df['cumulative_pnl'].round(2)
        
        st.dataframe(
            display_df[['timestamp', 'signal_type', 'price', 'simulated_pnl', 'cumulative_pnl', 'reason']],
            use_container_width=True
        )
    else:
        st.info("No trades recorded yet")
    
    # Daily Summary
    st.subheader("📅 Daily Summary (Last 7 Days)")
    if daily_summaries:
        df_summary = pd.DataFrame(daily_summaries)
        df_summary['date'] = pd.to_datetime(df_summary['date']).dt.strftime('%Y-%m-%d')
        
        st.dataframe(
            df_summary[['date', 'trades', 'pnl', 'signals']],
            use_container_width=True
        )
    
    # Auto-refresh
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    # Auto-refresh every 30 seconds
    st.markdown("---")
    st.markdown("🔄 Auto-refreshes every 30 seconds")

if __name__ == "__main__":
    main()
