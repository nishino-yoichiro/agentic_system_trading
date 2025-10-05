"""
BTC-Only Streamlit App
======================

Features:
- TradingView BTC chart embed
- Live signal generation (BTC) using existing pipeline
- Sentiment-adjusted signals (if news available)
- Lightweight backtest over recent window

Deployment: Render compatible (see render.yaml)
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np
import streamlit as st
import requests
from dotenv import load_dotenv

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Local imports
from crypto_signal_integration import CryptoSignalIntegration
from crypto_analysis_engine import CryptoAnalysisEngine


st.set_page_config(page_title="BTC Dashboard", layout="wide")
load_dotenv()


@st.cache_data(show_spinner=False)
def fetch_btc_data_fallback(days: int = 7, interval_minutes: int = 60) -> pd.DataFrame:
    """Fallback data fetcher if local engine/parquet isn't available.
    Tries Coinbase Advanced if creds exist; otherwise returns empty DF.
    """
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")
    if not api_key or not api_secret:
        return pd.DataFrame()

    product_id = "BTC-USD"
    granularity = interval_minutes * 60
    end = int(time.time())
    start = end - days * 24 * 3600

    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles?granularity={granularity}&start={start}&end={end}"
    # Note: This public endpoint may be rate-limited/legacy. For Advanced API, a signed request is required.
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()
        # Coinbase returns [time, low, high, open, close, volume]
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.sort_values("time").reset_index(drop=True)
        df = df.set_index("time")
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_btc_data(days: int = 7) -> pd.DataFrame:
    engine = CryptoAnalysisEngine()
    try:
        df = engine.load_symbol_data("BTC", days=days)
        if df is not None and len(df) > 0:
            return df
    except Exception:
        pass
    return fetch_btc_data_fallback(days=days)


def tradingview_widget(height: int = 520) -> None:
    widget = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_btc"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%",
        "height": {height},
        "symbol": "COINBASE:BTCUSD",
        "interval": "60",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#222222",
        "hide_legend": false,
        "allow_symbol_change": false,
        "details": true,
        "studies": ["MASimple@tv-basicstudies"],
        "container_id": "tradingview_btc"
      }});
      </script>
    </div>
    """
    st.components.v1.html(widget, height=height + 30, scrolling=False)


def show_signals(days: int, alpha: float, selected_strategies: list = None) -> Dict[str, Any]:
    # Only run selected strategies, or default to just NY session if none selected
    strategies_to_run = selected_strategies if selected_strategies else ['btc_ny_session']
    integration = CryptoSignalIntegration(selected_strategies=strategies_to_run)
    signals = integration.generate_signals(["BTC"], days=days, strategies=strategies_to_run)
    st.subheader("Signals")
    if not signals:
        st.info("No signals generated for current market conditions.")
    else:
        for s in signals:
            st.write(f"- {s['strategy']}: {s['signal_type']} @ ${s['entry_price']:.2f} | conf={s['confidence']:.2f} | risk={s['risk_size']:.2f}")
            st.caption(s["reason"])
    return {"signals": signals}


def show_sentiment(days: int, alpha: float) -> None:
    st.subheader("Sentiment (impact on signals)")
    # Sentiment is integrated within CryptoSentimentGenerator used by dashboard.
    # Here, we show a simple note since detailed news APIs may not be configured on Render.
    st.write("Sentiment adjustment alpha:", alpha)
    st.caption("If news is not available, sentiment impact defaults to 0.")


def show_backtest(symbols: list, strategies: list, days: int, initial_capital: float, alpha: float, verbose: bool = False) -> None:
    """Run backtest using subprocess to capture terminal output"""
    st.subheader(f"Backtest Results - {', '.join(symbols)}")
    
    # Validate inputs
    if not symbols:
        st.error("❌ No symbols selected for backtesting.")
        return
    
    if not strategies:
        st.error("❌ No strategies selected for backtesting.")
        return
    
    if days < 1 or days > 365:
        st.error("❌ Days must be between 1 and 365.")
        return
    
    if initial_capital <= 0:
        st.error("❌ Initial capital must be positive.")
        return
    
    # Show parameters
    with st.expander("Backtest Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Symbols:** {', '.join(symbols)}")
            st.write(f"**Strategies:** {', '.join(strategies)}")
            st.write(f"**Days:** {days}")
        with col2:
            st.write(f"**Capital:** ${initial_capital:,.0f}")
            st.write(f"**Alpha:** {alpha}")
            st.write(f"**Verbose:** {verbose}")
    
    try:
        import subprocess
        import sys
        import os
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        output_container = st.empty()
        
        status_text.text("Preparing backtest command...")
        progress_bar.progress(10)
        
        # Build command to run backtest using the virtual environment
        venv_python = os.path.join(os.getcwd(), "enh_venv", "Scripts", "python.exe")
        if not os.path.exists(venv_python):
            # Fallback to system python
            venv_python = sys.executable
        
        # Check if backtest script exists
        backtest_script = "scripts/backtesting/multi_symbol_backtester.py"
        if not os.path.exists(backtest_script):
            st.error(f"❌ Backtest script not found at: {backtest_script}")
            return
        
        cmd = [
            venv_python, 
            backtest_script,
            "--symbols"] + symbols + [
            "--days", str(days),
            "--capital", str(initial_capital),
            "--alpha", str(alpha)
        ]
        
        if strategies:
            cmd.extend(["--strategies"] + strategies)
        
        if verbose:
            cmd.append("--verbose")
        
        status_text.text("Running backtest...")
        progress_bar.progress(30)
        
        # Run backtest as subprocess
        with st.spinner("Running backtest (this may take a moment)..."):
            # Add debugging info
            with st.expander("Debug Information", expanded=False):
                st.write(f"**Command:** `{' '.join(cmd)}`")
                st.write(f"**Python executable:** {venv_python}")
                st.write(f"**Working directory:** {os.getcwd()}")
                st.write(f"**Script exists:** {os.path.exists(backtest_script)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=300  # 5 minute timeout
            )
        
        progress_bar.progress(80)
        status_text.text("Processing results...")
        
        # Display output
        if result.stdout:
            with output_container.container():
                st.text("Backtest Output:")
                st.code(result.stdout)
        
        if result.stderr:
            st.warning("Backtest Warnings/Errors:")
            st.code(result.stderr)
        
        if result.returncode != 0:
            st.error(f"❌ Backtest failed with return code {result.returncode}")
            if result.stderr:
                st.code(result.stderr)
            
            # Provide helpful error messages
            if "No valid symbols found" in result.stderr:
                st.error("💡 **Tip:** Make sure you have data files for the selected symbols in the data/crypto_db directory.")
            elif "ModuleNotFoundError" in result.stderr:
                st.error("💡 **Tip:** Make sure all required Python packages are installed in the virtual environment.")
            elif "FileNotFoundError" in result.stderr:
                st.error("💡 **Tip:** Make sure the backtest script and data files exist.")
            
            return
        
        progress_bar.progress(100)
        status_text.text("Backtest completed!")
        st.success("✅ Backtest completed successfully!")
        
        # Show signals as part of backtest results
        st.subheader("Live Signals")
        try:
            # Generate signals for display
            integration = CryptoSignalIntegration(selected_strategies=strategies)
            signals = integration.generate_signals(symbols, days=days, strategies=strategies)
            if signals:
                for s in signals:
                    st.write(f"- {s['strategy']}: {s['signal_type']} @ ${s['entry_price']:.2f} | conf={s['confidence']:.2f} | risk={s['risk_size']:.2f}")
                    st.caption(s["reason"])
            else:
                st.info("No signals generated for current market conditions.")
        except Exception as e:
            st.warning(f"Could not generate signals for display: {e}")
        
        # Try to parse and display results from output
        if result.stdout:
            # Look for summary information in the output
            lines = result.stdout.split('\n')
            summary_section = False
            results_data = []
            
            for line in lines:
                if "BACKTEST SUMMARY" in line:
                    summary_section = True
                    continue
                elif summary_section and line.strip():
                    if ":" in line and not line.startswith("="):
                        # Parse result line like "BTC: Return: 5.2%"
                        parts = line.split(":")
                        if len(parts) >= 2:
                            symbol = parts[0].strip()
                            metrics = ":".join(parts[1:]).strip()
                            results_data.append({"Symbol": symbol, "Metrics": metrics})
            
            if results_data:
                st.write("**Backtest Summary:**")
                for result_item in results_data:
                    st.write(f"**{result_item['Symbol']}:** {result_item['Metrics']}")
            else:
                st.info("Backtest completed. Check the output above for detailed results.")
        
    except subprocess.TimeoutExpired:
        st.error("❌ Backtest timed out after 5 minutes. Try reducing the number of days or strategies.")
        st.info("💡 **Tip:** Consider reducing the number of days or symbols to speed up the backtest.")
    except FileNotFoundError as e:
        st.error(f"❌ Required file not found: {e}")
        st.info("💡 **Tip:** Make sure the virtual environment and backtest script are properly set up.")
    except Exception as e:
        st.error(f"❌ Backtest failed: {str(e)}")
        import traceback
        with st.expander("Full Error Details", expanded=False):
            st.code(traceback.format_exc())
        st.write("💡 **Tip:** Check that all required dependencies are installed and data files exist.")


def main() -> None:
    st.title("BTC Dashboard (TradingView + Signals + Backtest)")
    with st.sidebar:
        st.header("Controls")
        
        # Symbol selection
        available_symbols = ['BTC', 'ETH', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'SOL', 'UNI']
        selected_symbols = st.multiselect(
            "Select symbols to backtest", 
            available_symbols, 
            default=['BTC'],
            help="Choose which crypto symbols to include in the backtest"
        )
        
        # Strategy selection
        available_strategies = [
            'btc_asia_sweep', 'eth_breakout_continuation', 'btc_mean_reversion',
            'eth_funding_arb', 'btc_vol_compression', 'eth_basis_trade',
            'btc_ny_open_london_sweep', 'btc_ny_session'
        ]
        selected_strategies = st.multiselect(
            "Select strategies", 
            available_strategies, 
            default=available_strategies,
            help="Choose which trading strategies to use"
        )
        
        # Backtest parameters
        st.subheader("Backtest Settings")
        days = st.slider("Days of data", 7, 120, 30, step=1, help="Number of days of historical data to use for backtesting")
        alpha = st.slider("Sentiment alpha", 0.0, 1.0, 0.5, step=0.05, help="Sentiment adjustment factor (0 = no sentiment, 1 = full sentiment)")
        initial_capital = st.number_input("Initial capital", min_value=1000, value=100000, step=1000, help="Starting capital for backtesting")
        verbose = st.checkbox("Verbose output", value=False, help="Show detailed trade execution logs")
        
        # Add some spacing
        st.write("")
        
        # Backtest button with better styling
        run_bt = st.button("🚀 Run Backtest", type="primary", help="Click to start the backtest with selected parameters")

    # Top: TradingView chart
    tradingview_widget(height=520)

    # Data status
    df = load_btc_data(days=days)
    st.caption(f"Loaded BTC data: {len(df)} rows")
    if df.empty:
        st.warning("No live BTC data available in app context. TradingView still shows live chart.")

    # Signals & sentiment (only show if strategies are selected)
    st.subheader("Signals")
    if selected_strategies:
        st.info("Signals will be generated when you run the backtest. Select strategies and click 'Run Backtest' to see live signals.")
    else:
        st.info("Please select strategies above to see signals")
    
    show_sentiment(days=days, alpha=alpha)

    # Backtest (on-demand)
    if run_bt:
        if not selected_symbols:
            st.error("Please select at least one symbol to backtest.")
        elif not selected_strategies:
            st.error("Please select at least one strategy to use.")
        else:
            # Show backtest with all parameters
            show_backtest(
                symbols=selected_symbols, 
                strategies=selected_strategies, 
                days=days, 
                initial_capital=initial_capital,
                alpha=alpha,
                verbose=verbose
            )


if __name__ == "__main__":
    main()


