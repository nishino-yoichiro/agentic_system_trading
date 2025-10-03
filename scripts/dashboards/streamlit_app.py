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


def show_signals(days: int, alpha: float) -> Dict[str, Any]:
    integration = CryptoSignalIntegration()
    signals = integration.generate_signals(["BTC"], days=days)
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


def show_backtest(days: int, initial_capital: float, step: int = 10) -> None:
    integration = CryptoSignalIntegration()
    results = integration.backtest_signals(["BTC"], days=days, initial_capital=initial_capital, step=step)
    st.subheader("Backtest (BTC)")
    if "error" in results:
        st.warning(results["error"])
        return
    cols = st.columns(6)
    cols[0].metric("Initial", f"${results['initial_capital']:,.0f}")
    cols[1].metric("Final", f"${results['final_capital']:,.0f}")
    cols[2].metric("Return", f"{results['total_return']*100:.1f}%")
    cols[3].metric("Sharpe", f"{results['sharpe_ratio']:.2f}")
    cols[4].metric("Max DD", f"{results['max_drawdown']*100:.1f}%")
    cols[5].metric("Trades", f"{results['total_trades']}")

    if results.get("equity_curve"):
        eq = pd.Series(results["equity_curve"]).reset_index(drop=True)
        st.line_chart(eq.rename("Equity"))


def main() -> None:
    st.title("BTC Dashboard (TradingView + Signals + Backtest)")
    with st.sidebar:
        st.header("Controls")
        days = st.slider("Days of data", 7, 120, 30, step=1)
        alpha = st.slider("Sentiment alpha", 0.0, 1.0, 0.5, step=0.05)
        initial_capital = st.number_input("Initial capital", min_value=1000, value=100000, step=1000)
        speed_vs_resolution = st.slider("Speed vs Resolution (higher = faster)", 1, 20, 10, step=1)
        run_bt = st.button("Run backtest")

    # Top: TradingView chart
    tradingview_widget(height=520)

    # Data status
    df = load_btc_data(days=days)
    st.caption(f"Loaded BTC data: {len(df)} rows")
    if df.empty:
        st.warning("No live BTC data available in app context. TradingView still shows live chart.")

    # Signals & sentiment
    show_signals(days=days, alpha=alpha)
    show_sentiment(days=days, alpha=alpha)

    # Backtest (on-demand)
    if run_bt:
        with st.spinner("Running backtest..."):
            show_backtest(days=days, initial_capital=initial_capital, step=speed_vs_resolution)


if __name__ == "__main__":
    main()


