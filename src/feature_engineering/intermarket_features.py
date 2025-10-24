"""
Intermarket features: rolling correlations to BTC, rolling beta to BTC, and ETH/BTC ratio z-score.
Uses existing OHLCV parquet data in data/crypto_db.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def compute_log_returns(price_series: pd.Series) -> pd.Series:
    if price_series.isnull().all():
        return pd.Series(dtype=float, index=price_series.index)
    return np.log(price_series / price_series.shift(1)).replace([np.inf, -np.inf], np.nan)


def rolling_corr_to_btc(
    prices: pd.DataFrame,
    window: int = 60,
    price_col: str = "close",
) -> pd.Series:
    # Expect columns include the given price_col; index aligned across assets
    returns = prices[price_col].unstack("symbol").pipe(lambda df: df.apply(compute_log_returns)).dropna(how="all")
    if "BTC" not in returns.columns:
        return pd.Series(dtype=float)
    result = {}
    for symbol in returns.columns:
        if symbol == "BTC":
            continue
        result[symbol] = returns[symbol].rolling(window).corr(returns["BTC"])  # type: ignore
    # Return as multiindex Series: index time, name symbol
    if not result:
        return pd.Series(dtype=float)
    out = pd.concat(result, axis=1)
    out = out.stack().rename("corr_to_btc")
    return out


def rolling_beta_to_btc(
    prices: pd.DataFrame,
    window: int = 60,
    price_col: str = "close",
) -> pd.Series:
    returns = prices[price_col].unstack("symbol").pipe(lambda df: df.apply(compute_log_returns)).dropna(how="all")
    if "BTC" not in returns.columns:
        return pd.Series(dtype=float)
    var_btc = returns["BTC"].rolling(window).var()
    result = {}
    for symbol in returns.columns:
        if symbol == "BTC":
            continue
        cov = returns[symbol].rolling(window).cov(returns["BTC"])  # type: ignore
        beta = cov / var_btc
        result[symbol] = beta
    if not result:
        return pd.Series(dtype=float)
    out = pd.concat(result, axis=1)
    out = out.stack().rename("beta_to_btc")
    return out


def eth_btc_ratio_features(
    eth_prices: pd.Series,
    btc_prices: pd.Series,
    window: int = 1440,
) -> pd.DataFrame:
    ratio = (eth_prices / btc_prices).replace([np.inf, -np.inf], np.nan)
    mean = ratio.rolling(window).mean()
    std = ratio.rolling(window).std()
    z = (ratio - mean) / std
    return pd.DataFrame({
        "eth_btc_ratio": ratio,
        "eth_btc_ratio_z": z,
    })


def build_intermarket_features(
    price_panels: Dict[str, pd.DataFrame],
    corr_window_short: int = 30,
    corr_window_medium: int = 240,
    beta_window_medium: int = 240,
) -> pd.DataFrame:
    """
    Parameters
    - price_panels: mapping symbol -> DataFrame with at least column 'close' and DateTimeIndex (UTC)

    Returns wide DataFrame indexed by timestamp with columns:
      corr_to_btc_{symbol}_{win}, beta_to_btc_{symbol}_{win}, eth_btc_ratio, eth_btc_ratio_z
    """
    # Align close prices into a Panel-like DataFrame: index time, columns MultiIndex(levels: [price_col, symbol])
    aligned = []
    for symbol, df in price_panels.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        tmp = df[["close"]].copy()
        tmp.columns = pd.MultiIndex.from_product([["close"], [symbol]], names=["field", "symbol"])
        aligned.append(tmp)
    if not aligned:
        return pd.DataFrame()
    prices = pd.concat(aligned, axis=1).sort_index()

    # Correlations to BTC
    corr_short = rolling_corr_to_btc(prices, window=corr_window_short)
    corr_med = rolling_corr_to_btc(prices, window=corr_window_medium)
    corr_short = corr_short.unstack().add_prefix("corr_to_btc_").add_suffix(f"_{corr_window_short}") if not corr_short.empty else pd.DataFrame()
    corr_med = corr_med.unstack().add_prefix("corr_to_btc_").add_suffix(f"_{corr_window_medium}") if not corr_med.empty else pd.DataFrame()

    # Betas to BTC
    beta_med = rolling_beta_to_btc(prices, window=beta_window_medium)
    beta_med = beta_med.unstack().add_prefix("beta_to_btc_").add_suffix(f"_{beta_window_medium}") if not beta_med.empty else pd.DataFrame()

    # ETH/BTC ratio z-score (use available prices if both present)
    eth_btc_df = pd.DataFrame()
    close_panel = prices["close"]
    if set(["ETH", "BTC"]).issubset(close_panel.columns):
        eth_btc_df = eth_btc_ratio_features(close_panel["ETH"], close_panel["BTC"])  # type: ignore

    parts = [df for df in [corr_short, corr_med, beta_med, eth_btc_df] if df is not None and not df.empty]
    if not parts:
        return pd.DataFrame(index=prices.index)
    out = pd.concat(parts, axis=1).sort_index()
    out = out.loc[~out.index.duplicated()]
    return out



