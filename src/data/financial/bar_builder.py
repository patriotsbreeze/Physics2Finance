"""
Dollar-bar and volume-bar resampling for financial tick data.

Per the project plan: "volume-bars or dollar-bars rather than tick-bars to
approximate continuous hydrodynamic flow."  Dollar bars sample uniformly in
dollar-volume space rather than clock time, reducing microstructure noise and
producing more stationary features — closer to the continuous fluid fields the
pre-trained ViT backbone was trained on.
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


def build_dollar_bars(
    trades: pd.DataFrame,
    dollar_threshold: float,
    price_col: str = "price",
    qty_col: str = "qty",
    timestamp_col: str = "timestamp",
    side_col: Optional[str] = "is_buyer_maker",
) -> pd.DataFrame:
    """
    Aggregate trades into dollar bars.

    Each bar closes when cumulative dollar volume (price × qty) exceeds
    `dollar_threshold`.  Returns OHLCV + buy/sell volume split.

    Args:
        trades: DataFrame of individual trades, sorted by timestamp.
        dollar_threshold: dollar volume per bar (e.g., 1_000_000 for $1M bars).
        price_col, qty_col, timestamp_col: column names.
        side_col: column indicating whether buyer was maker (Binance aggTrades).
                  None → no buy/sell split.

    Returns:
        DataFrame with columns:
          timestamp_open, timestamp_close, open, high, low, close,
          volume, dollar_volume, buy_volume, sell_volume, n_trades
    """
    trades = trades.sort_values(timestamp_col).reset_index(drop=True)

    bars = []
    bar_open = None
    bar_high = -np.inf
    bar_low = np.inf
    bar_vol = 0.0
    bar_dollar = 0.0
    bar_buy_vol = 0.0
    bar_sell_vol = 0.0
    bar_n = 0
    bar_ts_open = None

    for _, row in trades.iterrows():
        p = float(row[price_col])
        q = float(row[qty_col])
        ts = row[timestamp_col]
        dollar = p * q

        if bar_open is None:
            bar_open = p
            bar_ts_open = ts

        bar_high = max(bar_high, p)
        bar_low = min(bar_low, p)
        bar_close = p
        bar_vol += q
        bar_dollar += dollar
        bar_n += 1

        if side_col and side_col in row.index:
            # Binance: is_buyer_maker=True means the buyer was the passive side
            # → the aggressive order was a sell → this is a sell-initiated trade
            if row[side_col]:
                bar_sell_vol += q
            else:
                bar_buy_vol += q

        if bar_dollar >= dollar_threshold:
            bars.append({
                "timestamp_open":  bar_ts_open,
                "timestamp_close": ts,
                "open":            bar_open,
                "high":            bar_high,
                "low":             bar_low,
                "close":           bar_close,
                "volume":          bar_vol,
                "dollar_volume":   bar_dollar,
                "buy_volume":      bar_buy_vol,
                "sell_volume":     bar_sell_vol,
                "n_trades":        bar_n,
            })
            bar_open = None
            bar_high = -np.inf
            bar_low = np.inf
            bar_vol = 0.0
            bar_dollar = 0.0
            bar_buy_vol = 0.0
            bar_sell_vol = 0.0
            bar_n = 0
            bar_ts_open = None

    if bars:
        df = pd.DataFrame(bars)
        logger.info(
            f"Built {len(df)} dollar bars "
            f"(threshold=${dollar_threshold:,.0f}, "
            f"{len(trades)} trades in)"
        )
        return df
    return pd.DataFrame(columns=[
        "timestamp_open", "timestamp_close", "open", "high", "low", "close",
        "volume", "dollar_volume", "buy_volume", "sell_volume", "n_trades",
    ])


def build_volume_bars(
    trades: pd.DataFrame,
    volume_threshold: float,
    price_col: str = "price",
    qty_col: str = "qty",
    timestamp_col: str = "timestamp",
    side_col: Optional[str] = "is_buyer_maker",
) -> pd.DataFrame:
    """
    Aggregate trades into volume bars (same as dollar bars but threshold on qty).
    """
    trades = trades.sort_values(timestamp_col).reset_index(drop=True)

    bars = []
    bar_open = None
    bar_high = -np.inf
    bar_low = np.inf
    bar_vol = 0.0
    bar_dollar = 0.0
    bar_buy_vol = 0.0
    bar_sell_vol = 0.0
    bar_n = 0
    bar_ts_open = None

    for _, row in trades.iterrows():
        p = float(row[price_col])
        q = float(row[qty_col])
        ts = row[timestamp_col]

        if bar_open is None:
            bar_open = p
            bar_ts_open = ts

        bar_high = max(bar_high, p)
        bar_low = min(bar_low, p)
        bar_close = p
        bar_vol += q
        bar_dollar += p * q
        bar_n += 1

        if side_col and side_col in row.index:
            if row[side_col]:
                bar_sell_vol += q
            else:
                bar_buy_vol += q

        if bar_vol >= volume_threshold:
            bars.append({
                "timestamp_open":  bar_ts_open,
                "timestamp_close": ts,
                "open":            bar_open,
                "high":            bar_high,
                "low":             bar_low,
                "close":           bar_close,
                "volume":          bar_vol,
                "dollar_volume":   bar_dollar,
                "buy_volume":      bar_buy_vol,
                "sell_volume":     bar_sell_vol,
                "n_trades":        bar_n,
            })
            bar_open = None
            bar_high = -np.inf
            bar_low = np.inf
            bar_vol = 0.0
            bar_dollar = 0.0
            bar_buy_vol = 0.0
            bar_sell_vol = 0.0
            bar_n = 0
            bar_ts_open = None

    if bars:
        return pd.DataFrame(bars)
    return pd.DataFrame(columns=[
        "timestamp_open", "timestamp_close", "open", "high", "low", "close",
        "volume", "dollar_volume", "buy_volume", "sell_volume", "n_trades",
    ])


def compute_dollar_threshold(
    trades: pd.DataFrame,
    bars_per_day: int = 50,
    price_col: str = "price",
    qty_col: str = "qty",
    timestamp_col: str = "timestamp",
) -> float:
    """
    Auto-compute dollar threshold to produce approximately `bars_per_day` bars.
    Uses the median daily dollar volume divided by target bar count.
    """
    trades = trades.copy()
    trades["dollar"] = trades[price_col].astype(float) * trades[qty_col].astype(float)
    trades["date"] = pd.to_datetime(trades[timestamp_col], unit="ms").dt.date
    daily_dollar = trades.groupby("date")["dollar"].sum()
    median_daily = daily_dollar.median()
    threshold = median_daily / bars_per_day
    logger.info(
        f"Auto dollar threshold: ${threshold:,.0f} "
        f"(target {bars_per_day} bars/day, median daily vol ${median_daily:,.0f})"
    )
    return float(threshold)
