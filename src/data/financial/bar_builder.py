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
    Aggregate trades into dollar bars using vectorized cumsum + groupby.

    Each bar closes when cumulative dollar volume (price × qty) exceeds
    `dollar_threshold`.  Memory-efficient — no Python-level row iteration.

    Returns DataFrame with columns:
      timestamp_open, timestamp_close, open, high, low, close,
      volume, dollar_volume, buy_volume, sell_volume, n_trades
    """
    trades = trades.sort_values(timestamp_col).reset_index(drop=True)

    price = trades[price_col].to_numpy(dtype=np.float64)
    qty   = trades[qty_col].to_numpy(dtype=np.float64)
    ts    = trades[timestamp_col].to_numpy()

    dollar = price * qty
    cum_dollar = np.cumsum(dollar)

    # Bar ID for each trade: integer division of cumulative dollar by threshold
    bar_id = (cum_dollar / dollar_threshold).astype(np.int64)

    # Build bar index arrays (first/last trade index per bar)
    _, first_idx = np.unique(bar_id, return_index=True)
    last_idx = np.append(first_idx[1:] - 1, len(bar_id) - 1)
    bar_lengths = np.diff(np.append(first_idx, len(bar_id)))

    # OHLCV via numpy reduceat — no DataFrame copy, O(N) time and memory
    open_  = price[first_idx]
    close_ = price[last_idx]
    high_  = np.maximum.reduceat(price, first_idx)
    low_   = np.minimum.reduceat(price, first_idx)
    vol_   = np.add.reduceat(qty,    first_idx)
    dvol_  = np.add.reduceat(dollar, first_idx)
    n_     = bar_lengths
    ts_open_  = ts[first_idx]
    ts_close_ = ts[last_idx]

    buy_vol  = np.zeros(len(open_))
    sell_vol = np.zeros(len(open_))
    if side_col and side_col in trades.columns:
        is_bm    = trades[side_col].to_numpy(dtype=bool)
        buy_qty  = np.where(~is_bm, qty, 0.0)
        sell_qty = np.where( is_bm, qty, 0.0)
        buy_vol  = np.add.reduceat(buy_qty,  first_idx)
        sell_vol = np.add.reduceat(sell_qty, first_idx)

    result = pd.DataFrame({
        "timestamp_open":  ts_open_,
        "timestamp_close": ts_close_,
        "open":            open_,
        "high":            high_,
        "low":             low_,
        "close":           close_,
        "volume":          vol_,
        "dollar_volume":   dvol_,
        "buy_volume":      buy_vol,
        "sell_volume":     sell_vol,
        "n_trades":        n_,
    })

    logger.info(
        f"Built {len(result)} dollar bars "
        f"(threshold=${dollar_threshold:,.0f}, {len(trades):,} trades in)"
    )
    return result


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
    dollar = trades[price_col].to_numpy(dtype=np.float64) * trades[qty_col].to_numpy(dtype=np.float64)
    dates  = pd.to_datetime(trades[timestamp_col], unit="ms").dt.date.to_numpy()
    unique_dates, date_idx = np.unique(dates, return_inverse=True)
    daily_sums = np.bincount(date_idx, weights=dollar)
    daily_dollar = pd.Series(daily_sums, index=unique_dates)
    median_daily = daily_dollar.median()
    threshold = median_daily / bars_per_day
    logger.info(
        f"Auto dollar threshold: ${threshold:,.0f} "
        f"(target {bars_per_day} bars/day, median daily vol ${median_daily:,.0f})"
    )
    return float(threshold)
