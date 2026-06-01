"""
Binance LOB dataset loader.

Downloads Binance aggTrades (monthly zip files from data.binance.vision),
applies dollar-bar resampling to approximate continuous hydrodynamic flow,
constructs synthetic LOB snapshots from trade data, and creates heatmap +
realized-volatility pairs for the linear probe.

Download path:
  https://data.binance.vision/data/spot/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{YYYY-MM}.zip

Saved as: data/financial/binance_btcusdt.parquet (dollar-bar OHLCV + LOB columns)

Usage:
  from src.data.financial.binance_loader import BinanceDataDownloader, BinanceDataset
  dl = BinanceDataDownloader("data/financial/binance")
  dl.download_agg_trades("BTCUSDT", "2023-01-01", "2023-03-31")
"""

import io
import zipfile
import tempfile
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger

from src.data.financial.lob_to_heatmap import LOBHeatmapEncoder
from src.data.financial.bar_builder import build_dollar_bars, compute_dollar_threshold


# ── Downloader ────────────────────────────────────────────────────────────────

class BinanceDataDownloader:
    """
    Downloads Binance aggTrades from the public data archive and saves as parquet.

    AggTrades columns (from Binance):
      a: aggregate trade ID
      p: price
      q: quantity
      f: first trade ID
      l: last trade ID
      T: timestamp (milliseconds)
      m: is_buyer_maker (True = sell-initiated trade)
      M: was_best_price_match (deprecated)
    """

    BASE_URL = "https://data.binance.vision/data/spot/monthly/aggTrades"

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_agg_trades(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        bars_per_day: int = 100,
    ) -> Path:
        """
        Download aggTrades for the given date range, apply dollar-bar resampling,
        and save as a single parquet file.

        Returns path to saved parquet.
        """
        symbol = symbol.upper()
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")

        months = []
        cur = start.replace(day=1)
        while cur <= end:
            months.append(cur)
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1)
            else:
                cur = cur.replace(month=cur.month + 1)

        all_trades = []
        for month in months:
            month_str = month.strftime("%Y-%m")
            url = f"{self.BASE_URL}/{symbol}/{symbol}-aggTrades-{month_str}.zip"
            logger.info(f"Downloading {url}")
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        df = pd.read_csv(
                            f,
                            names=["a", "p", "q", "f", "l", "T", "m", "M"],
                            dtype={"p": float, "q": float, "T": np.int64, "m": bool},
                        )
                df = df.rename(columns={
                    "p": "price", "q": "qty", "T": "timestamp", "m": "is_buyer_maker"
                })[["timestamp", "price", "qty", "is_buyer_maker"]]
                all_trades.append(df)
                logger.info(f"  {symbol} {month_str}: {len(df):,} trades")
            except Exception as e:
                logger.error(f"  Failed to download {symbol} {month_str}: {e}")

        if not all_trades:
            raise RuntimeError("No trade data downloaded.")

        # Estimate threshold from first month — avoids holding all months in RAM
        ts_start = int(start.timestamp() * 1000)
        ts_end   = int((end + timedelta(days=1)).timestamp() * 1000)
        threshold = compute_dollar_threshold(all_trades[0], bars_per_day=bars_per_day)

        # Process each month independently: build bars + LOB, then discard raw trades
        all_bars = []
        for trades_month in all_trades:
            trades_month = trades_month[
                (trades_month["timestamp"] >= ts_start) &
                (trades_month["timestamp"] < ts_end)
            ].copy()
            if len(trades_month) == 0:
                continue
            bars_month = build_dollar_bars(trades_month, dollar_threshold=threshold)
            bars_month = self._add_synthetic_lob(bars_month, trades_month, lob_levels=10)
            all_bars.append(bars_month)

        if not all_bars:
            raise RuntimeError("No bars produced after date filtering.")

        bars = pd.concat(all_bars, ignore_index=True)
        bars = bars.sort_values("timestamp_open").reset_index(drop=True)

        out_path = self.data_dir.parent / f"binance_{symbol.lower()}.parquet"
        bars.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(bars)} dollar bars → {out_path}")
        return out_path

    @staticmethod
    def _add_synthetic_lob(
        bars: pd.DataFrame,
        trades: pd.DataFrame,
        lob_levels: int = 10,
    ) -> pd.DataFrame:
        """
        Construct synthetic LOB from trades within each bar's time window.

        For each bar, we look at the trades that occurred during it, bin them
        into price buckets relative to mid-price, and assign buy vol → bid depth,
        sell vol → ask depth.  This approximates a snapshot LOB from flow data.
        """
        bars = bars.copy()

        for level in range(1, lob_levels + 1):
            bars[f"ask_price_{level}"] = np.nan
            bars[f"ask_vol_{level}"]   = 0.0
            bars[f"bid_price_{level}"] = np.nan
            bars[f"bid_vol_{level}"]   = 0.0

        for i, bar in bars.iterrows():
            ts_open  = bar["timestamp_open"]
            ts_close = bar["timestamp_close"]
            mid      = (bar["high"] + bar["low"]) / 2.0

            mask = (trades["timestamp"] >= ts_open) & (trades["timestamp"] <= ts_close)
            bar_trades = trades[mask]

            if len(bar_trades) == 0:
                continue

            # Separate buy and sell trades
            sells = bar_trades[bar_trades["is_buyer_maker"] == True]  # noqa: E712
            buys  = bar_trades[bar_trades["is_buyer_maker"] == False]

            tick = mid * 0.0001   # 1 bps tick size

            for level in range(1, lob_levels + 1):
                ask_price = mid + level * tick
                bid_price = mid - level * tick

                ask_mask = (
                    (sells["price"] >= ask_price - tick / 2) &
                    (sells["price"] <  ask_price + tick / 2)
                )
                bid_mask = (
                    (buys["price"] >= bid_price - tick / 2) &
                    (buys["price"] <  bid_price + tick / 2)
                )

                bars.at[i, f"ask_price_{level}"] = ask_price
                bars.at[i, f"ask_vol_{level}"]   = float(sells[ask_mask]["qty"].sum())
                bars.at[i, f"bid_price_{level}"] = bid_price
                bars.at[i, f"bid_vol_{level}"]   = float(buys[bid_mask]["qty"].sum())

        return bars


# ── Dataset ───────────────────────────────────────────────────────────────────

class BinanceDataset(Dataset):
    """
    Sliding-window dataset over Binance dollar-bar LOB snapshots.

    Reads the parquet file produced by BinanceDataDownloader.
    Returns the same dict format as FI2010Dataset for compatibility.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        window_size: int = 100,
        horizons: List[int] = (10, 50, 100, 500),
        lob_levels: int = 10,
        img_size: int = 224,
        stride: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.data_path = Path(data_path)
        self.split = split
        self.window_size = window_size
        self.horizons = list(horizons)
        self.lob_levels = lob_levels
        self.img_size = img_size
        self.stride = stride

        self.encoder = LOBHeatmapEncoder(lob_levels=lob_levels, img_size=img_size)

        self._lob_matrix: Optional[np.ndarray] = None
        self._mid_prices: Optional[np.ndarray] = None
        self._log_returns: Optional[np.ndarray] = None
        self._indices: List[int] = []

        self._load(train_ratio, val_ratio)

    def _load(self, train_ratio: float, val_ratio: float):
        if not self.data_path.exists():
            logger.warning(
                f"Binance parquet not found: {self.data_path}. "
                "Run BinanceDataDownloader.download_agg_trades() first."
            )
            return

        bars = pd.read_parquet(self.data_path)
        n = len(bars)

        # Time-ordered split
        if self.split == "train":
            bars = bars.iloc[: int(train_ratio * n)]
        elif self.split == "val":
            s = int(train_ratio * n)
            e = int((train_ratio + val_ratio) * n)
            bars = bars.iloc[s:e]
        else:
            bars = bars.iloc[int((train_ratio + val_ratio) * n):]

        if len(bars) == 0:
            return

        # Build LOB matrix in FI-2010 interleaved format: [P_a1,V_a1,P_b1,V_b1,...]
        k = self.lob_levels
        lob_cols = []
        for level in range(1, k + 1):
            ap_col = f"ask_price_{level}"
            av_col = f"ask_vol_{level}"
            bp_col = f"bid_price_{level}"
            bv_col = f"bid_vol_{level}"
            for col in [ap_col, av_col, bp_col, bv_col]:
                if col not in bars.columns:
                    bars[col] = 0.0
            lob_cols += [ap_col, av_col, bp_col, bv_col]

        self._lob_matrix = bars[lob_cols].values.astype(np.float32)

        # Mid-price from best ask / best bid
        if "ask_price_1" in bars.columns and "bid_price_1" in bars.columns:
            self._mid_prices = (
                bars["ask_price_1"].values + bars["bid_price_1"].values
            ) / 2.0
        else:
            self._mid_prices = bars["close"].values.astype(np.float32)

        mp = self._mid_prices.copy()
        mp = np.where(mp <= 0, np.nan, mp)
        self._log_returns = np.concatenate([[0.0], np.diff(np.log(mp))])
        self._log_returns = np.nan_to_num(self._log_returns, nan=0.0).astype(np.float32)

        max_h = max(self.horizons) if self.horizons else 0
        n = len(self._lob_matrix)
        self._indices = list(range(self.window_size, n - max_h, self.stride))

        logger.info(
            f"BinanceDataset [{self.split}]: {len(self._indices)} windows "
            f"from {n} bars (window={self.window_size}, horizons={self.horizons})"
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        t = self._indices[idx]
        window = self._lob_matrix[t - self.window_size: t]

        image = self.encoder.encode_matrix(window)

        targets = {}
        for h in self.horizons:
            if t + h <= len(self._log_returns):
                rv = float(np.sum(self._log_returns[t: t + h] ** 2))
                targets[h] = torch.tensor(rv, dtype=torch.float32)

        return {
            "image":     image,
            "targets":   targets,
            "mid_price": float(self._mid_prices[t]),
        }

    def get_log_returns(self) -> np.ndarray:
        if self._log_returns is None:
            return np.array([])
        return self._log_returns.copy()
