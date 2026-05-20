"""
fetch_derivatives_historical.py
================================
Fetches REAL historical derivatives data from Binance Futures REST API
and merges into the existing cleaned 15m Parquet file.

Endpoints used:
  - GET /fapi/v1/fundingRate          — 8-hour funding rate snapshots
  - GET /futures/data/openInterestHist — 15-min open interest history
  - GET /fapi/v1/markPriceKlines       — 15-min mark price OHLCV
  - GET /futures/data/globalLongShortAccountRatio — 15-min L/S ratio

All data is forward-filled to align with 15-minute candles.
No synthetic or simulated values are used.

Usage:
    python data_pipeline/collectors/fetch_derivatives_historical.py
    python data_pipeline/collectors/fetch_derivatives_historical.py --symbol BTCUSDT --tf 15m
"""

import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Binance Futures Base URLs
# ---------------------------------------------------------------------------
FAPI_BASE   = "https://fapi.binance.com"
DAPI_BASE   = "https://fapi.binance.com"   # USDT-M futures
DDATA_BASE  = "https://fapi.binance.com"

# Max rows per API call
FUNDING_LIMIT   = 1000
OI_LIMIT        = 500
MARK_LIMIT      = 1500
LS_LIMIT        = 500

RETRY_ATTEMPTS  = 3
RETRY_DELAY_S   = 2
REQUEST_TIMEOUT = 30

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AdaptiveRiskSystem/0.1"})


# ---------------------------------------------------------------------------
# Low-level paginated fetch helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict) -> list:
    """GET with retry; returns parsed JSON list."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            r = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.warning(f"  Attempt {attempt} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY_S * attempt)
    raise RuntimeError(f"Failed to fetch {url} after {RETRY_ATTEMPTS} attempts")


def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch full funding rate history from /fapi/v1/fundingRate.
    Returns DataFrame with columns: [timestamp_ms, fundingTime, fundingRate, markPrice].
    Funding is snapshotted at 00:00, 08:00, 16:00 UTC — forward-filled to 15m bars.
    """
    logger.info(f"  [Funding] Fetching {symbol} from {_ms_to_utc(start_ms)} to {_ms_to_utc(end_ms)}")
    records = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": FUNDING_LIMIT,
        }
        data = _get(f"{FAPI_BASE}/fapi/v1/fundingRate", params)
        if not data:
            break
        records.extend(data)
        # Advance past the last returned timestamp
        last_ts = int(data[-1]["fundingTime"])
        if last_ts <= current_start:
            break
        current_start = last_ts + 1
        if len(data) < FUNDING_LIMIT:
            break
        time.sleep(0.2)

    if not records:
        logger.warning("  [Funding] No records returned.")
        return pd.DataFrame(columns=["open_time", "funding_rate"])

    df = pd.DataFrame(records)
    df["open_time"] = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["open_time", "funding_rate"]].sort_values("open_time").reset_index(drop=True)
    logger.info(f"  [Funding] Retrieved {len(df)} records ({df['open_time'].min()} → {df['open_time'].max()})")
    return df


def fetch_open_interest(symbol: str, period: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch open interest history from /futures/data/openInterestHist.
    period: '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
    Returns DataFrame with columns: [open_time, open_interest_usd].
    """
    logger.info(f"  [OI] Fetching {symbol} {period} from {_ms_to_utc(start_ms)} to {_ms_to_utc(end_ms)}")
    records = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": current_start,
            "endTime": min(end_ms, current_start + OI_LIMIT * _period_ms(period)),
            "limit": OI_LIMIT,
        }
        data = _get(f"{FAPI_BASE}/futures/data/openInterestHist", params)
        if not data:
            break
        records.extend(data)
        last_ts = int(data[-1]["timestamp"])
        if last_ts <= current_start:
            break
        current_start = last_ts + _period_ms(period)
        if len(data) < OI_LIMIT:
            break
        time.sleep(0.2)

    if not records:
        logger.warning("  [OI] No records returned.")
        return pd.DataFrame(columns=["open_time", "open_interest_usd"])

    df = pd.DataFrame(records)
    df["open_time"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms", utc=True)
    df["open_interest_usd"] = df["sumOpenInterestValue"].astype(float)
    df = df[["open_time", "open_interest_usd"]].sort_values("open_time").reset_index(drop=True)
    logger.info(f"  [OI] Retrieved {len(df)} records ({df['open_time'].min()} → {df['open_time'].max()})")
    return df


def fetch_mark_price_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch mark price OHLCV from /fapi/v1/markPriceKlines.
    Returns DataFrame with columns: [open_time, mark_open, mark_high, mark_low, mark_close].
    """
    logger.info(f"  [Mark] Fetching {symbol} {interval} mark price klines")
    records = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": MARK_LIMIT,
        }
        data = _get(f"{FAPI_BASE}/fapi/v1/markPriceKlines", params)
        if not data:
            break
        records.extend(data)
        last_ts = int(data[-1][0])
        if last_ts <= current_start:
            break
        current_start = last_ts + _period_ms(interval)
        if len(data) < MARK_LIMIT:
            break
        time.sleep(0.2)

    if not records:
        logger.warning("  [Mark] No records returned.")
        return pd.DataFrame(columns=["open_time", "mark_open", "mark_high", "mark_low", "mark_close"])

    df = pd.DataFrame(records, columns=[
        "open_time_ms", "mark_open", "mark_high", "mark_low", "mark_close",
        "ignore1", "close_time_ms", "ignore2", "ignore3", "ignore4", "ignore5", "ignore6"
    ])
    df["open_time"] = pd.to_datetime(df["open_time_ms"].astype(np.int64), unit="ms", utc=True)
    for col in ["mark_open", "mark_high", "mark_low", "mark_close"]:
        df[col] = df[col].astype(float)
    df = df[["open_time", "mark_open", "mark_high", "mark_low", "mark_close"]].sort_values("open_time").reset_index(drop=True)
    logger.info(f"  [Mark] Retrieved {len(df)} records ({df['open_time'].min()} → {df['open_time'].max()})")
    return df


def fetch_long_short_ratio(symbol: str, period: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch global long/short account ratio from /futures/data/globalLongShortAccountRatio.
    Returns DataFrame with columns: [open_time, long_short_ratio].
    """
    logger.info(f"  [L/S] Fetching {symbol} {period} long/short ratio")
    records = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": current_start,
            "endTime": min(end_ms, current_start + LS_LIMIT * _period_ms(period)),
            "limit": LS_LIMIT,
        }
        data = _get(f"{FAPI_BASE}/futures/data/globalLongShortAccountRatio", params)
        if not data:
            break
        records.extend(data)
        last_ts = int(data[-1]["timestamp"])
        if last_ts <= current_start:
            break
        current_start = last_ts + _period_ms(period)
        if len(data) < LS_LIMIT:
            break
        time.sleep(0.2)

    if not records:
        logger.warning("  [L/S] No records returned. Returning NaN column.")
        return pd.DataFrame(columns=["open_time", "long_short_ratio"])

    df = pd.DataFrame(records)
    df["open_time"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms", utc=True)
    df["long_short_ratio"] = df["longShortRatio"].astype(float)
    df = df[["open_time", "long_short_ratio"]].sort_values("open_time").reset_index(drop=True)
    logger.info(f"  [L/S] Retrieved {len(df)} records ({df['open_time'].min()} → {df['open_time'].max()})")
    return df


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def merge_derivatives_into_parquet(
    symbol: str = "BTCUSDT",
    tf: str = "15m",
    project_root: Path = None
) -> pd.DataFrame:
    """
    Fetches all derivatives data for the date range covered by the existing
    cleaned parquet and merges it in by UTC timestamp using asof join.

    Columns added to the parquet:
      - funding_rate        (forward-filled from 8h snapshots)
      - open_interest_usd
      - mark_open, mark_high, mark_low, mark_close
      - long_short_ratio
      - futures_basis       (close - mark_close, computed here)

    Saves the enriched DataFrame back to the same parquet path.
    Returns the enriched DataFrame.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    parquet_path = project_root / "data" / "cleaned" / symbol / f"{tf}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Cleaned parquet not found: {parquet_path}")

    base_df = pd.read_parquet(parquet_path)
    base_df = base_df.sort_values("open_time").reset_index(drop=True)

    # Ensure open_time is UTC-aware
    if base_df["open_time"].dt.tz is None:
        base_df["open_time"] = base_df["open_time"].dt.tz_localize("UTC")

    start_ms = int(base_df["open_time"].min().timestamp() * 1000)
    end_ms   = int(base_df["open_time"].max().timestamp() * 1000) + _period_ms(tf)

    logger.info(f"Base parquet: {len(base_df)} rows from {base_df['open_time'].min()} to {base_df['open_time'].max()}")

    # ---- 1. Funding Rates ------------------------------------------------
    funding_df = fetch_funding_rates(symbol, start_ms, end_ms)

    # ---- 2. Open Interest ------------------------------------------------
    oi_df = fetch_open_interest(symbol, tf, start_ms, end_ms)

    # ---- 3. Mark Price Klines --------------------------------------------
    mark_df = fetch_mark_price_klines(symbol, tf, start_ms, end_ms)

    # ---- 4. Long/Short Ratio ---------------------------------------------
    ls_df = fetch_long_short_ratio(symbol, tf, start_ms, end_ms)

    # ---- Merge via asof (backward fill) ----------------------------------
    result = base_df.copy()

    # Funding rate — 8-hour snapshots, forward-fill to every 15m bar
    if not funding_df.empty:
        result = pd.merge_asof(
            result.sort_values("open_time"),
            funding_df.sort_values("open_time"),
            on="open_time",
            direction="backward"
        )
        # Forward-fill any leading NaN
        result["funding_rate"] = result["funding_rate"].ffill().fillna(0.0)
    else:
        result["funding_rate"] = 0.0

    # Open interest — same frequency as base, direct merge
    if not oi_df.empty:
        result = pd.merge_asof(
            result.sort_values("open_time"),
            oi_df.sort_values("open_time"),
            on="open_time",
            direction="backward"
        )
        result["open_interest_usd"] = result["open_interest_usd"].ffill().fillna(method="bfill")
    else:
        result["open_interest_usd"] = np.nan

    # Mark price klines
    if not mark_df.empty:
        result = pd.merge_asof(
            result.sort_values("open_time"),
            mark_df.sort_values("open_time"),
            on="open_time",
            direction="backward"
        )
        for col in ["mark_open", "mark_high", "mark_low", "mark_close"]:
            result[col] = result[col].ffill().fillna(result["close"])
    else:
        result["mark_open"] = result["open"]
        result["mark_high"] = result["high"]
        result["mark_low"]  = result["low"]
        result["mark_close"] = result["close"]

    # Long/Short ratio
    if not ls_df.empty:
        result = pd.merge_asof(
            result.sort_values("open_time"),
            ls_df.sort_values("open_time"),
            on="open_time",
            direction="backward"
        )
        result["long_short_ratio"] = result["long_short_ratio"].ffill().fillna(1.0)
    else:
        result["long_short_ratio"] = 1.0

    # Derived: futures basis (spot close vs mark price)
    result["futures_basis"] = result["close"] - result["mark_close"]

    result = result.sort_values("open_time").reset_index(drop=True)

    # Save enriched parquet
    result.to_parquet(parquet_path, engine="pyarrow", index=False)
    logger.info(f"Saved enriched parquet ({len(result)} rows, {len(result.columns)} cols) → {parquet_path}")

    # Print coverage summary
    _null_summary(result, ["funding_rate", "open_interest_usd", "mark_close", "long_short_ratio", "futures_basis"])

    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _period_ms(period: str) -> int:
    """Convert period string to milliseconds."""
    mapping = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000,
        "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
        "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
        "1d": 86_400_000,
    }
    return mapping.get(period, 900_000)


def _ms_to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _null_summary(df: pd.DataFrame, cols: list):
    """Print null count for key columns."""
    print("\n  Null coverage summary:")
    for col in cols:
        if col in df.columns:
            n_null = df[col].isna().sum()
            pct = n_null / len(df) * 100
            status = "OK" if n_null == 0 else f"WARN: {n_null} ({pct:.1f}%)"
            print(f"    {col:<30s}: {status}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Fetch real Binance derivatives history")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--tf",     default="15m")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    print(f"\n{'=' * 60}")
    print(f"  FETCHING REAL DERIVATIVES DATA ({args.symbol} {args.tf})")
    print(f"{'=' * 60}")
    print(f"  Endpoints:")
    print(f"    /fapi/v1/fundingRate")
    print(f"    /futures/data/openInterestHist (period={args.tf})")
    print(f"    /fapi/v1/markPriceKlines (interval={args.tf})")
    print(f"    /futures/data/globalLongShortAccountRatio (period={args.tf})")
    print(f"{'=' * 60}\n")

    df = merge_derivatives_into_parquet(args.symbol, args.tf, project_root)

    print(f"\n  Done. Final shape: {df.shape}")
    print(f"  New columns: funding_rate, open_interest_usd, mark_open/high/low/close, long_short_ratio, futures_basis")
