"""
Fetch Binance perpetuals derivatives history and cache it as parquet.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
REQUEST_TIMEOUT = 30
REQUEST_SLEEP_SECONDS = 0.5
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "derivatives" / SYMBOL

FUNDING_ENDPOINT = "/fapi/v1/fundingRate"
FUNDING_LIMIT = 1000

OPEN_INTEREST_ENDPOINT = "/futures/data/openInterestHist"
OPEN_INTEREST_LIMIT = 500
OPEN_INTEREST_PERIOD = "15m"

LONG_SHORT_ENDPOINT = "/futures/data/globalLongShortAccountRatio"
LONG_SHORT_LIMIT = 500
LONG_SHORT_PERIOD = "15m"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AI-Risk-Management/derivatives-fetcher"})


def _request_json(endpoint: str, params: dict) -> list:
    response = SESSION.get(
        f"{BASE_URL}{endpoint}",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected response for {endpoint}: {payload}")
    return payload


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("int64"), unit="ms", utc=True)


def _print_progress(total_rows: int, latest_timestamp: pd.Timestamp):
    print(f"Fetched {total_rows} rows, latest: {latest_timestamp}")


def _print_gap_summary(name: str, df: pd.DataFrame):
    if df.empty:
        print(f"{name}: no rows fetched")
        return

    timestamps = df["timestamp"].sort_values().reset_index(drop=True)
    gaps = timestamps.diff().dropna()
    large_gaps = gaps[gaps > pd.Timedelta(minutes=30)]
    start = timestamps.iloc[0]
    end = timestamps.iloc[-1]

    print(f"{name}: {len(df)} rows | {start} -> {end}")
    if large_gaps.empty:
        print(f"{name}: no gaps larger than 30 minutes")
    else:
        print(f"{name}: gaps larger than 30 minutes = {len(large_gaps)}")
        for gap_idx, gap in large_gaps.head(10).items():
            prev_ts = timestamps.iloc[gap_idx - 1] if gap_idx - 1 >= 0 else None
            curr_ts = timestamps.loc[gap_idx]
            print(f"  gap: {prev_ts} -> {curr_ts} ({gap})")


def fetch_funding_rate_history(symbol: str = SYMBOL) -> pd.DataFrame:
    rows = []
    start_time = None
    end_time = 9_999_999_999_999

    while True:
        params = {
            "symbol": symbol,
            "limit": FUNDING_LIMIT,
        }
        if start_time is None:
            params["endTime"] = end_time
        else:
            params["startTime"] = start_time

        payload = _request_json(FUNDING_ENDPOINT, params)
        if not payload:
            break

        frame = pd.DataFrame(payload)
        frame = frame[["fundingTime", "fundingRate", "markPrice"]].copy()
        frame["timestamp"] = _to_utc(frame["fundingTime"])
        frame["fundingRate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
        frame["markPrice"] = pd.to_numeric(frame["markPrice"], errors="coerce")
        frame = frame[["timestamp", "fundingRate", "markPrice"]]
        rows.append(frame)

        combined_rows = sum(len(chunk) for chunk in rows)
        latest_timestamp = frame["timestamp"].iloc[-1]
        _print_progress(combined_rows, latest_timestamp)

        if len(frame) < FUNDING_LIMIT:
            break

        start_time = int(payload[-1]["fundingTime"]) + 1
        time.sleep(REQUEST_SLEEP_SECONDS)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "fundingRate", "markPrice"])

    result = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return result.reset_index(drop=True)


def fetch_open_interest_history(symbol: str = SYMBOL) -> pd.DataFrame:
    rows = []
    end_time = 9_999_999_999_999
    step_ms = 15 * 60 * 1000

    while True:
        payload = _request_json(
            OPEN_INTEREST_ENDPOINT,
            {
                "symbol": symbol,
                "period": OPEN_INTEREST_PERIOD,
                "limit": OPEN_INTEREST_LIMIT,
                "endTime": end_time,
            },
        )
        if not payload:
            break

        frame = pd.DataFrame(payload)
        frame = frame[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]].copy()
        frame["timestamp"] = _to_utc(frame["timestamp"])
        frame["sumOpenInterest"] = pd.to_numeric(frame["sumOpenInterest"], errors="coerce")
        frame["sumOpenInterestValue"] = pd.to_numeric(frame["sumOpenInterestValue"], errors="coerce")
        rows.append(frame)

        combined_rows = sum(len(chunk) for chunk in rows)
        latest_timestamp = frame["timestamp"].iloc[-1]
        _print_progress(combined_rows, latest_timestamp)

        if len(frame) < OPEN_INTEREST_LIMIT:
            break

        first_raw_ts = int(payload[0]["timestamp"])
        end_time = first_raw_ts - step_ms
        time.sleep(REQUEST_SLEEP_SECONDS)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "sumOpenInterest", "sumOpenInterestValue"])

    result = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return result.reset_index(drop=True)


def fetch_long_short_ratio_history(symbol: str = SYMBOL) -> pd.DataFrame:
    rows = []
    end_time = 9_999_999_999_999
    step_ms = 15 * 60 * 1000

    while True:
        payload = _request_json(
            LONG_SHORT_ENDPOINT,
            {
                "symbol": symbol,
                "period": LONG_SHORT_PERIOD,
                "limit": LONG_SHORT_LIMIT,
                "endTime": end_time,
            },
        )
        if not payload:
            break

        frame = pd.DataFrame(payload)
        frame = frame[["timestamp", "longShortRatio", "longAccount", "shortAccount"]].copy()
        frame["timestamp"] = _to_utc(frame["timestamp"])
        frame["longShortRatio"] = pd.to_numeric(frame["longShortRatio"], errors="coerce")
        frame["longAccount"] = pd.to_numeric(frame["longAccount"], errors="coerce")
        frame["shortAccount"] = pd.to_numeric(frame["shortAccount"], errors="coerce")
        rows.append(frame)

        combined_rows = sum(len(chunk) for chunk in rows)
        latest_timestamp = frame["timestamp"].iloc[-1]
        _print_progress(combined_rows, latest_timestamp)

        if len(frame) < LONG_SHORT_LIMIT:
            break

        first_raw_ts = int(payload[0]["timestamp"])
        end_time = first_raw_ts - step_ms
        time.sleep(REQUEST_SLEEP_SECONDS)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "longShortRatio", "longAccount", "shortAccount"])

    result = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return result.reset_index(drop=True)


def fetch_all(symbol: str = SYMBOL):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Binance derivatives history for {symbol}...")

    funding_df = fetch_funding_rate_history(symbol=symbol)
    open_interest_df = fetch_open_interest_history(symbol=symbol)
    long_short_df = fetch_long_short_ratio_history(symbol=symbol)

    funding_path = OUTPUT_DIR / "funding_rate.parquet"
    open_interest_path = OUTPUT_DIR / "open_interest.parquet"
    long_short_path = OUTPUT_DIR / "long_short_ratio.parquet"

    funding_df.to_parquet(funding_path, index=False)
    open_interest_df.to_parquet(open_interest_path, index=False)
    long_short_df.to_parquet(long_short_path, index=False)

    print("\nFetch summary:")
    _print_gap_summary("funding_rate", funding_df)
    _print_gap_summary("open_interest", open_interest_df)
    _print_gap_summary("long_short_ratio", long_short_df)

    return {
        "funding_rate": funding_df,
        "open_interest": open_interest_df,
        "long_short_ratio": long_short_df,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fetch_all()
