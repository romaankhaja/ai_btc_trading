"""
Derivatives feature merge for the 15-minute master feature set.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DERIVATIVES_COLUMNS = [
    "funding_rate_raw",
    "funding_rate_zscore",
    "funding_rate_extreme",
    "oi_raw",
    "oi_momentum_4h",
    "oi_price_divergence",
    "long_short_ratio",
    "ls_ratio_zscore",
]


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    zscore = (series - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return zscore.replace([np.inf, -np.inf], np.nan)


def _load_derivatives_frames(base_dir: Path):
    funding_df = pd.read_parquet(base_dir / "funding_rate.parquet").copy()
    open_interest_df = pd.read_parquet(base_dir / "open_interest.parquet").copy()
    long_short_df = pd.read_parquet(base_dir / "long_short_ratio.parquet").copy()

    for frame in (funding_df, open_interest_df, long_short_df):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame.sort_values("timestamp", inplace=True)

    return funding_df, open_interest_df, long_short_df


def merge_derivatives_features(master_df: pd.DataFrame) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / "data" / "derivatives" / "BTCUSDT"
    funding_df, open_interest_df, long_short_df = _load_derivatives_frames(base_dir)

    merged = master_df.copy()
    merged["open_time"] = pd.to_datetime(merged["open_time"], utc=True)
    merged.sort_values("open_time", inplace=True)

    time_index = pd.DatetimeIndex(merged["open_time"])
    price_series = pd.Series(merged["close"].values, index=time_index, name="close")

    funding_15m = (
        funding_df.set_index("timestamp")[["fundingRate"]]
        .resample("15min")
        .last()
        .reindex(time_index)
        .ffill()
    )
    funding_15m.rename(columns={"fundingRate": "funding_rate_raw"}, inplace=True)

    oi_15m = (
        open_interest_df.set_index("timestamp")[["sumOpenInterestValue"]]
        .resample("15min")
        .last()
        .reindex(time_index)
        .ffill()
    )
    oi_15m.rename(columns={"sumOpenInterestValue": "oi_raw"}, inplace=True)

    ls_15m = (
        long_short_df.set_index("timestamp")[["longShortRatio"]]
        .resample("15min")
        .last()
        .reindex(time_index)
        .ffill()
    )
    ls_15m.rename(columns={"longShortRatio": "long_short_ratio"}, inplace=True)

    derivatives_df = pd.concat([funding_15m, oi_15m, ls_15m], axis=1)
    real_data_mask = derivatives_df.notna().any(axis=1)

    derivatives_df["funding_rate_zscore"] = _rolling_zscore(derivatives_df["funding_rate_raw"], window=2880)
    derivatives_df["funding_rate_extreme"] = (derivatives_df["funding_rate_zscore"].abs() > 2.0).astype(float)
    derivatives_df["oi_momentum_4h"] = derivatives_df["oi_raw"].pct_change(periods=16)

    price_up = price_series.diff(periods=16) > 0
    price_down = price_series.diff(periods=16) < 0
    oi_down = derivatives_df["oi_momentum_4h"] < 0
    oi_up = derivatives_df["oi_momentum_4h"] > 0
    derivatives_df["oi_price_divergence"] = 0.0
    derivatives_df.loc[price_up & oi_down, "oi_price_divergence"] = 1.0
    derivatives_df.loc[price_down & oi_up, "oi_price_divergence"] = -1.0

    derivatives_df["ls_ratio_zscore"] = _rolling_zscore(derivatives_df["long_short_ratio"], window=96)
    derivatives_df = derivatives_df[DERIVATIVES_COLUMNS]
    derivatives_df = derivatives_df.fillna(0.0)

    merged = merged.merge(
        derivatives_df.reset_index().rename(columns={"index": "open_time"}),
        on="open_time",
        how="left",
    )
    merged[DERIVATIVES_COLUMNS] = merged[DERIVATIVES_COLUMNS].fillna(0.0)

    real_rows = int(real_data_mask.sum())
    total_rows = len(real_data_mask)
    real_pct = (real_rows / total_rows * 100.0) if total_rows else 0.0
    zero_pct = 100.0 - real_pct
    real_start = derivatives_df.index[real_data_mask.argmax()] if real_rows > 0 else "no real data"

    print(f"Derivatives merge added {len(DERIVATIVES_COLUMNS)} new columns")
    print(f"Rows with real derivatives data: {real_pct:.2f}% | filled zeros: {zero_pct:.2f}%")
    print(f"Real derivatives data starts: {real_start}")

    return merged
