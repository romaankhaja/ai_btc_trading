"""
Liquidity Features - Phase 3 Feature Set.

Builds the non-derivatives liquidity proxies used by the pipeline.
Derivative inputs are intentionally left out until live ingestion is
available end-to-end.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_liquidity_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute liquidity and imbalance proxies from klines.

    1. Volume Delta = taker_buy_volume - aggressive_sell_volume
    2. Trade Imbalance = Volume Delta / Total Volume
    3. Amihud Illiquidity = abs(Close - Open) / Volume
    4. Volatility/Liquidity Ratio = ATR_14 / Volume
    """
    df = df.copy()

    # 1 & 2: Imbalance proxies
    if "taker_buy_volume" in df.columns and "volume" in df.columns:
        buy_vol = df["taker_buy_volume"]
        sell_vol = df["volume"] - df["taker_buy_volume"]

        df["volume_delta"] = buy_vol - sell_vol
        df["trade_imbalance"] = df["volume_delta"] / df["volume"].replace(0, pd.NA)
        df["delta_velocity"] = df["volume_delta"] - df["volume_delta"].shift(1)
        df["aggressive_buy_ratio"] = buy_vol / df["volume"].replace(0, pd.NA)
        logger.info("  Computed imbalance proxies (volume_delta, trade_imbalance, delta_velocity)")
    else:
        logger.warning("Missing taker_buy_volume; imbalance proxies will be zero.")
        for col in ["volume_delta", "trade_imbalance", "delta_velocity", "aggressive_buy_ratio"]:
            df[col] = 0.0

    # 3 & 4: Spread / illiquidity proxies
    if "volume" in df.columns:
        safe_volume = df["volume"].replace(0, pd.NA)
        df["amihud_illiquidity"] = (df["close"] - df["open"]).abs() / safe_volume

        if "atr_14" in df.columns:
            df["volatility_liquidity_ratio"] = df["atr_14"] / safe_volume

        if "volume_ratio" in df.columns:
            df["volume_spike_score"] = df["volume_ratio"].rolling(200).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )

        if all(c in df.columns for c in ["trade_imbalance", "delta_velocity", "volatility_liquidity_ratio"]):
            vl_ratio = df["volatility_liquidity_ratio"].replace(0, np.nan)
            df["liquidity_pressure_score"] = (
                df["trade_imbalance"] * df["delta_velocity"] * (1.0 / vl_ratio)
            )
        logger.info("  Computed amihud_illiquidity and liquidity pressure score")

    return df


def build_liquidity_features(symbol: str = "BTCUSDT", base_tf: str = "15m") -> pd.DataFrame:
    """
    Loads multi-TF merged features and computes the liquidity proxies.
    Returns the enriched DataFrame.
    """
    project_root = Path(__file__).resolve().parent.parent
    features_dir = project_root / "data" / "features" / symbol
    base_path = features_dir / f"multi_tf_merged_{base_tf}.parquet"

    if not base_path.exists():
        logger.error(f"Multi-TF merged features not found at {base_path}")
        return None

    df = pd.read_parquet(base_path)
    logger.info(f"Loaded {len(df)} rows for liquidity feature computation.")

    df = compute_liquidity_proxies(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    merged_df = build_liquidity_features()
    if merged_df is not None:
        print(f"Total columns: {len(merged_df.columns)}")
        for col in ["volume_delta", "trade_imbalance", "amihud_illiquidity", "liquidity_pressure_score"]:
            status = "OK" if col in merged_df.columns else "MISSING"
            print(f"  {col}: {status}")
