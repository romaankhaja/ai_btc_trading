"""
Behavioral Labeler — Composite anomaly detection label.

Binary classification target:
  1 = Trader in emotionally compromised state (anomaly)
  0 = Normal trading behavior

Uses only current-row features (no future data).
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def label_behavioral(df, threshold=3):
    """
    Generate market-stress anomaly labels from current-candle inputs.

    Args:
        df: DataFrame with market-derived feature columns
        threshold: Number of concurrent stress conditions required

    Returns:
        numpy array of binary labels (0/1)
    """

    def _series(name, default=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    stress_count = (
        (_series("volatility_percentile") > 0.85).astype(int)
        + (_series("atr_expansion_ratio") > 1.5).astype(int)
        + (_series("volume_spike_score") > 0.90).astype(int)
        + (_series("bb_width_percentile") > 0.90).astype(int)
        + (_series("trade_imbalance").abs() > 0.65).astype(int)
    )

    labels = (stress_count >= threshold).astype(int).values

    n_anomaly = labels.sum()
    n_normal = len(labels) - n_anomaly
    logger.info(
        f"Market-stress labels: {n_anomaly} anomaly ({n_anomaly/len(labels)*100:.1f}%) / "
        f"{n_normal} normal ({n_normal/len(labels)*100:.1f}%)"
    )

    return labels
