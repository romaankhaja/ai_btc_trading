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


def label_behavioral(df, threshold=0.6):
    """
    Generate behavioral anomaly labels.

    Builds a composite anomaly score from the available causal
    behavioral proxies. This avoids depending on the removed
    legacy composite field while keeping the target compatible with
    the current behavioral feature set.

    Args:
        df: DataFrame with behavioral feature columns
        threshold: Score above which behavior is labeled anomalous

    Returns:
        numpy array of binary labels (0/1)
    """

    def _series(name, default=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    oversized = _series("oversized_trade_score")
    overtrading = _series("overtrading_score")
    revenge = _series("revenge_trade_score")
    panic = _series("panic_exit_score")
    fomo = _series("fomo_score")
    discipline = _series("discipline_score", default=1.0)
    strategy_health = _series("strategy_health_score", default=0.5)
    recent_accuracy = _series("strategy_recent_accuracy", default=0.5)
    recent_rr = _series("strategy_avg_rr", default=1.0)
    last_5_winrate = _series("last_5_trade_winrate", default=0.5)

    risk_score = (
        0.25 * revenge
        + 0.20 * overtrading
        + 0.15 * oversized
        + 0.10 * panic
        + 0.10 * fomo
        + 0.10 * (1.0 - discipline)
        + 0.05 * (1.0 - strategy_health)
        + 0.03 * (1.0 - recent_accuracy)
        + 0.02 * (1.0 - np.clip(recent_rr / 3.0, 0.0, 1.0))
        + 0.05 * (1.0 - last_5_winrate)
    ).clip(0.0, 1.0)

    labels = (risk_score > threshold).astype(int).values

    n_anomaly = labels.sum()
    n_normal = len(labels) - n_anomaly
    logger.info(
        f"Behavioral labels: {n_anomaly} anomaly ({n_anomaly/len(labels)*100:.1f}%) / "
        f"{n_normal} normal ({n_normal/len(labels)*100:.1f}%)"
    )

    return labels
