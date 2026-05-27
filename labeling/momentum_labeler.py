"""
Momentum Labeler - Clean percentage TP/SL label.

Binary classification target:
  1 = long TP hit before long SL within N candles
  0 = long SL hit before long TP within N candles
  NaN = unresolved within horizon
"""

import logging
import numpy as np

from training.config import (
    MOMENTUM_HORIZON_BARS,
    MOMENTUM_SL_PCT,
    MOMENTUM_TP_PCT,
)

logger = logging.getLogger(__name__)


def label_momentum(
    df,
    n_candles=MOMENTUM_HORIZON_BARS,
    tp_pct=MOMENTUM_TP_PCT,
    sl_pct=MOMENTUM_SL_PCT,
):
    """
    Generate clean momentum labels for the entire DataFrame.

    For each candle i:
      - Entry = current close
      - TP = entry * (1 + tp_pct)
      - SL = entry * (1 - sl_pct)
      - Label 1 if TP hits before SL
      - Label 0 if SL hits before TP
      - Leave unresolved paths as NaN
    """
    open_col = 'mark_open' if 'mark_open' in df.columns else 'open'
    close_col = 'mark_close' if 'mark_close' in df.columns else 'close'
    high_col = 'mark_high' if 'mark_high' in df.columns else 'high'
    low_col = 'mark_low' if 'mark_low' in df.columns else 'low'

    opens = df[open_col].values
    closes = df[close_col].values
    highs = df[high_col].values
    lows = df[low_col].values

    n = len(df)
    labels = np.full(n, np.nan)

    for i in range(n - n_candles):
        entry = closes[i]
        if entry <= 0 or np.isnan(entry):
            continue

        tp = entry * (1.0 + tp_pct)
        sl = entry * (1.0 - sl_pct)

        tp_hit = False
        sl_hit = False

        for j in range(i + 1, i + 1 + n_candles):
            if highs[j] >= tp and lows[j] <= sl:
                if j + 1 < len(closes):
                    if opens[j + 1] >= entry:
                        tp_hit = True
                    else:
                        sl_hit = True
                else:
                    tp_hit = True
                break
            if highs[j] >= tp:
                tp_hit = True
                break
            if lows[j] <= sl:
                sl_hit = True
                break

        if tp_hit and not sl_hit:
            labels[i] = 1.0
        elif sl_hit and not tp_hit:
            labels[i] = 0.0

    logger.info(
        f"Momentum labels: {int(np.nansum(labels))} positive / "
        f"{int(np.nansum(labels == 0))} negative / "
        f"{int(np.isnan(labels).sum())} unlabeled (tail/skipped)"
    )

    return labels
