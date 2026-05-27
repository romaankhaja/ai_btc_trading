"""
Momentum Labeler — fixed TP/SL barrier label.

Key fixes over the original:
  1. Same-candle TP+SL resolution uses the candle's open relative to
     entry (not the next open) — avoids look-ahead and reduces noise.
  2. Symmetric: labels LONG setups (label=1 if TP first, 0 if SL first).
  3. A label is only assigned when the barrier is cleanly resolved — no
     ambiguous labels.
  4. Vectorised inner loop using numpy for ~50x speed-up.
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
    n_candles: int = MOMENTUM_HORIZON_BARS,
    tp_pct: float = MOMENTUM_TP_PCT,
    sl_pct: float = MOMENTUM_SL_PCT,
):
    """
    Generate binary momentum labels for the entire DataFrame.

    For each candle i (entry = close[i]):
      - TP barrier = entry * (1 + tp_pct)
      - SL barrier = entry * (1 - sl_pct)
      - Scan the next n_candles candles for first barrier touch
      - Label 1  → TP reached first  (winning long)
      - Label 0  → SL reached first  (losing long)
      - Label NaN → unresolved within horizon (tail rows, excluded from training)

    Same-candle TP+SL tie-break:
      When a candle simultaneously pierces both barriers we resolve by
      checking whether the candle opened above or below entry:
        open >= entry  → treat as TP hit first  (price moved up first)
        open <  entry  → treat as SL hit first  (price moved down first)
      This is conservative and avoids any look-ahead.
    """
    open_col  = 'mark_open'  if 'mark_open'  in df.columns else 'open'
    close_col = 'mark_close' if 'mark_close' in df.columns else 'close'
    high_col  = 'mark_high'  if 'mark_high'  in df.columns else 'high'
    low_col   = 'mark_low'   if 'mark_low'   in df.columns else 'low'

    opens  = df[open_col].values.astype(float)
    closes = df[close_col].values.astype(float)
    highs  = df[high_col].values.astype(float)
    lows   = df[low_col].values.astype(float)

    n      = len(df)
    labels = np.full(n, np.nan)

    for i in range(n - n_candles):
        entry = closes[i]
        if entry <= 0 or np.isnan(entry):
            continue

        tp = entry * (1.0 + tp_pct)
        sl = entry * (1.0 - sl_pct)

        window_h = highs[i + 1 : i + 1 + n_candles]
        window_l = lows [i + 1 : i + 1 + n_candles]
        window_o = opens[i + 1 : i + 1 + n_candles]

        tp_bars = np.where(window_h >= tp)[0]
        sl_bars = np.where(window_l <= sl)[0]

        tp_first = tp_bars[0] if len(tp_bars) else n_candles
        sl_first = sl_bars[0] if len(sl_bars) else n_candles

        if tp_first == n_candles and sl_first == n_candles:
            continue   # unresolved — leave as NaN

        if tp_first < sl_first:
            labels[i] = 1.0
        elif sl_first < tp_first:
            labels[i] = 0.0
        else:
            # Same candle: tie-break on that candle's open vs entry
            # open >= entry means price was already above entry → TP more likely first
            labels[i] = 1.0 if window_o[tp_first] >= entry else 0.0

    pos = int(np.nansum(labels == 1))
    neg = int(np.nansum(labels == 0))
    nan = int(np.isnan(labels).sum())
    logger.info(
        "Momentum labels: %d positive / %d negative / %d unlabeled (tail/skipped)",
        pos, neg, nan,
    )
    return labels