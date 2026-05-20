"""
CUSUM Filter — Structural Break Detection.

Implements the symmetric CUSUM filter described in Lopez de Prado's
"Advances in Financial Machine Learning" (Chapter 2).

The filter detects when the cumulative sum of log-returns deviates by more
than a dynamic threshold, signalling a structural break or regime change.

Two signals are emitted:
  - cusum_up   : Upward structural break (potential bearish reversal / crash entry)
  - cusum_down : Downward structural break (potential bullish reversal / recovery)
  - cusum_break: Either direction (OR of the above) — used to block new entries

Usage:
    from regime.cusum import CUSUMFilter
    filt = CUSUMFilter(threshold_pct=0.02)
    df = filt.compute(df)  # adds cusum_up, cusum_down, cusum_break columns
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CUSUMFilter:
    """
    Symmetric CUSUM filter for structural break detection on price series.

    Parameters
    ----------
    threshold_pct : float
        Break threshold expressed as a fraction of current price (e.g. 0.02 = 2%).
        When the cumulative deviation exceeds this fraction of the rolling ATR,
        a break is flagged.
    atr_multiplier : float
        Scales the ATR to set the adaptive threshold.
        threshold = atr_multiplier * ATR_14
    lookback_bars  : int
        Number of bars to sustain the break signal (prevents rapid flickering).
    """

    def __init__(
        self,
        threshold_pct: float = None,
        atr_multiplier: float = 2.0,
        lookback_bars: int = 4,
    ):
        self.threshold_pct  = threshold_pct
        self.atr_multiplier = atr_multiplier
        self.lookback_bars  = lookback_bars

    def _compute_threshold(self, df: pd.DataFrame) -> pd.Series:
        """
        Adaptive threshold: atr_multiplier * ATR_14.
        If ATR_14 is not present, uses rolling std of log-returns.
        """
        if "atr_14" in df.columns:
            return (self.atr_multiplier * df["atr_14"]).clip(lower=1e-6)
        else:
            log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)
            rolling_vol = log_ret.rolling(14).std().fillna(log_ret.std())
            return (self.atr_multiplier * rolling_vol * df["close"]).clip(lower=1e-6)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute symmetric CUSUM filter on log-returns.

        Adds columns:
          - cusum_s_pos  : Running upward CUSUM accumulator
          - cusum_s_neg  : Running downward CUSUM accumulator
          - cusum_up     : 1 if upward structural break fired
          - cusum_down   : 1 if downward structural break fired
          - cusum_break  : 1 if either direction broke (entry block signal)

        The CUSUM accumulators are reset to 0 after each break event.
        A break persists for `lookback_bars` bars to prevent single-bar spikes.
        """
        df = df.copy()
        closes = df["close"].values
        n      = len(closes)

        log_ret   = np.concatenate([[0], np.log(closes[1:] / closes[:-1])])
        threshold = self._compute_threshold(df).values

        s_pos = np.zeros(n)
        s_neg = np.zeros(n)
        cusum_up   = np.zeros(n, dtype=int)
        cusum_down = np.zeros(n, dtype=int)

        for t in range(1, n):
            r = log_ret[t]
            h = threshold[t]

            s_pos[t] = max(0.0, s_pos[t-1] + r)
            s_neg[t] = min(0.0, s_neg[t-1] + r)

            if s_pos[t] >= h:
                cusum_up[t] = 1
                s_pos[t]    = 0.0   # reset accumulator

            if s_neg[t] <= -h:
                cusum_down[t] = 1
                s_neg[t]      = 0.0  # reset accumulator

        # Persist signal for lookback_bars
        if self.lookback_bars > 1:
            kernel = np.ones(self.lookback_bars)
            cusum_up_filled   = np.minimum(1, np.convolve(cusum_up,   kernel, mode="same").astype(int))
            cusum_down_filled = np.minimum(1, np.convolve(cusum_down, kernel, mode="same").astype(int))
        else:
            cusum_up_filled   = cusum_up
            cusum_down_filled = cusum_down

        df["cusum_s_pos"]  = s_pos
        df["cusum_s_neg"]  = s_neg
        df["cusum_up"]     = cusum_up_filled
        df["cusum_down"]   = cusum_down_filled
        df["cusum_break"]  = (cusum_up_filled | cusum_down_filled).astype(int)

        n_up   = int(cusum_up_filled.sum())
        n_down = int(cusum_down_filled.sum())
        logger.info(
            f"CUSUM filter: {n_up} upward breaks, {n_down} downward breaks "
            f"({(n_up + n_down) / n * 100:.1f}% of bars flagged) | "
            f"atr_mult={self.atr_multiplier}, lookback={self.lookback_bars}"
        )
        return df

    def compute_series(self, close_series: pd.Series, atr_series: pd.Series = None) -> pd.Series:
        """
        Lightweight version: returns only the cusum_break boolean Series.
        Useful for feature engineering without modifying a full DataFrame.
        """
        dummy = pd.DataFrame({"close": close_series})
        if atr_series is not None:
            dummy["atr_14"] = atr_series.values
        out = self.compute(dummy)
        return out["cusum_break"].astype(bool)


# ---------------------------------------------------------------------------
# Vectorized batch version (faster for large datasets, no reset-on-break)
# ---------------------------------------------------------------------------

def cusum_events(log_returns: np.ndarray, h: float) -> np.ndarray:
    """
    Fast vectorized CUSUM without per-bar reset.
    Returns boolean array of break events (either direction).
    h: fixed scalar threshold on log-return scale.
    """
    s_pos = 0.0
    s_neg = 0.0
    events = np.zeros(len(log_returns), dtype=bool)
    for i, r in enumerate(log_returns):
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos >= h:
            events[i] = True
            s_pos = 0.0
        if s_neg <= -h:
            events[i] = True
            s_neg = 0.0
    return events
