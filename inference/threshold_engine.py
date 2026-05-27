"""
Adaptive Threshold Engine — Regime-aware momentum thresholds.

Key fixes over original:
  1. REGIME_PERCENTILE_THRESHOLDS now imported from training.config — the
     original hardcoded 0.55 for trending_up overrode the config value of 0.70,
     which is why 577 trades fired in adaptive mode instead of ~30.
  2. Minimum absolute floor added (MIN_ABSOLUTE_THRESHOLD). When the training
     distribution is narrow (median ~0.5), percentile lookups collapse to
     values like 0.51 — meaningless. The floor ensures we never trade below
     a confidence level that has any real edge.
  3. Regime confidence multiplier: low-confidence regimes now raise the
     effective threshold rather than just the volatility adjustment.
  4. _fitted guard raises rather than silently returning 0.31, so misconfigured
     callers fail loudly during testing.
"""

import logging
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Minimum absolute probability threshold regardless of percentile lookup.
# If the model's 70th-percentile output is 0.51, that is noise — we floor
# at MIN_ABSOLUTE_THRESHOLD so we never lower our bar below this.
MIN_ABSOLUTE_THRESHOLD = 0.55


@dataclass
class ThresholdState:
    """Current adaptive threshold with metadata."""
    base_threshold: float
    adjusted_threshold: float
    regime: str
    adjustments: dict = field(default_factory=dict)


class AdaptiveThresholdEngine:
    """
    Calibrates thresholds based on the actual training probability distribution.

    Instead of fixed absolute thresholds (which fail when calibrated
    probabilities land in a narrow range), we compute percentile-based
    thresholds from the training set's momentum_probability distribution,
    then apply a hard floor so a narrow distribution cannot push thresholds
    below a meaningful edge level.
    """

    def __init__(self):
        self._percentiles = {}
        self._fitted = False

    def fit(self, train_momentum_probs: np.ndarray):
        """
        Fit the threshold engine to the training probability distribution.

        Precomputes percentile values at 1% granularity (O(1) lookup at inference).
        """
        valid = train_momentum_probs[~np.isnan(train_momentum_probs)]
        if len(valid) == 0:
            raise ValueError("No valid training probabilities to fit on.")

        for pct in range(0, 101):
            self._percentiles[pct / 100.0] = float(np.percentile(valid, pct))

        logger.info(
            "Threshold engine fitted: "
            "p25=%.3f p50=%.3f p75=%.3f p90=%.3f p95=%.3f",
            self._percentiles[0.25],
            self._percentiles[0.50],
            self._percentiles[0.75],
            self._percentiles[0.90],
            self._percentiles[0.95],
        )

        prob_range = self._percentiles[1.0] - self._percentiles[0.0]
        if prob_range < 0.20:
            logger.warning(
                "Narrow probability range (%.3f) — model outputs are near-uniform. "
                "Re-train Phase 4 with cleaner features before relying on these thresholds.",
                prob_range,
            )

        self._fitted = True

    def get_threshold(
        self,
        regime_label: str,
        volatility_percentile: float = 0.5,
        regime_confidence: float = 0.5,
    ) -> ThresholdState:
        """
        Compute the adaptive momentum threshold.

        Uses percentile-based lookup from config.REGIME_PERCENTILE_THRESHOLDS,
        applies soft adjustments for adverse conditions, then floors at
        MIN_ABSOLUTE_THRESHOLD so a narrow probability distribution can never
        collapse thresholds below a meaningful level.
        """
        if not self._fitted:
            raise RuntimeError(
                "AdaptiveThresholdEngine.fit() must be called before get_threshold()."
            )

        # Import here to avoid circular imports and always get the live value
        try:
            from training.config import REGIME_PERCENTILE_THRESHOLDS
        except ImportError:
            REGIME_PERCENTILE_THRESHOLDS = {
                'trending_up':   0.70,
                'trending_down': 0.70,
                'mixed':         0.80,
                'ranging':       0.95,
                'unknown':       0.90,
            }

        target_pct = REGIME_PERCENTILE_THRESHOLDS.get(regime_label, 0.90)
        pct_key = round(target_pct, 2)
        base = self._percentiles.get(pct_key, self._percentiles.get(0.75, 0.55))

        adjustments = {}
        adjusted = base

        # High volatility: be more selective (shift target percentile upward)
        if volatility_percentile > 0.7:
            pct_shift = (volatility_percentile - 0.7) * 0.15
            higher_pct = min(0.99, round(target_pct + pct_shift, 2))
            candidate = self._percentiles.get(higher_pct, adjusted)
            if candidate > adjusted:
                adjustments['vol_shift'] = pct_shift
                adjusted = candidate

        # Low regime confidence: be more selective (additive shift)
        if regime_confidence < 0.5:
            conf_penalty = (0.5 - regime_confidence) * 0.04
            adjusted = adjusted + conf_penalty
            adjustments['conf_penalty'] = conf_penalty

        # Hard floor: never trade on a signal below MIN_ABSOLUTE_THRESHOLD
        if adjusted < MIN_ABSOLUTE_THRESHOLD:
            adjustments['floor_applied'] = MIN_ABSOLUTE_THRESHOLD - adjusted
            adjusted = MIN_ABSOLUTE_THRESHOLD

        return ThresholdState(
            base_threshold=base,
            adjusted_threshold=adjusted,
            regime=regime_label,
            adjustments=adjustments,
        )


# ── Module-level convenience API ─────────────────────────────────────────────

_global_engine = AdaptiveThresholdEngine()


def fit_threshold_engine(train_momentum_probs: np.ndarray):
    """Fit the global threshold engine."""
    _global_engine.fit(train_momentum_probs)


def compute_adaptive_threshold(
    regime_label: str,
    volatility_percentile: float = 0.5,
    regime_confidence: float = 0.5,
) -> ThresholdState:
    """Compute threshold using the global engine."""
    return _global_engine.get_threshold(
        regime_label,
        volatility_percentile,
        regime_confidence,
    )