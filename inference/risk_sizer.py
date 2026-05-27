"""
Risk Sizing Engine — Fractional Kelly Sizing.

Replaces fixed risk percent with dynamic sizing based on Kelly Criterion,
modulated by predicted volatility to maintain constant variance.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    risk_percent: float
    sl_distance: float
    tp_distance: float
    sl_price: float
    tp_price: float
    reward_risk_ratio: float
    position_size_usd: float


def compute_kelly_sizing(
    equity: float,
    entry_price: float,
    direction: int,
    meta_probability: float,
    predicted_volatility: float,
    atr_14: float,
    sl_pct: float = None,
    tp_pct: float = None,
    sl_multiplier: float = 1.5,
    tp_multiplier: float = 1.5,
    regime_risk_modifier: float = 1.0,
    regime_kelly_multiplier: float = 1.0,
    max_risk_percent: float = 1.0,      # Absolute cap on position risk at 1% of equity
    kelly_fraction: float = 0.5,       # Half-Kelly for safety
) -> SizingResult:
    """
    Computes position size using Binomial Kelly Criterion:
      f* = (p * (b + 1) - 1) / b
    where p = Platt-calibrated probability, b = TP_multiplier / SL_multiplier.
    """
    # 1. Use target-compatible percentage barriers for evaluation when provided.
    if sl_pct is not None and tp_pct is not None:
        sl_dist = entry_price * sl_pct
        tp_dist = entry_price * tp_pct
    else:
        sl_dist = atr_14 * sl_multiplier
        tp_dist = atr_14 * tp_multiplier
    
    # Prices
    if direction == 1:
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist
        
    # 2. Reward-Risk Ratio (b)
    if sl_dist <= 0:
        b = 1.0
    else:
        b = tp_dist / sl_dist
        
    p = meta_probability
    
    # Kelly Formula: f* = (p * (b + 1) - 1) / b
    if b > 0:
        f_star = (p * (b + 1) - 1) / b
    else:
        f_star = 0.0
        
    # Cap negative edge to 0
    f_star = max(0.0, f_star)
    
    # 3. Volatility Targeting Scalar
    # High predicted volatility reduces the raw Kelly fraction
    vol_target_scalar = min(2.0, max(0.5, 0.01 / max(predicted_volatility, 0.001)))
    
    # 4. Regime-based scalar modifiers
    raw_risk_pct = f_star * kelly_fraction * vol_target_scalar * regime_risk_modifier * 100.0
    raw_risk_pct *= regime_kelly_multiplier
        
    # 5. Final Risk Percent
    # Hard cap risk at max_risk_percent.
    risk_percent = min(raw_risk_pct, max_risk_percent)
    
    # 6. Position Size USD
    risk_amount = equity * (risk_percent / 100.0)
    stop_fraction = sl_dist / entry_price if entry_price > 0 else 0.01
    
    if stop_fraction > 0:
        position_size_usd = risk_amount / stop_fraction
    else:
        position_size_usd = 0.0
        
    return SizingResult(
        risk_percent=risk_percent,
        sl_distance=sl_dist,
        tp_distance=tp_dist,
        sl_price=sl_price,
        tp_price=tp_price,
        reward_risk_ratio=b,
        position_size_usd=position_size_usd
    )
