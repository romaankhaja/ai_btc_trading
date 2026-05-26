"""
Policy Engine - The Final Authority Layer.

Models output probabilities. The Policy Engine makes decisions.
It applies hard safety constraints, soft scoring adjustments,
and regime-aware risk modifications.
"""

import logging
from dataclasses import dataclass, field
from typing import List

from inference.model_ensemble import ModelOutputs
from training.config import NO_TRADE_REGIMES

logger = logging.getLogger(__name__)


@dataclass
class PolicyDecision:
    """Output of the Policy Engine."""
    allow_trade: bool = False
    risk_percent: float = 0.0
    sl_multiplier: float = 1.5
    tp_multiplier: float = 1.5
    block_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    regime_action: str = 'normal'  # normal, reduce, block


class PolicyEngine:
    """
    Institutional-grade policy engine.

    Applies in order:
    1. Hard blocks (instant rejection, no override)
    2. Regime routing (regime-specific behavior)
    3. Soft adjustments (risk reduction based on conditions)
    4. Final authorization
    """

    STRATEGY_DISABLE_THRESHOLD = 0.25
    ILLIQUIDITY_BLOCK_THRESHOLD = 0.15
    SL_COOLDOWN_BARS = 12

    def evaluate(self, model_outputs: ModelOutputs, features: dict) -> PolicyDecision:
        """
        Run the full policy evaluation.
        """
        decision = PolicyDecision(allow_trade=True, risk_percent=1.0)

        # Stage 1: hard blocks
        if model_outputs.regime_label in NO_TRADE_REGIMES:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(f'HARD_BLOCK: {model_outputs.regime_label} regime detected')
            decision.regime_action = 'block'
            return decision

        if model_outputs.behavioral_anomaly:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append('HARD_BLOCK: behavioral anomaly detected by IsolationForest')
            return decision

        illiquidity = features.get('amihud_illiquidity', 0.0)
        if illiquidity > self.ILLIQUIDITY_BLOCK_THRESHOLD:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(
                f'HARD_BLOCK: amihud_illiquidity={illiquidity:.4f} > {self.ILLIQUIDITY_BLOCK_THRESHOLD}'
            )
            return decision

        health = features.get('strategy_health_score', 1.0)
        if health < self.STRATEGY_DISABLE_THRESHOLD:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(
                f'HARD_BLOCK: strategy_health={health:.2f} < {self.STRATEGY_DISABLE_THRESHOLD}'
            )
            return decision
        if health < 0.30:
            decision.risk_percent *= 0.5
            decision.warnings.append(
                f'SOFT: strategy_health={health:.2f} < 0.30 -> 50% risk reduction'
            )

        if features.get('cusum_break', 0) == 1:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append('HARD_BLOCK: CUSUM structural break signal is active')
            return decision

        discipline = features.get('discipline_score', 1.0)
        if discipline < 0.25:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(
                f'HARD_BLOCK: discipline_score={discipline:.2f} < 0.25'
            )
            return decision
        if discipline < 0.5:
            decision.risk_percent *= 0.5
            decision.warnings.append(
                f'SOFT: discipline_score={discipline:.2f} < 0.50 -> 50% risk reduction'
            )

        aggression = features.get('revenge_trade_score', 0.0)
        overtrading = features.get('overtrading_score', 0.0)
        if aggression > 0.85:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(
                f'HARD_BLOCK: revenge_trade_score={aggression:.2f} > 0.85 (Revenge trading detected)'
            )
            return decision
        if aggression > 0.70:
            decision.risk_percent *= 0.5
            decision.warnings.append(
                f'SOFT: revenge_trade_score={aggression:.2f} > 0.70 -> 50% risk reduction'
            )

        if overtrading > 0.85:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.block_reasons.append(
                f'HARD_BLOCK: overtrading_score={overtrading:.2f} > 0.85 (Overtrading detected)'
            )
            return decision
        if overtrading > 0.70:
            decision.risk_percent *= 0.75
            decision.warnings.append(
                f'SOFT: overtrading_score={overtrading:.2f} > 0.70 -> 25% risk reduction'
            )

        # Consecutive SL circuit breaker
        sl_streak = int(features.get('consecutive_sl_count', 0))
        sl_cooldown = int(features.get('sl_cooldown_bars_remaining', 0))

        if sl_cooldown > 0:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.regime_action = 'block'
            decision.block_reasons.append(
                f'HARD_BLOCK: SL circuit breaker cooldown active ({sl_cooldown} bars remaining)'
            )
            return decision

        if sl_streak >= 4:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.regime_action = 'block'
            decision.block_reasons.append(
                f'HARD_BLOCK: consecutive_sl_count={sl_streak} >= 4 (full block)'
            )
            return decision

        if sl_streak == 3:
            decision.allow_trade = False
            decision.risk_percent = 0.0
            decision.regime_action = 'block'
            decision.block_reasons.append(
                f'HARD_BLOCK: consecutive_sl_count={sl_streak} -> 12-bar cooldown'
            )
            return decision

        if sl_streak == 2:
            decision.risk_percent *= 0.5
            decision.warnings.append('CIRCUIT_BREAKER: 2 consecutive SLs -> 50% Kelly reduction')

        # Risk model says NO_TRADE - treat as strong soft reduction
        if model_outputs.risk_level == 'NO_TRADE':
            decision.risk_percent *= 0.10
            decision.warnings.append('SOFT: risk_model=NO_TRADE -> 90% risk reduction')

        # Stage 2: regime routing
        regime = model_outputs.regime_label

        if regime in ('trending_up', 'trending_down'):
            decision.regime_action = 'normal'
            decision.sl_multiplier = 1.0
            decision.tp_multiplier = 2.5

        elif regime == 'mixed':
            decision.regime_action = 'reduce'
            decision.sl_multiplier = 1.2
            decision.tp_multiplier = 1.6
            decision.risk_percent *= 0.75
            decision.warnings.append('REGIME: mixed - reduced sizing')

        # Stage 3: trend alignment
        ema_50_slope = features.get('ema_50_slope', 0.0)
        direction = model_outputs.predicted_direction
        if direction == -1 and ema_50_slope > 0:
            decision.risk_percent *= 0.5
            decision.warnings.append('TREND: SHORT against bullish ema_50_slope -> 50% risk reduction')
        elif direction == 1 and ema_50_slope < 0:
            decision.risk_percent *= 0.5
            decision.warnings.append('TREND: LONG against bearish ema_50_slope -> 50% risk reduction')

        # Stage 4: soft adjustments
        vol_pct = features.get('volatility_percentile', 0.5)
        if vol_pct > 0.75:
            vol_factor = max(0.5, 1.0 - (vol_pct - 0.75))
            decision.risk_percent *= vol_factor
            decision.sl_multiplier *= 1.2
            decision.warnings.append(f'SOFT: vol_percentile {vol_pct:.2f} -> widened SL, reduced size')

        if model_outputs.risk_level == 'HIGH_RISK':
            decision.risk_percent *= 0.5
            decision.warnings.append('SOFT: risk_model=HIGH_RISK -> 50% risk reduction')
        elif model_outputs.risk_level == 'MEDIUM_RISK':
            decision.risk_percent *= 0.75
            decision.warnings.append('SOFT: risk_model=MEDIUM_RISK -> 25% risk reduction')

        oversized = features.get('oversized_trade_score', 0.0)
        if oversized > 0.5:
            decision.risk_percent = min(decision.risk_percent, 0.5)
            decision.warnings.append(f'SOFT: oversized_trade_score={oversized:.2f} -> capped risk')

        decision.risk_percent = max(0.1, min(1.5, decision.risk_percent))
        assert model_outputs.regime_label not in NO_TRADE_REGIMES or not decision.allow_trade, (
            'NO_TRADE_REGIMES must always resolve to NO_TRADE'
        )
        return decision
