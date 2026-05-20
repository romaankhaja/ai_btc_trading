"""
Trade Decision Orchestrator — Rolling Z-Score Execution.

Replaces static probability thresholds with cross-sectional 
Z-score ranking based on the raw log-odds (margin) of the Meta-Model.
"""

import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List

from inference.model_ensemble import ModelEnsemble, ModelOutputs
from inference.policy_engine import PolicyEngine, PolicyDecision
from inference.risk_sizer import compute_kelly_sizing, SizingResult
from training.config import REGIME_KELLY_MULTIPLIER

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    """Complete trade decision output."""
    action: str = 'NO_TRADE'   # LONG, SHORT, NO_TRADE
    
    # Confidence
    meta_probability: float = 0.0
    meta_margin_zscore: float = 0.0
    zscore_threshold: float = 0.25  # selective but not overly restrictive
    
    # Risk parameters
    risk_percent: float = 0.0
    sl_distance: float = 0.0
    tp_distance: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    reward_risk_ratio: float = 0.0
    position_size_usd: float = 0.0
    
    # Context
    regime: str = 'unknown'
    regime_confidence: float = 0.0
    raw_regime: str = 'unknown'
    regime_override_applied: bool = False
    regime_override_reason: str = ''
    atr_ratio: float = 1.0
    
    # Governance
    block_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    policy_allowed: bool = False


class TradeDecisionEngine:
    """
    Institutional Z-Score Ranking Orchestrator.
    """
    
    def __init__(self, models_dir: str):
        self.ensemble = ModelEnsemble(models_dir)
        self.policy = PolicyEngine()
        self._loaded = False
        
        # Rolling window for Z-score calculation (last 1000 bars)
        self.margin_window = deque(maxlen=1000)
    
    def load(self):
        """Load all models."""
        self.ensemble.load()
        self._loaded = True
    
    def decide(self, features: dict, equity: float = 10000.0) -> TradeDecision:
        decision = TradeDecision()
        
        # 1. Model Inference
        outputs = self.ensemble.predict(features)
        decision.meta_probability = outputs.meta_probability
        decision.regime = outputs.regime_label
        decision.regime_confidence = outputs.regime_confidence
        decision.raw_regime = outputs.regime_label

        atr = features.get('atr_14', 0.0)
        atr_mean = features.get('atr_20bar_mean', 0.0)
        atr_ratio = float(atr) / float(atr_mean) if atr_mean else 0.0
        decision.atr_ratio = atr_ratio
        if decision.raw_regime == 'trending_low_vol' and atr_ratio > 1.4:
            outputs.regime_label = 'trending_high_vol'
            features['regime_label'] = 'trending_high_vol'
            decision.regime = 'trending_high_vol'
            decision.regime_override_applied = True
            decision.regime_override_reason = f'ATR ratio {atr_ratio:.2f} > 1.40 -> trending_high_vol'

        # Update rolling margin window
        self.margin_window.append(outputs.meta_margin)
        
        # 2. Compute Rolling Z-Score
        if len(self.margin_window) > 100:
            window_mean = np.mean(self.margin_window)
            window_std = np.std(self.margin_window)
            if window_std > 0:
                decision.meta_margin_zscore = (outputs.meta_margin - window_mean) / window_std
            else:
                decision.meta_margin_zscore = 0.0
        else:
            # Not enough data for Z-score, block trade
            decision.action = 'NO_TRADE'
            decision.block_reasons.append('WARMUP: Insufficient history for Z-score')
            return decision
        
        # 3. Policy Engine Gating
        policy = self.policy.evaluate(outputs, features)
        decision.policy_allowed = policy.allow_trade
        decision.block_reasons = policy.block_reasons
        decision.warnings = policy.warnings
        
        if not policy.allow_trade:
            decision.action = 'NO_TRADE'
            return decision
            
        # 4. Z-Score Execution Threshold
        # We only trade if the signal is in the top tier of recent history
        if decision.meta_margin_zscore < decision.zscore_threshold:
            decision.action = 'NO_TRADE'
            decision.block_reasons.append(
                f'THRESHOLD: zscore={decision.meta_margin_zscore:.2f} < {decision.zscore_threshold:.2f}'
            )
            return decision
        
        # 5. Action Direction
        direction = outputs.predicted_direction
        decision.action = 'LONG' if direction == 1 else 'SHORT'
        
        # 6. Kelly Sizing
        atr = features.get('atr_14', 0.0)
        entry_price = features.get('close', 0.0)
        pred_vol = outputs.predicted_volatility
        
        if atr > 0 and entry_price > 0:
            sizing = compute_kelly_sizing(
                equity=equity,
                entry_price=entry_price,
                direction=direction,
                meta_probability=outputs.meta_probability,
                predicted_volatility=pred_vol,
                atr_14=atr,
                sl_multiplier=policy.sl_multiplier,
                tp_multiplier=policy.tp_multiplier,
                regime_risk_modifier=policy.risk_percent, # Using Policy's modified risk as a scalar
                regime_kelly_multiplier=REGIME_KELLY_MULTIPLIER.get(decision.regime, 1.0),
                kelly_fraction=getattr(outputs, 'kelly_fraction', 0.5),
                consecutive_losses=int(features.get('consecutive_losses', 0)),
                recent_drawdown=float(features.get('recent_drawdown', 0.0))
            )
            
            decision.risk_percent = sizing.risk_percent
            decision.sl_distance = sizing.sl_distance
            decision.tp_distance = sizing.tp_distance
            decision.sl_price = sizing.sl_price
            decision.tp_price = sizing.tp_price
            decision.reward_risk_ratio = sizing.reward_risk_ratio
            decision.position_size_usd = sizing.position_size_usd
            
            # Additional sanity check on Kelly sizing
            if sizing.risk_percent <= 0:
                decision.action = 'NO_TRADE'
                decision.block_reasons.append(f'KELLY: Negative edge, risk_percent={sizing.risk_percent:.2f}%')
        
        return decision
