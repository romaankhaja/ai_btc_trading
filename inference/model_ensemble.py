"""
Model Ensemble — Hierarchical Meta-Ensemble Loader.

Loads the canonical Phase 3/4 architecture:
1. KMeans Regime Model
2. Global Momentum Model
3. Volatility Model
4. Risk Model
5. Behavioral Model
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelOutputs:
    regime_label: str = 'unknown'
    regime_confidence: float = 0.0
    predicted_direction: int = 0
    meta_probability: float = 0.0
    meta_margin: float = 0.0
    kelly_fraction: float = 0.5
    predicted_volatility: float = 0.0
    risk_level: str = 'NO_TRADE'
    behavioral_anomaly: bool = False


class ModelEnsemble:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.regime_scaler = None
        self.regime_kmeans = None
        self.regime_mapping = None
        self.momentum_model = None
        self.volatility_model = None
        self.risk_model = None
        self.behavioral_model = None
        self._loaded = False

    def load(self):
        logger.info("Loading Phase 3/4 ensemble models...")
        
        # 1. KMeans Regime
        reg_dir = self.models_dir / 'regime'
        self.regime_scaler = joblib.load(reg_dir / 'regime_scaler.pkl')
        self.regime_kmeans = joblib.load(reg_dir / 'regime_kmeans.pkl')
        self.regime_mapping = joblib.load(reg_dir / 'regime_mapping.pkl')
        
        # 2. Global momentum model
        mom_dir = self.models_dir / 'momentum'
        self.momentum_model = joblib.load(mom_dir / 'momentum_model.pkl')
        
        # 3. Volatility
        vol_dir = self.models_dir / 'volatility'
        self.volatility_model = joblib.load(vol_dir / 'volatility_model.pkl')
        
        # 4. Risk
        risk_dir = self.models_dir / 'risk'
        self.risk_model = joblib.load(risk_dir / 'risk_model.pkl')
        
        # 5. Behavioral
        beh_dir = self.models_dir / 'behavioral'
        self.behavioral_model = joblib.load(beh_dir / 'behavioral_iforest.pkl')
        
        self._loaded = True
        logger.info("All Phase 3/4 models loaded successfully.")
        
    def predict(self, features: dict) -> ModelOutputs:
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
            
        out = ModelOutputs()
        
        # 1. Regime (KMeans on current bar)
        try:
            from labeling.regime_labeler import REGIME_FEATURES, assign_regime_labels

            regime_input = pd.DataFrame([features])
            for col in REGIME_FEATURES:
                if col not in regime_input.columns:
                    regime_input[col] = 0.0
            regime_input = regime_input[REGIME_FEATURES]
            labeled = assign_regime_labels(regime_input, self.regime_scaler, self.regime_kmeans, self.regime_mapping)
            last_row = labeled.iloc[0]
            
            out.regime_label = last_row['regime_label']
            out.regime_confidence = float(last_row['regime_confidence'])
            
            features['regime_state'] = int(last_row['regime_state'])
            features['regime_label'] = last_row['regime_label']
            features['regime_confidence'] = out.regime_confidence
        except Exception as e:
            logger.warning(f"Regime prediction error: {e}. Defaulting to mixed.")
            out.regime_label = 'mixed'
            features['regime_state'] = 0
            features['regime_label'] = 'mixed'
            features['regime_confidence'] = 0.5
            
        # Import feature list constants
        from training.config import MOMENTUM_FEATURES, VOLATILITY_FEATURES, RISK_FEATURES, BEHAVIORAL_FEATURES
        
        # 2. Global momentum model
        x_mom = np.array([[features.get(f, 0.0) for f in MOMENTUM_FEATURES]])
        mom_proba = self.momentum_model.predict_proba(x_mom)
        if getattr(mom_proba, "ndim", 1) == 2:
            out.meta_probability = float(mom_proba[0, 1])
        else:
            out.meta_probability = float(mom_proba[0] if np.ndim(mom_proba) else mom_proba)
        out.predicted_direction = 1 if out.meta_probability >= 0.5 else -1
        p = np.clip(out.meta_probability, 1e-6, 1 - 1e-6)
        out.meta_margin = float(np.log(p / (1.0 - p)))
        out.kelly_fraction = float(getattr(self.momentum_model, 'kelly_fraction', 0.5))

        features['momentum_probability'] = out.meta_probability
        
        # 4. Volatility
        x_vol = np.array([[features.get(f, 0.0) for f in VOLATILITY_FEATURES]])
        out.predicted_volatility = self.volatility_model.predict(x_vol)[0]
        features['predicted_volatility'] = out.predicted_volatility
        
        # 5. Risk
        x_risk = np.array([[features.get(f, 0.0) for f in RISK_FEATURES]])
        out.risk_level = self.risk_model.predict(x_risk)[0]
        
        # 6. Behavioral
        x_beh = np.array([[features.get(f, 0.0) for f in BEHAVIORAL_FEATURES]])
        beh_df = pd.DataFrame(x_beh, columns=BEHAVIORAL_FEATURES)
        anomaly = self.behavioral_model.predict(beh_df)[0]
        out.behavioral_anomaly = (anomaly == -1)
        
        return out
