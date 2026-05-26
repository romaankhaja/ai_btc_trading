"""
Model Ensemble — Hierarchical Meta-Ensemble Loader.

Loads the new architecture:
1. Non-Homogeneous HMM Regime Model
2. Four specialised Meta-Ensemble Models (routed by HMM regime)
3. Volatility Model
4. Risk Model
5. Behavioral Model
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import deque
from dataclasses import dataclass

from regime.nhhmm import NHHMMRegimeDetector

logger = logging.getLogger(__name__)

# Rolling window size for HMM temporal context
HMM_WINDOW_SIZE = 50


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
        self.regime_detector = None
        self.regime_models = {}
        self.volatility_model = None
        self.risk_model = None
        self.behavioral_model = None
        self._loaded = False
        
        # Rolling buffer for HMM temporal context
        self._hmm_buffer = deque(maxlen=HMM_WINDOW_SIZE)
        
    def load(self):
        logger.info("Loading HMM Meta-Ensemble models...")
        
        # 1. HMM Regime
        reg_dir = self.models_dir / 'regime'
        self.regime_detector = NHHMMRegimeDetector.load(reg_dir)
        
        # 2. Three Specialized Meta-Ensembles
        regimes = ["trending", "sideways", "high_risk"]
        mom_dir = self.models_dir / 'momentum'
        
        for r in regimes:
            path = mom_dir / r
            if (path / 'primary_model.pkl').exists():
                self.regime_models[r] = {
                    'primary': joblib.load(path / 'primary_model.pkl'),
                    'meta': joblib.load(path / 'calibrated_meta_model.pkl')
                }
                logger.info(f"  Loaded specialized meta-ensemble for regime: {r}")
            else:
                logger.warning(f"  No saved meta-ensemble model found for regime: {r}")
        
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
        logger.info("All specialized HMM models loaded successfully.")
        
    def predict(self, features: dict) -> ModelOutputs:
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
            
        out = ModelOutputs()
        
        # 1. Regime (HMM with custom Viterbi decoding)
        try:
            # Append features dict to buffer
            self._hmm_buffer.append(features.copy())
            
            # Construct sequence DataFrame
            seq_df = pd.DataFrame(list(self._hmm_buffer))
            
            # Decode using NHHMM
            seq_df_labeled = self.regime_detector.assign_labels(seq_df)
            last_row = seq_df_labeled.iloc[-1]
            
            out.regime_label = last_row['regime_label']
            out.regime_confidence = float(last_row['regime_confidence'])
            
            # Inject state metrics into features dictionary
            features['regime_state'] = int(last_row['regime_state'])
            features['regime_label'] = last_row['regime_label']
            features['regime_confidence'] = out.regime_confidence
        except Exception as e:
            logger.warning(f"NHHMM regime prediction error: {e}. Defaulting to sideways.")
            out.regime_label = 'sideways'
            features['regime_state'] = 1
            features['regime_label'] = 'sideways'
            features['regime_confidence'] = 0.5
            
        # Import feature list constants
        from training.config import MOMENTUM_FEATURES, META_FEATURES, VOLATILITY_FEATURES, RISK_FEATURES, BEHAVIORAL_FEATURES
        
        # 2 & 3. Route to corresponding specialized Meta-Ensemble
        if out.regime_label in self.regime_models:
            models = self.regime_models[out.regime_label]
            primary_model = models['primary']
            meta_model = models['meta']
            
            # Primary Direction
            x_mom = np.array([[features.get(f, 0.0) for f in MOMENTUM_FEATURES]])
            pred_dir = primary_model.predict(x_mom)[0]
            out.predicted_direction = 1 if pred_dir == 1 else -1
            
            # Meta-Model (Calibrated)
            x_meta = np.array([[features.get(f, 0.0) for f in META_FEATURES]])
            meta_proba = meta_model.predict_proba(x_meta)
            if getattr(meta_proba, "ndim", 1) == 2:
                out.meta_probability = float(meta_proba[0, 1])
            else:
                out.meta_probability = float(meta_proba[0] if np.ndim(meta_proba) else meta_proba)
            out.meta_margin = meta_model.predict(x_meta, output_margin=True)[0]
            out.kelly_fraction = float(getattr(meta_model, 'kelly_fraction', 0.5))
        else:
            # Flat/no trades for unhandled or highly choppy/crash regimes
            out.predicted_direction = 0
            out.meta_probability = 0.0
            out.meta_margin = -999.0
            out.kelly_fraction = 0.5
            
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
