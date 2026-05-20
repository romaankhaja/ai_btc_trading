"""
Regime-Specific Hierarchical Meta-Ensemble Training.

Phase 6 Redesign (Upgraded):
Trains FOUR specialized meta-ensembles based on the NHHMM regime:
  1. Model_Trend_LowVol  (trending_low_vol)
  2. Model_Trend_HighVol (trending_high_vol)
  3. Model_Chop          (sideways_low_vol)
  4. Model_Crisis        (crash_mode)

For each regime:
  - Primary Model: Predicts Direction (LONG vs SHORT) using MOMENTUM_FEATURES.
  - Meta Model   : Predicts Success (TP hit before SL) using META_FEATURES.
  - Calibration  : Platt Scaling (fitting a LogisticRegression on the raw meta-model margins).
  - Validation   : Expected Calibration Error (ECE) check. High ECE (>8%) triggers warnings.
"""

import logging
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from training.config import MOMENTUM_FEATURES, META_FEATURES, MOMENTUM_PARAMS
from training.evaluate import evaluate_momentum, get_feature_importance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE) metric helper
# ---------------------------------------------------------------------------

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) using uniform-width binning.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(probs)
    if n_samples == 0:
        return 0.0
        
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece


# ---------------------------------------------------------------------------
# Platt Calibration Wrapper
# ---------------------------------------------------------------------------

class PlattCalibratedMetaModel:
    """
    XGBoost Meta-Model calibrated using Platt Scaling (Logistic Regression on raw margins).
    """
    def __init__(self, xgb_model, scaler=None, platt_lr=None):
        self.xgb_model = xgb_model
        self.scaler = scaler or StandardScaler()
        self.platt_lr = platt_lr or LogisticRegression()
        self.kelly_fraction = 0.5

    def fit_calibration(self, X_val, y_val):
        """
        Fits logistic regression calibration model on validation/calibration set margins.
        """
        margins = self.xgb_model.predict(X_val, output_margin=True).reshape(-1, 1)
        margins_scaled = self.scaler.fit_transform(margins)
        self.platt_lr.fit(margins_scaled, y_val)
        return self

    def predict_proba(self, X):
        """
        Predict calibrated probabilities.
        """
        margins = self.xgb_model.predict(X, output_margin=True).reshape(-1, 1)
        margins_scaled = self.scaler.transform(margins)
        proba = self.platt_lr.predict_proba(margins_scaled)[:, 1]
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X, output_margin=False):
        if output_margin:
            return self.xgb_model.predict(X, output_margin=True)
        return self.xgb_model.predict(X)


# ---------------------------------------------------------------------------
# Training Orchestration
# ---------------------------------------------------------------------------

def train_single_regime_ensemble(
    regime_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path
) -> dict:
    """
    Trains and calibrates a single meta-ensemble for a specific regime.
    """
    logger.info(f"\n--- Training Meta-Ensemble for Regime: {regime_name} ---")
    
    # 1. Filter splits for specific regime
    train_sub = train_df[train_df['regime_label'] == regime_name].copy()
    val_sub   = val_df[val_df['regime_label'] == regime_name].copy()
    test_sub  = test_df[test_df['regime_label'] == regime_name].copy()
    
    logger.info(f"  Samples: Train={len(train_sub)}, Val={len(val_sub)}, Test={len(test_sub)}")
    
    if len(train_sub) < 50:
        logger.warning(f"  Too few samples for {regime_name} training! Using fallback model.")
        return None
        
    # Drop target NaNs (e.g. tail rows or choppy_high_vol NaN-labeled skips)
    train_prim_clean = train_sub.dropna(subset=['primary_label'])
    train_meta_clean = train_sub.dropna(subset=['meta_label'])
    val_meta_clean   = val_sub.dropna(subset=['meta_label'])
    test_meta_clean  = test_sub.dropna(subset=['meta_label'])
    
    if len(train_prim_clean) < 10 or len(train_meta_clean) < 10:
        logger.warning(f"  No valid target labels for {regime_name}! Skipped training.")
        return None

    # ---- A. Train Primary Directional Model --------------------------------
    X_train_prim = train_prim_clean[MOMENTUM_FEATURES]
    y_train_prim = (train_prim_clean['primary_label'] == 1).astype(int)  # 1=Long, 0=Short/Flat
    
    logger.info("  Training Primary Model...")
    prim_model = xgb.XGBClassifier(**MOMENTUM_PARAMS, eval_metric='logloss')
    prim_model.fit(X_train_prim, y_train_prim)
    
    # ---- B. Train Secondary Meta-Confidence Model --------------------------
    X_train_meta = train_meta_clean[META_FEATURES]
    y_train_meta = train_meta_clean['meta_label'].astype(int)
    
    logger.info("  Training Meta-Confidence Model...")
    meta_model = xgb.XGBClassifier(**MOMENTUM_PARAMS, eval_metric='logloss')
    meta_model.fit(X_train_meta, y_train_meta)
    
    # ---- C. Platt Scaling Calibration -------------------------------------
    calibrated_meta = PlattCalibratedMetaModel(meta_model)
    
    if not val_meta_clean.empty:
        logger.info("  Calibrating Meta-Model via Platt Scaling on validation set...")
        X_val_meta = val_meta_clean[META_FEATURES]
        y_val_meta = val_meta_clean['meta_label'].astype(int)
        
        calibrated_meta.fit_calibration(X_val_meta, y_val_meta)
        
        # Compute ECE on Val set
        val_probs_raw = meta_model.predict_proba(X_val_meta)[:, 1]
        val_probs_cal = calibrated_meta.predict_proba(X_val_meta)[:, 1]
        
        ece_raw = compute_ece(val_probs_raw, y_val_meta.values)
        ece_cal = compute_ece(val_probs_cal, y_val_meta.values)
        
        logger.info(f"  [Validation ECE] Raw ECE: {ece_raw*100:.2f}%, Calibrated ECE: {ece_cal*100:.2f}%")
        
        # Hard limits check
        if ece_cal > 0.08:
            logger.warning(f"  [CRITICAL WARNING] Calibrated ECE ({ece_cal*100:.2f}%) exceeds acceptable threshold (8%)!")
            calibrated_meta.kelly_fraction = 0.25
        elif ece_cal < 0.05:
            logger.info(f"  Calibration quality APPROVED (ECE < 5.0%)")
            calibrated_meta.kelly_fraction = 0.5
    else:
        logger.warning("  Validation set empty for calibration. Fitting on training set as fallback...")
        calibrated_meta.fit_calibration(X_train_meta, y_train_meta)
        ece_cal = 0.0
        calibrated_meta.kelly_fraction = 0.5
        
    # ---- D. Evaluation on Test Set -----------------------------------------
    if not test_meta_clean.empty:
        X_test_meta = test_meta_clean[META_FEATURES]
        y_test_meta = test_meta_clean['meta_label'].astype(int)
        
        test_probs = calibrated_meta.predict_proba(X_test_meta)[:, 1]
        test_ece   = compute_ece(test_probs, y_test_meta.values)
        
        eval_metrics = evaluate_momentum(calibrated_meta, X_test_meta, y_test_meta)
        logger.info(f"  [Test Performance] AUC: {eval_metrics['auc']:.4f}, Brier: {eval_metrics['brier']:.4f}, ECE: {test_ece*100:.2f}%")
    else:
        logger.warning("  Test set empty for evaluation.")
        
    # ---- E. Save model and calibration parameters --------------------------
    regime_dir = output_dir / regime_name
    regime_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(prim_model, regime_dir / 'primary_model.pkl')
    joblib.dump(prim_model, regime_dir / 'momentum_model.pkl')
    joblib.dump(calibrated_meta, regime_dir / 'calibrated_meta_model.pkl')
    joblib.dump(calibrated_meta, regime_dir / 'meta_model.pkl')
    logger.info(f"  Saved meta-ensemble models to {regime_dir}")
    
    return {
        'primary_model': prim_model,
        'calibrated_meta': calibrated_meta,
        'ece': ece_cal,
        'kelly_fraction': calibrated_meta.kelly_fraction
    }


def train_regime_meta_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str
) -> pd.DataFrame:
    """
    Orchestrator to train regime-specific Meta-Ensembles and generate full predictions.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    regimes = ["trending_low_vol", "trending_high_vol", "sideways_low_vol", "crash_mode"]
    models = {}
    
    # Train each regime meta-ensemble
    for r in regimes:
        res = train_single_regime_ensemble(r, train_df, val_df, test_df, out_path)
        if res:
            models[r] = res

    train_regime_meta_ensemble.last_results = models

    # Generate predictions across ALL records using the correct regime-routed model
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pred_dir  = np.zeros(len(df))
        mom_prob  = np.zeros(len(df))
        meta_marg = np.zeros(len(df))
        
        # Group by regime to make routed inference
        for r in regimes:
            idx = df[df['regime_label'] == r].index
            if len(idx) == 0:
                continue
                
            if r in models:
                prim = models[r]['primary_model']
                meta = models[r]['calibrated_meta']
                
                X_prim = df.loc[idx, MOMENTUM_FEATURES]
                X_meta = df.loc[idx, META_FEATURES]
                
                # Direction (1=Long, -1=Short)
                prim_preds = prim.predict(X_prim)
                pred_dir[idx] = np.where(prim_preds == 1, 1, -1)
                
                # Confidence
                mom_prob[idx]  = meta.predict_proba(X_meta)[:, 1]
                meta_marg[idx] = meta.predict(X_meta, output_margin=True)
            else:
                # Fallback for skipped or un-trained regimes (e.g. choppy_high_vol / fallback flat)
                pred_dir[idx]  = 0.0
                mom_prob[idx]  = 0.0
                meta_marg[idx] = -999.0
                
        df['predicted_direction'] = pred_dir
        df['momentum_probability'] = mom_prob
        df['meta_margin'] = meta_marg
        
    return train_df, val_df, test_df
