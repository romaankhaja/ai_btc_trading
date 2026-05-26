"""
Momentum Model Training — LightGBM with Isotonic Calibration.

Outputs calibrated momentum_probability [0-1].
Focus: Calibration quality over raw accuracy.
"""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import joblib
from pathlib import Path

from training.config import MOMENTUM_FEATURES, MOMENTUM_PARAMS, LABEL_MOMENTUM, WF_N_SPLITS
from training.walk_forward import walk_forward_cv, summarize_cv_results
from training.evaluate import evaluate_momentum, get_feature_importance

logger = logging.getLogger(__name__)


def _build_momentum_sample_weight(y_train):
    """Return fold-specific sample weights that mirror the class imbalance."""
    y_arr = np.asarray(y_train, dtype=float)
    pos_count = int(np.nansum(y_arr))
    neg_count = int(np.sum(~np.isnan(y_arr)) - pos_count)
    scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0
    return np.where(y_arr == 1, scale_pos_weight, 1.0), scale_pos_weight, pos_count, neg_count


def _train_momentum_model(X_train, y_train, sample_weight=None):
    """Train a single XGBoost + calibration model."""
    if sample_weight is None:
        sample_weight, scale_pos_weight, pos_count, neg_count = _build_momentum_sample_weight(y_train)
    else:
        _, scale_pos_weight, pos_count, neg_count = _build_momentum_sample_weight(y_train)

    logger.info(
        "  Momentum fold balance: pos=%d neg=%d scale_pos_weight=%.4f",
        pos_count,
        neg_count,
        scale_pos_weight,
    )
    base = xgb.XGBClassifier(**MOMENTUM_PARAMS, scale_pos_weight=scale_pos_weight)
    model = CalibratedClassifierCV(base, method='isotonic', cv=3)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_momentum(train_df, val_df, test_df, output_dir):
    """
    Full momentum training pipeline.
    
    1. Walk-forward CV on training set
    2. Train final model on full training set
    3. Evaluate on val and test
    4. Generate momentum_probability for downstream models
    5. Save model
    
    Returns:
        model, cv_summary, val_metrics, test_metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("MOMENTUM MODEL TRAINING")
    logger.info("=" * 50)
    
    # ---- Walk-Forward CV ----
    logger.info("Walk-Forward Cross-Validation...")
    cv_results = walk_forward_cv(
        train_df, MOMENTUM_FEATURES, LABEL_MOMENTUM,
        train_fn=_train_momentum_model,
        eval_fn=evaluate_momentum,
        n_splits=WF_N_SPLITS,
        fit_params_fn=lambda X_train, y_train, fold=None: {
            'sample_weight': _build_momentum_sample_weight(y_train)[0],
        },
    )
    cv_summary = summarize_cv_results(cv_results)
    
    logger.info(f"  CV AUC: {cv_summary['auc']['mean']:.4f} +/- {cv_summary['auc']['std']:.4f}")
    logger.info(f"  CV Brier: {cv_summary['brier']['mean']:.4f} +/- {cv_summary['brier']['std']:.4f}")
    
    # ---- Train Final Model on Full Training Set ----
    logger.info("Training final model on full training set...")
    valid_train = train_df.dropna(subset=[LABEL_MOMENTUM])
    X_train = valid_train[MOMENTUM_FEATURES]
    y_train = valid_train[LABEL_MOMENTUM]

    final_model = _train_momentum_model(X_train, y_train, sample_weight=_build_momentum_sample_weight(y_train)[0])
    
    # ---- Evaluate on Val/Test ----
    valid_val = val_df.dropna(subset=[LABEL_MOMENTUM])
    valid_test = test_df.dropna(subset=[LABEL_MOMENTUM])
    
    val_metrics = evaluate_momentum(final_model, valid_val[MOMENTUM_FEATURES], valid_val[LABEL_MOMENTUM])
    test_metrics = evaluate_momentum(final_model, valid_test[MOMENTUM_FEATURES], valid_test[LABEL_MOMENTUM])
    
    logger.info(f"  Val AUC: {val_metrics['auc']:.4f} | Brier: {val_metrics['brier']:.4f}")
    logger.info(f"  Test AUC: {test_metrics['auc']:.4f} | Brier: {test_metrics['brier']:.4f}")
    
    # ---- Feature Importance ----
    fi = get_feature_importance(final_model, MOMENTUM_FEATURES)
    if not fi.empty:
        logger.info("  Feature Importance (top 5):")
        for _, row in fi.head(5).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")
    
    # ---- Generate momentum_probability for all splits ----
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        proba = final_model.predict_proba(df[MOMENTUM_FEATURES])[:, 1]
        df['momentum_probability'] = proba
    
    # ---- Save ----
    joblib.dump(final_model, output_dir / 'momentum_model.pkl')
    if not fi.empty:
        fi.to_csv(output_dir / 'momentum_feature_importance.csv', index=False)
    
    logger.info(f"  Saved momentum model to {output_dir}")
    
    return final_model, cv_summary, val_metrics, test_metrics
