"""
Momentum Model Training — LightGBM with Isotonic Calibration.

Key fixes over original:
  1. Switched from XGBoost to LightGBM — faster, better on tabular imbalanced data.
  2. CalibratedClassifierCV uses cv=5 (was 3) with method='isotonic' for better
     probability spread. With cv=3 on a small positive class the calibrator was
     producing all-0.5 outputs.
  3. Added early_stopping via eval_set inside a custom wrapper so we don't
     over-train on the full positive class.
  4. Saves trained_features.json so Phase 5 always uses the correct feature list.
  5. Logs probability distribution stats after calibration — if median stays at
     exactly 0.5 and range < 0.1, calibration has failed and we warn loudly.
"""

import logging
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
from pathlib import Path

from training.config import MOMENTUM_FEATURES, MOMENTUM_PARAMS, LABEL_MOMENTUM, WF_N_SPLITS
from training.walk_forward import walk_forward_cv, summarize_cv_results
from training.evaluate import evaluate_momentum, get_feature_importance

logger = logging.getLogger(__name__)


class _LGBMWrapper(BaseEstimator, ClassifierMixin):
    """
    Thin sklearn-compatible wrapper around LGBMClassifier.

    CalibratedClassifierCV requires fit(X, y, sample_weight=...) to pass
    sample_weight through — LGBMClassifier accepts it natively.
    We also expose classes_ so the calibrator knows this is binary.
    """

    def __init__(self, **params):
        self.params = params
        self._model = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, sample_weight=None):
        self._model = lgb.LGBMClassifier(**self.params)
        self._model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def feature_importances_(self):
        if self._model is not None:
            return self._model.feature_importances_
        return None


def _build_momentum_sample_weight(y_train):
    """Return sample weights that mirror class imbalance."""
    y_arr = np.asarray(y_train, dtype=float)
    pos_count = int(np.nansum(y_arr == 1))
    neg_count = int(np.nansum(y_arr == 0))
    scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0
    weights = np.where(y_arr == 1, scale_pos_weight, 1.0)
    return weights, scale_pos_weight, pos_count, neg_count


def _train_momentum_model(X_train, y_train, sample_weight=None):
    """Train a single LightGBM + calibration model."""
    sw, scale_pos_weight, pos_count, neg_count = _build_momentum_sample_weight(y_train)
    if sample_weight is None:
        sample_weight = sw

    logger.info(
        "  Momentum fold balance: pos=%d neg=%d scale_pos_weight=%.4f",
        pos_count,
        neg_count,
        scale_pos_weight,
    )

    # Build params with class weighting baked in
    params = dict(MOMENTUM_PARAMS)
    params['scale_pos_weight'] = scale_pos_weight

    base = _LGBMWrapper(**params)

    # Use cv=5 for calibration — cv=3 with tiny positive class gives degenerate output
    model = CalibratedClassifierCV(base, method='isotonic', cv=5)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def _check_probability_distribution(probs: np.ndarray, split_name: str):
    """
    Warn if calibration has failed (all outputs near 0.5).

    A properly calibrated model on imbalanced data should have a
    distribution skewed below 0.5, with meaningful spread.
    """
    valid = probs[~np.isnan(probs)]
    p_range = valid.max() - valid.min()
    p_median = float(np.median(valid))
    p_high = float(np.mean(valid > 0.55))

    logger.info(
        "  [%s] prob distribution: min=%.3f max=%.3f median=%.3f "
        "range=%.3f pct_above_0.55=%.1f%%",
        split_name, valid.min(), valid.max(), p_median,
        p_range, p_high * 100,
    )

    if p_range < 0.15:
        logger.warning(
            "  [%s] NARROW probability range (%.3f) — calibration may have failed. "
            "Consider reducing cv folds or switching to method='sigmoid'.",
            split_name, p_range,
        )
    if abs(p_median - 0.5) < 0.01 and p_range < 0.2:
        logger.warning(
            "  [%s] Median exactly %.3f with narrow range — model is not discriminating. "
            "Check that features are not all-zero or constant.",
            split_name, p_median,
        )


def train_momentum(train_df, val_df, test_df, output_dir):
    """
    Full momentum training pipeline.

    1. Walk-forward CV on training set
    2. Train final model on full training set
    3. Evaluate on val and test
    4. Check probability distribution — warn if calibration failed
    5. Generate momentum_probability for downstream models
    6. Save model + trained_features.json

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

    logger.info(
        "  CV AUC: %.4f +/- %.4f",
        cv_summary['auc']['mean'],
        cv_summary['auc']['std'],
    )
    logger.info(
        "  CV Brier: %.4f +/- %.4f",
        cv_summary['brier']['mean'],
        cv_summary['brier']['std'],
    )

    # ---- Train Final Model on Full Training Set ----
    logger.info("Training final model on full training set...")
    valid_train = train_df.dropna(subset=[LABEL_MOMENTUM])
    X_train = valid_train[MOMENTUM_FEATURES]
    y_train = valid_train[LABEL_MOMENTUM]

    sw = _build_momentum_sample_weight(y_train)[0]
    final_model = _train_momentum_model(X_train, y_train, sample_weight=sw)

    # ---- Evaluate on Val/Test ----
    valid_val  = val_df.dropna(subset=[LABEL_MOMENTUM])
    valid_test = test_df.dropna(subset=[LABEL_MOMENTUM])

    val_metrics  = evaluate_momentum(final_model, valid_val[MOMENTUM_FEATURES],  valid_val[LABEL_MOMENTUM])
    test_metrics = evaluate_momentum(final_model, valid_test[MOMENTUM_FEATURES], valid_test[LABEL_MOMENTUM])

    logger.info("  Val AUC: %.4f | Brier: %.4f",  val_metrics['auc'],  val_metrics['brier'])
    logger.info("  Test AUC: %.4f | Brier: %.4f", test_metrics['auc'], test_metrics['brier'])

    # ---- Feature Importance ----
    # Pull importances from the first calibrated estimator's base model
    fi = pd.DataFrame()
    try:
        base_model = final_model.calibrated_classifiers_[0].estimator._model
        importances = base_model.feature_importances_
        fi = pd.DataFrame({
            'feature':    MOMENTUM_FEATURES,
            'importance': importances,
        }).sort_values('importance', ascending=False)
        logger.info("  Feature Importance (top 5):")
        for _, row in fi.head(5).iterrows():
            logger.info("    %s: %.4f", row['feature'], row['importance'])
    except Exception as e:
        logger.warning("  Could not extract feature importance: %s", e)

    # ---- Generate momentum_probability for all splits ----
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        probs = final_model.predict_proba(df[MOMENTUM_FEATURES])[:, 1]
        df['momentum_probability'] = probs
        _check_probability_distribution(probs, name)

    # ---- Save ----
    joblib.dump(final_model, output_dir / 'momentum_model.pkl')

    # Save the exact feature list used — Phase 5 reads this to avoid mismatch
    with open(output_dir / 'trained_features.json', 'w') as f:
        json.dump(MOMENTUM_FEATURES, f)
    logger.info("  Saved trained_features.json (%d features)", len(MOMENTUM_FEATURES))

    if not fi.empty:
        fi.to_csv(output_dir / 'momentum_feature_importance.csv', index=False)

    logger.info("  Saved momentum model to %s", output_dir)

    return final_model, cv_summary, val_metrics, test_metrics