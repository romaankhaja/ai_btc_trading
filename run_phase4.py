"""
Phase 4 Orchestrator — Model Training Pipeline.

Trains the 4 downstream models in mandatory order:
1. Momentum -> 2. Volatility -> 3. Risk -> 4. Behavioral

Includes MLflow integration and final backtest validation.

Key change: checks derivatives feature drift BEFORE training momentum
model and disables derivatives features if any exceed PSI threshold.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.drift_detector import DriftDetector
import training.config as cfg
from training.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_MOMENTUM, MLFLOW_EXPERIMENT_VOLATILITY,
    MLFLOW_EXPERIMENT_RISK, MLFLOW_EXPERIMENT_BEHAVIORAL,
    MOMENTUM_PARAMS, VOLATILITY_PARAMS, RISK_PARAMS,
    DERIVATIVES_FEATURES, DERIVATIVES_DRIFT_PSI_LIMIT,
    MOMENTUM_BASE_FEATURES,
)
from training.train_momentum import train_momentum
from training.train_volatility import train_volatility
from training.train_risk import train_risk
from training.train_behavioral import train_behavioral
from training.backtest import run_backtest_validation


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

logger = logging.getLogger(__name__)


def check_derivatives_drift(train_df: pd.DataFrame, val_df: pd.DataFrame) -> bool:
    """
    Fit drift detector on train, check val for derivatives feature drift.

    Returns True if derivatives features are safe to use, False if any
    critical derivative feature exceeds DERIVATIVES_DRIFT_PSI_LIMIT.
    """
    available = [f for f in DERIVATIVES_FEATURES if f in train_df.columns]
    if not available:
        logger.info("  No derivatives features found in dataset — skipping drift check.")
        return False

    detector = DriftDetector()
    detector.fit(train_df, available)
    report = detector.check(val_df, available)

    drifted = [
        f for f in available
        if report.psi_scores.get(f, 0) > DERIVATIVES_DRIFT_PSI_LIMIT
    ]

    if drifted:
        logger.warning(
            "  Derivatives drift detected on val set — disabling for training: %s",
            drifted,
        )
        for f in drifted:
            logger.warning("    %s: PSI=%.4f (limit=%.2f)", f, report.psi_scores[f], DERIVATIVES_DRIFT_PSI_LIMIT)
        return False

    logger.info("  Derivatives features within drift bounds — including in training.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Model Training")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--force-no-derivatives", action="store_true",
        help="Force disable derivatives features regardless of drift check"
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    mlflow.set_tracking_uri((PROJECT_ROOT / MLFLOW_TRACKING_URI).as_uri())

    print("\n" + "#" * 60)
    print("#  ADAPTIVE AI RISK MANAGEMENT SYSTEM")
    print("#  Phase 4: Model Training & Validation")
    print("#" * 60)

    # ── Load labeled data ──────────────────────────────────────────────────
    print("\nLoading labeled data from Phase 3...")
    data_dir = PROJECT_ROOT / "data" / "labeled" / "BTCUSDT"
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df   = pd.read_parquet(data_dir / "val.parquet")
    test_df  = pd.read_parquet(data_dir / "test.parquet")

    out_dir = PROJECT_ROOT / "models"

    # ── Derivatives drift gate ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 0: DERIVATIVES DRIFT CHECK (before training)")
    print("=" * 60)

    if args.force_no_derivatives:
        use_derivatives = False
        print("  Derivatives features DISABLED (--force-no-derivatives flag).")
    else:
        use_derivatives = check_derivatives_drift(train_df, val_df)

    # Patch the live config so all downstream training modules see the change
    cfg.USE_DERIVATIVES_FEATURES = use_derivatives
    cfg.MOMENTUM_FEATURES = (
        MOMENTUM_BASE_FEATURES + DERIVATIVES_FEATURES
        if use_derivatives
        else list(MOMENTUM_BASE_FEATURES)
    )
    print(f"\n  USE_DERIVATIVES_FEATURES = {use_derivatives}")
    print(f"  MOMENTUM_FEATURES ({len(cfg.MOMENTUM_FEATURES)} features): {cfg.MOMENTUM_FEATURES}")

    # ── Momentum Model ─────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT_MOMENTUM)
    with mlflow.start_run(run_name="xgb_calibrated_v1"):
        mlflow.log_params({**MOMENTUM_PARAMS, "use_derivatives": use_derivatives})
        mom_model, mom_cv, mom_val, mom_test = train_momentum(
            train_df, val_df, test_df, out_dir / "momentum"
        )
        mlflow.log_metrics({"val_"  + k: v for k, v in mom_val.items()})
        mlflow.log_metrics({"test_" + k: v for k, v in mom_test.items()})

    # AUC gate — warn loudly but don't abort (let backtest show the damage)
    test_auc = mom_test.get("auc", 0)
    if test_auc < cfg.THRESHOLDS["momentum_auc_min"]:
        logger.warning(
            "  Momentum test AUC %.4f is below minimum %.2f. "
            "Phase 5 results will likely be poor.",
            test_auc, cfg.THRESHOLDS["momentum_auc_min"],
        )

    # ── Volatility Model ───────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT_VOLATILITY)
    with mlflow.start_run(run_name="xgboost_v1"):
        mlflow.log_params(VOLATILITY_PARAMS)
        vol_model, vol_cv, vol_val, vol_test = train_volatility(
            train_df, val_df, test_df, out_dir / "volatility"
        )
        mlflow.log_metrics({"val_"  + k: v for k, v in vol_val.items()})
        mlflow.log_metrics({"test_" + k: v for k, v in vol_test.items()})

    # ── Risk Model ─────────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT_RISK)
    with mlflow.start_run(run_name="xgb_multiclass_v1"):
        mlflow.log_params(RISK_PARAMS)
        risk_model, risk_cv, risk_val, risk_test = train_risk(
            train_df, val_df, test_df, out_dir / "risk"
        )
        mlflow.log_metrics({"val_"  + k: v for k, v in risk_val.items()})
        mlflow.log_metrics({"test_" + k: v for k, v in risk_test.items()})

    # ── Behavioral Model ───────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT_BEHAVIORAL)
    with mlflow.start_run(run_name="iforest_v1"):
        mlflow.log_params({"contamination": "training_label_rate"})
        beh_model, beh_val, beh_test = train_behavioral(
            train_df, val_df, test_df, out_dir / "behavioral"
        )
        mlflow.log_metrics({"val_"  + k: v for k, v in beh_val.items()})
        mlflow.log_metrics({"test_" + k: v for k, v in beh_test.items()})

    # ── Backtest Validation ────────────────────────────────────────────────
    run_backtest_validation(test_df, train_df=train_df)

    print("\n" + "#" * 60)
    print("#  PHASE 4 COMPLETE")
    print("#  All models trained and saved to /models")
    print("#" * 60)


if __name__ == "__main__":
    main()