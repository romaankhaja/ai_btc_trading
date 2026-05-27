"""
Phase 5 Orchestrator — Policy Engine Validation.

Key fix: the saved momentum model was trained on a specific feature set.
Phase 5 MUST use that same feature set for inference — it cannot switch
to a reduced feature set at runtime on an already-trained model.

The drift gate now does two things:
  1. Warns when derivatives features have drifted.
  2. Writes a flag file so the NEXT Phase 4 run trains without derivatives.

It does NOT change the feature set used for inference in this run.
"""

import sys
import argparse
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from execution.paper_trader import PaperTrader
from monitoring.drift_detector import DriftDetector
from inference.model_ensemble import ModelEnsemble
import training.config as cfg
from training.config import (
    MOMENTUM_FEATURES, MOMENTUM_BASE_FEATURES,
    DERIVATIVES_FEATURES, DERIVATIVES_DRIFT_PSI_LIMIT,
    VOLATILITY_FEATURES,
)


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

logger = logging.getLogger(__name__)

TRAINED_FEATURES_PATH = PROJECT_ROOT / "models" / "momentum" / "trained_features.json"


def load_trained_features() -> list:
    """Load the feature list the saved momentum model was actually trained on."""
    if TRAINED_FEATURES_PATH.exists():
        with open(TRAINED_FEATURES_PATH) as f:
            features = json.load(f)
        logger.info("  Loaded trained features from %s (%d features)", TRAINED_FEATURES_PATH, len(features))
        return features
    else:
        logger.warning(
            "  trained_features.json not found — using config MOMENTUM_FEATURES. "
            "Re-run Phase 4 to generate this file."
        )
        return list(MOMENTUM_FEATURES)


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Policy Engine Validation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    print("\n" + "#" * 60)
    print("#  ADAPTIVE AI RISK MANAGEMENT SYSTEM")
    print("#  Phase 5: Policy Engine & Paper Trading Validation")
    print("#" * 60)

    # ── Load Data ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: LOAD DATA")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "data" / "labeled" / "BTCUSDT"
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df  = pd.read_parquet(data_dir / "test.parquet")

    print(f"  Train: {len(train_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")
    print(f"  Test range: {test_df['open_time'].min()} to {test_df['open_time'].max()}")

    # ── Load trained feature set ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: LOAD TRAINED FEATURE SET")
    print("=" * 60)

    inference_features = load_trained_features()
    print(f"  Inference features ({len(inference_features)}): {inference_features}")

    # Patch config so ModelEnsemble and PaperTrader use the correct features
    cfg.MOMENTUM_FEATURES = inference_features

    # ── Drift Detection ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: DRIFT DETECTION")
    print("=" * 60)

    all_features = list(dict.fromkeys(inference_features + VOLATILITY_FEATURES))
    drift = DriftDetector()
    drift.fit(train_df, all_features)
    report = drift.check(test_df, all_features)

    print(f"  Features checked: {len(report.psi_scores)}")
    print(f"  Features drifted: {len(report.features_drifted)}")
    if report.features_drifted:
        for f in report.features_drifted:
            print(f"    {f}: PSI={report.psi_scores[f]:.4f}")

    available_deriv = [f for f in DERIVATIVES_FEATURES if f in train_df.columns]
    drifted_deriv = [
        f for f in available_deriv
        if report.psi_scores.get(f, 0.0) > DERIVATIVES_DRIFT_PSI_LIMIT
    ]

    print("\n  DERIVATIVES DRIFT ASSESSMENT:")
    flag_path = PROJECT_ROOT / "models" / "disable_derivatives.flag"
    if drifted_deriv:
        print(f"  WARNING: {len(drifted_deriv)} derivatives features drifted (PSI > {DERIVATIVES_DRIFT_PSI_LIMIT}):")
        for f in drifted_deriv:
            print(f"    {f}: PSI={report.psi_scores[f]:.4f}")
        print("  => Writing flag: next Phase 4 run will train WITHOUT derivatives.")
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        flag_path.write_text("derivatives disabled due to drift detected in Phase 5")
    else:
        print("  Derivatives within drift bounds.")
        if flag_path.exists():
            flag_path.unlink()

    if report.retrain_recommended:
        print("\n  WARNING: Retraining recommended due to significant drift!")

    # ── Calibrate Threshold Engine ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: CALIBRATE ADAPTIVE THRESHOLD ENGINE")
    print("=" * 60)

    ens = ModelEnsemble(str(PROJECT_ROOT / "models"))
    ens.load()

    sample_indices = range(0, len(train_df), 5)
    train_probs = []
    for i in sample_indices:
        row = train_df.iloc[i]
        features = {
            col: float(row[col])
            if isinstance(row[col], (int, float, np.integer, np.floating))
            else row[col]
            for col in train_df.columns
        }
        out = ens.predict(features)
        train_probs.append(out.meta_probability)

    train_probs = np.array(train_probs)

    print(f"  Fitted on {len(train_probs)} training samples")
    print(f"  Probability range: [{train_probs.min():.3f}, {train_probs.max():.3f}]")
    print(f"  Median: {np.median(train_probs):.3f}")
    print(f"  P25={np.percentile(train_probs,25):.3f}  P75={np.percentile(train_probs,75):.3f}  P95={np.percentile(train_probs,95):.3f}")

    prob_range = train_probs.max() - train_probs.min()
    if prob_range < 0.3:
        logger.warning(
            "  Narrow probability range (%.3f) — model likely confused by drifted features. "
            "Re-run Phase 4 after drift flag is written.",
            prob_range,
        )

    # ── Paper Trading ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: PAPER TRADING SIMULATION")
    print("=" * 60)

    trader = PaperTrader(
        models_dir=str(PROJECT_ROOT / "models"),
        initial_equity=10000.0,
    )
    trader.load()
    trader.engine.fit_thresholds(train_probs)

    print("  Running simulation on test set...")
    result = trader.run(test_df)

    # ── Results ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: RESULTS")
    print("=" * 60)

    trained_with_deriv = len(inference_features) > len(MOMENTUM_BASE_FEATURES)
    print(f"\n  MODEL TRAINED WITH: {'full features (base + derivatives)' if trained_with_deriv else 'base features only'}")
    print(f"\n  PERFORMANCE METRICS:")
    print(f"    Total Return:     {result.total_return*100:+.2f}%")
    print(f"    Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"    Sortino Ratio:    {result.sortino_ratio:.2f}")
    print(f"    Max Drawdown:     {result.max_drawdown*100:.2f}%")
    print(f"    Total Trades:     {result.total_trades}")
    print(f"    Trade Frequency:  {result.trade_frequency_pct:.1f}% of bars")
    print(f"    Win Rate:         {result.win_rate*100:.1f}%")
    print(f"    Avg RR Realized:  {result.avg_rr_realized:.2f}")

    print(f"\n  REGIME BREAKDOWN:")
    for regime, stats in result.regime_performance.items():
        wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
        print(f"    {regime:20s}: {stats['trades']:3d} trades | PnL: ${stats['pnl']:8.2f} | WR: {wr:.0f}%")

    print(f"\n  BLOCK REASONS (top 10):")
    sorted_blocks = sorted(result.block_reasons_summary.items(), key=lambda x: -x[1])
    for reason, count in sorted_blocks[:10]:
        print(f"    {reason:30s}: {count:,}")

    print(f"\n  TOTAL PNL:")
    total_pnl = sum(trade.pnl for trade in result.trades)
    print(f"    Profit/Loss:      ${total_pnl:+.2f}")
    print(f"    Ending Equity:    ${10000.0 + total_pnl:.2f}")

    # ── Governance Validation ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: GOVERNANCE VALIDATION")
    print("=" * 60)

    trending_down_trades = [t for t in result.trades if t.regime == 'trending_down']
    print(f"  Trades in trending_down: {len(trending_down_trades)} (should be 0)")
    print(f"  Max Drawdown: {result.max_drawdown*100:.2f}% (target: < 5%)")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f} (target: > 0.5)")

    # ── Final verdict ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: SYSTEM VERDICT")
    print("=" * 60)

    passed = (
        result.sharpe_ratio >= 0.5
        and result.max_drawdown >= -0.05
        and len(trending_down_trades) == 0
    )
    if passed:
        print("  [PASS] System meets governance targets.")
    else:
        reasons = []
        if result.sharpe_ratio < 0.5:
            reasons.append(f"Sharpe {result.sharpe_ratio:.2f} < 0.5")
        if result.max_drawdown < -0.05:
            reasons.append(f"MaxDD {result.max_drawdown*100:.1f}% worse than -5%")
        if trending_down_trades:
            reasons.append(f"{len(trending_down_trades)} trades in trending_down")
        print(f"  [FAIL] {', '.join(reasons)}")

        if drifted_deriv and trained_with_deriv:
            print("\n  ROOT CAUSE: Model trained with drifted derivatives features.")
            print("  ACTION: Re-run Phase 4 — derivatives flag is now written, it will train base-only.")
        elif not trained_with_deriv:
            print("\n  Model already base-only. Next step: review feature quality or label horizon.")

    print("\n" + "#" * 60)
    print("#  PHASE 5 COMPLETE")
    print("#" * 60)


if __name__ == "__main__":
    main()