"""
Phase 5 Orchestrator — Policy Engine Validation.

Runs the full inference pipeline against the test set
in paper trading mode and validates system behavior.

Key change: checks derivatives feature drift against training set
BEFORE running inference and rebuilds the feature set used for
threshold calibration accordingly. Prints a clear drift summary.
"""

import sys
import argparse
import logging
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


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Policy Engine Validation")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--force-no-derivatives", action="store_true",
        help="Force disable derivatives features regardless of drift check"
    )
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

    # ── Drift Detection ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: DRIFT DETECTION")
    print("=" * 60)

    all_features = list(dict.fromkeys(MOMENTUM_FEATURES + VOLATILITY_FEATURES))
    drift = DriftDetector()
    drift.fit(train_df, all_features)
    report = drift.check(test_df, all_features)

    print(f"  Features checked: {len(report.psi_scores)}")
    print(f"  Features drifted: {len(report.features_drifted)}")
    if report.features_drifted:
        for f in report.features_drifted:
            print(f"    {f}: PSI={report.psi_scores[f]:.4f}")
    if report.retrain_recommended:
        print("  WARNING: Retraining recommended due to significant drift!")
    else:
        print("  Drift within acceptable bounds.")

    # ── Derivatives drift gate ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2b: DERIVATIVES DRIFT GATE")
    print("=" * 60)

    available_deriv = [f for f in DERIVATIVES_FEATURES if f in train_df.columns]

    if args.force_no_derivatives:
        use_derivatives = False
        print("  Derivatives features DISABLED (--force-no-derivatives flag).")
    else:
        drifted_deriv = [
            f for f in available_deriv
            if report.psi_scores.get(f, 0.0) > DERIVATIVES_DRIFT_PSI_LIMIT
        ]
        if drifted_deriv:
            use_derivatives = False
            print(f"  Derivatives drift exceeds PSI limit ({DERIVATIVES_DRIFT_PSI_LIMIT}):")
            for f in drifted_deriv:
                print(f"    {f}: PSI={report.psi_scores[f]:.4f}")
            print("  => Switching to BASE features only for threshold calibration.")
        else:
            use_derivatives = True
            print(f"  Derivatives features within drift bounds — using full feature set.")

    # Patch live config so ModelEnsemble/PaperTrader pick up the right features
    cfg.USE_DERIVATIVES_FEATURES = use_derivatives
    active_momentum_features = (
        MOMENTUM_BASE_FEATURES + DERIVATIVES_FEATURES
        if use_derivatives
        else list(MOMENTUM_BASE_FEATURES)
    )
    cfg.MOMENTUM_FEATURES = active_momentum_features
    print(f"\n  Active momentum features ({len(active_momentum_features)}): {active_momentum_features}")

    # ── Calibrate Threshold Engine ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: CALIBRATE ADAPTIVE THRESHOLD ENGINE")
    print("=" * 60)

    ens = ModelEnsemble(str(PROJECT_ROOT / "models"))
    ens.load()

    # Sample training set for speed (every 5th row)
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

    # Warn if distribution has collapsed (model is confused by drift)
    prob_range = train_probs.max() - train_probs.min()
    if prob_range < 0.2:
        logger.warning(
            "  Probability range is very narrow (%.3f). "
            "Model may be outputting near-random scores. "
            "Consider running with --force-no-derivatives.",
            prob_range,
        )

    # ── Paper Trading ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: PAPER TRADING SIMULATION")
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
    print("STEP 5: RESULTS")
    print("=" * 60)

    print(f"\n  FEATURE MODE: {'WITH derivatives' if use_derivatives else 'BASE only (derivatives disabled due to drift)'}")
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
    print("STEP 6: GOVERNANCE VALIDATION")
    print("=" * 60)

    trending_down_trades = [t for t in result.trades if t.regime == 'trending_down']
    print(f"  Trades in trending_down: {len(trending_down_trades)} (should be 0)")
    print(f"  Max Drawdown: {result.max_drawdown*100:.2f}% (target: < 5%)")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f} (target: > 0.5)")

    if result.total_trades == 0:
        print("\n  NOTE: Zero trades executed. Possible causes:")
        print("    - Calibrated probabilities below adaptive thresholds")
        print("    - Risk model blocking all bars as HIGH_RISK or NO_TRADE")
        print("    - Try running with --force-no-derivatives if drift is present")

    # ── Final verdict ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: SYSTEM VERDICT")
    print("=" * 60)

    passed = (
        result.sharpe_ratio >= 0.5
        and result.max_drawdown >= -0.05
        and len(trending_down_trades) == 0
    )
    if passed:
        print("  [PASS] System meets governance targets. Ready for live paper trading.")
    else:
        reasons = []
        if result.sharpe_ratio < 0.5:
            reasons.append(f"Sharpe {result.sharpe_ratio:.2f} < 0.5")
        if result.max_drawdown < -0.05:
            reasons.append(f"MaxDD {result.max_drawdown*100:.1f}% > -5%")
        if trending_down_trades:
            reasons.append(f"{len(trending_down_trades)} trades in trending_down")
        print(f"  [FAIL] Does not meet targets: {', '.join(reasons)}")
        if not use_derivatives:
            print("  Derivatives already disabled. Next step: review momentum labeler or feature set.")
        else:
            print("  Try re-running with: python run_phase5.py --force-no-derivatives")

    print("\n" + "#" * 60)
    print("#  PHASE 5 COMPLETE")
    print("#  Policy Engine validated against test set")
    print("#" * 60)


if __name__ == "__main__":
    main()