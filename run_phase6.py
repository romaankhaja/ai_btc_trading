"""
RUN PHASE 6: Institutional Redesign
Executes HMM Regime Detection, Regime-Specialised Meta-Ensemble Training, and Z-Score Execution.
"""

import sys
import logging
import json
import numpy as np
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

from training.train_hmm_regime import fit_hmm_model, assign_hmm_labels, save_hmm_model
from training.train_regime_xgboost import train_regime_meta_ensemble
from training.train_volatility import train_volatility
from training.train_risk import train_risk
from training.train_behavioral import train_behavioral
from execution.paper_trader import PaperTrader
from inference.trade_decision import TradeDecisionEngine
from labeling.meta_labeler import apply_triple_barrier
from labeling.momentum_labeler import label_momentum
from training.config import MIN_CONFIDENCE


def _apply_phase6_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Apply momentum and triple-barrier labels after regime assignment."""
    labeled = apply_triple_barrier(df)
    labeled['label_momentum'] = label_momentum(labeled)
    return labeled


def _assert_label_health(df: pd.DataFrame, name: str) -> None:
    """Fail loudly if label NaN ratios are too high after removing high_risk rows."""
    usable = df[df['regime_label'] != 'high_risk'].copy()
    if usable.empty:
        raise AssertionError(f"{name}: no usable rows after dropping high_risk regime")

    check_cols = ['label_momentum']
    if 'label_meta' in usable.columns:
        check_cols.append('label_meta')

    for col in check_cols:
        nan_ratio = usable[col].isna().mean()
        if nan_ratio > 0.05:
            raise AssertionError(
                f"{name}: {col} NaN ratio {nan_ratio:.2%} exceeds 5% threshold after dropping high_risk"
            )


def _compute_signal_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame, models_dir: Path) -> dict:
    """Run the model across the full test set and summarize raw action counts."""
    diagnostic_engine = TradeDecisionEngine(str(models_dir))
    diagnostic_engine.load()

    if 'momentum_probability' in train_df.columns:
        train_probs = pd.Series(train_df['momentum_probability']).dropna().astype(float).values
    elif 'meta_label' in train_df.columns:
        train_probs = pd.Series(train_df['meta_label']).dropna().astype(float).values
    else:
        train_probs = np.linspace(0.3, 0.7, 1000)

    if len(train_probs) == 0:
        train_probs = np.linspace(0.3, 0.7, 1000)

    diagnostic_engine.fit_thresholds(train_probs)

    counts = {'LONG': 0, 'SHORT': 0, 'NO_TRADE': 0}
    by_regime = {}
    atr_history = []
    for _, row in test_df.iterrows():
        features = {col: row[col] for col in test_df.columns}
        atr_history.append(float(row.get('atr_14', 0.0)))
        features['atr_20bar_mean'] = float(np.mean(atr_history[-20:])) if atr_history else 0.0
        decision = diagnostic_engine.decide(features, equity=10000.0)
        action = decision.action if decision.action in counts else 'NO_TRADE'
        counts[action] += 1

        regime = decision.regime
        if regime not in by_regime:
            by_regime[regime] = {'LONG': 0, 'SHORT': 0, 'NO_TRADE': 0, 'bars': 0}
        by_regime[regime][action] += 1
        by_regime[regime]['bars'] += 1

    total_bars = len(test_df)
    signal_distribution = {
        'total_bars': total_bars,
        'long_signals': counts['LONG'],
        'short_signals': counts['SHORT'],
        'no_trade': counts['NO_TRADE'],
        'long_pct': float(counts['LONG'] / total_bars * 100) if total_bars else 0.0,
        'short_pct': float(counts['SHORT'] / total_bars * 100) if total_bars else 0.0,
        'no_trade_pct': float(counts['NO_TRADE'] / total_bars * 100) if total_bars else 0.0,
        'by_regime': {},
    }

    for regime, regime_counts in by_regime.items():
        bars = regime_counts['bars'] or 1
        signal_distribution['by_regime'][regime] = {
            'bars': regime_counts['bars'],
            'long_signals': regime_counts['LONG'],
            'short_signals': regime_counts['SHORT'],
            'no_trade': regime_counts['NO_TRADE'],
            'long_pct': float(regime_counts['LONG'] / bars * 100),
            'short_pct': float(regime_counts['SHORT'] / bars * 100),
            'no_trade_pct': float(regime_counts['NO_TRADE'] / bars * 100),
        }

    return signal_distribution


def main():
    print("\n" + "#" * 60)
    print("#  ADAPTIVE AI RISK MANAGEMENT SYSTEM")
    print("#  Phase 6: Institutional Overhaul (NHHMM, Regime-Routed Meta-Labeling)")
    print("#" * 60 + "\n")
    
    data_dir = PROJECT_ROOT / "data" / "labeled" / "BTCUSDT"
    models_dir = PROJECT_ROOT / "models"
    
    # ---- 1. Load Data ----
    print("STEP 1: LOAD DATA")
    train_df = pd.read_parquet(data_dir / 'train.parquet')
    val_df = pd.read_parquet(data_dir / 'val.parquet')
    test_df = pd.read_parquet(data_dir / 'test.parquet')
    
    print(f"  Train: {len(train_df)} rows")
    
    # ---- 2. Train HMM Regime Model ----
    print("\nSTEP 2: HMM REGIME DETECTION (Non-Homogeneous HMM)")
    scaler, hmm_model, mapping = fit_hmm_model(train_df, n_components=3)
    
    train_df = assign_hmm_labels(train_df, scaler, hmm_model, mapping)
    val_df = assign_hmm_labels(val_df, scaler, hmm_model, mapping)
    test_df = assign_hmm_labels(test_df, scaler, hmm_model, mapping)

    save_hmm_model(scaler, hmm_model, mapping, models_dir / "regime")

    print("\nSTEP 2B: APPLY LABELING AFTER REGIME ASSIGNMENT")
    train_df = _apply_phase6_labels(train_df)
    val_df = _apply_phase6_labels(val_df)
    test_df = _apply_phase6_labels(test_df)

    _assert_label_health(train_df, 'train')
    _assert_label_health(val_df, 'val')
    _assert_label_health(test_df, 'test')

    data_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(data_dir / 'train.parquet', index=False)
    val_df.to_parquet(data_dir / 'val.parquet', index=False)
    test_df.to_parquet(data_dir / 'test.parquet', index=False)

    # ---- 3. Train Regime-Routed Meta-Ensemble ----
    print("\nSTEP 3: REGIME-ROUTED META-ENSEMBLE TRAINING (Triple-Barrier + Platt Scaling)")
    train_df, val_df, test_df = train_regime_meta_ensemble(
        train_df, val_df, test_df, models_dir / "momentum"
    )
    
    # ---- 4. Train Downstream Models ----
    print("\nSTEP 4: TRAIN DOWNSTREAM MODELS (Vol, Risk, Behavioral)")
    train_volatility(train_df, val_df, test_df, models_dir / "volatility")
    train_risk(train_df, val_df, test_df, models_dir / "risk")
    train_behavioral(train_df, val_df, test_df, models_dir / "behavioral")
    
    # ---- 5. Paper Trading (Adaptive Threshold + Kelly) ----
    print("\nSTEP 5: PAPER TRADING (Adaptive Threshold + Kelly Sizing)")
    print("Skipping paper trading per user instructions until accuracy > 53%.")
    sys.exit(0)

if __name__ == "__main__":
    main()
