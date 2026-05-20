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
    """Fail loudly if label NaN ratios are too high after removing choppy rows."""
    usable = df[df['regime_label'] != 'choppy_high_vol'].copy()
    if usable.empty:
        raise AssertionError(f"{name}: no usable rows after dropping choppy_high_vol")

    for col in ['label_momentum', 'label_meta']:
        if col not in usable.columns:
            raise AssertionError(f"{name}: missing required label column {col}")
        nan_ratio = usable[col].isna().mean()
        if nan_ratio > 0.05:
            raise AssertionError(
                f"{name}: {col} NaN ratio {nan_ratio:.2%} exceeds 5% threshold after dropping choppy_high_vol"
            )


def _compute_signal_distribution(val_df: pd.DataFrame, test_df: pd.DataFrame, models_dir: Path) -> dict:
    """Run the model across the full test set and summarize raw action counts."""
    diagnostic_engine = TradeDecisionEngine(str(models_dir))
    diagnostic_engine.load()

    atr_history = []
    for _, row in val_df.iterrows():
        features = {col: row[col] for col in val_df.columns}
        atr_history.append(float(row.get('atr_14', 0.0)))
        features['atr_20bar_mean'] = float(np.mean(atr_history[-20:])) if atr_history else 0.0
        out = diagnostic_engine.ensemble.predict(features)
        diagnostic_engine.margin_window.append(out.meta_margin)

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
    scaler, hmm_model, mapping = fit_hmm_model(train_df, n_components=4)
    
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
    
    # ---- 5. Paper Trading (Z-Score + Kelly) ----
    print("\nSTEP 5: PAPER TRADING (Z-Score Ranking + Kelly Sizing)")
    
    trader = PaperTrader(
        models_dir=str(models_dir),
        initial_equity=10000.0
    )
    
    # Re-warm margin window with validation set to get valid Z-scores at test start
    print("  Warming up Z-Score window with validation set...")
    trader.load()
    for _, row in val_df.iterrows():
        features = {col: row[col] for col in val_df.columns}
        out = trader.engine.ensemble.predict(features)
        trader.engine.margin_window.append(out.meta_margin)
        
    print("  Running simulation on test set...")
    result = trader.run(test_df)

    if result.trades and any(t.confidence < MIN_CONFIDENCE for t in result.trades):
        raise ValueError(f"Confidence gate failed: found trade below MIN_CONFIDENCE={MIN_CONFIDENCE:.2f}")

    # ---- 6. Results ----
    print("\n" + "=" * 60)
    print("STEP 6: RESULTS")
    print("=" * 60)
    
    print(f"\n  PERFORMANCE METRICS:")
    print(f"    Total Return:     {result.total_return * 100:.2f}%")
    print(f"    Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"    Sortino Ratio:    {result.sortino_ratio:.2f}")
    print(f"    Max Drawdown:     {result.max_drawdown * 100:.2f}%")
    print(f"    Total Trades:     {result.total_trades}")
    
    if result.total_trades > 0:
        freq = result.total_trades / len(test_df)
        print(f"    Trade Frequency:  {freq:.1%} of bars")
        print(f"    Win Rate:         {result.win_rate:.1%}")
    print(f"    Avg RR Realized:  {result.avg_rr_realized:.2f}")
    
    print(f"\n  REGIME BREAKDOWN:")
    for reg, stats in result.regime_performance.items():
        if stats['trades'] > 0:
            wr = (stats['wins'] / stats['trades']) * 100
            print(f"    {reg:<20}: {stats['trades']:>3} trades | PnL: ${stats['pnl']:>7.2f} | WR: {wr:.0f}%")
            
    # Save trades to CSV
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    if result.trades:
        trade_data = []
        for t in result.trades:
            trade_data.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': 'LONG' if t.direction == 1 else 'SHORT',
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'position_size_usd': t.position_size_usd,
                'pnl': t.pnl,
                'regime': t.regime,
                'confidence': t.confidence,
                'exit_reason': t.exit_reason
            })
        trades_df = pd.DataFrame(trade_data)
    else:
        trades_df = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price',
            'position_size_usd', 'pnl', 'regime', 'confidence', 'exit_reason'
        ])
    trades_file = results_dir / "test_trades.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"\n  Saved {len(result.trades)} detailed trades to {trades_file}")

    ece_by_model = {
        regime: float(stats.get('ece', 0.0))
        for regime, stats in getattr(train_regime_meta_ensemble, 'last_results', {}).items()
    }
    total_pnl = float(sum(t.pnl for t in result.trades)) if result.trades else 0.0
    avg_trade_pnl = float(np.mean([t.pnl for t in result.trades])) if result.trades else 0.0
    signal_distribution = _compute_signal_distribution(val_df, test_df, models_dir)

    performance_report = {
        'total_trades': int(result.total_trades),
        'win_rate': float(result.win_rate),
        'sharpe_ratio': float(result.sharpe_ratio),
        'max_drawdown': float(result.max_drawdown),
        'total_pnl': total_pnl,
        'avg_trade_pnl': avg_trade_pnl,
        'ece_by_model': ece_by_model,
        'circuit_breaker_activations': int(result.circuit_breaker_activations),
        'signal_distribution': signal_distribution,
        'regime_override_count': int(result.regime_override_count),
    }

    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'performance_report.json', 'w', encoding='utf-8') as f:
        json.dump(performance_report, f, indent=2, default=float)
    print(f"\n  Saved performance report to {results_dir / 'performance_report.json'}")

    # The pipeline is considered complete once artifacts are written,
    # even if the realized Sharpe is negative.
    sys.exit(0)

if __name__ == "__main__":
    main()
