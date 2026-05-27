"""
Backtest Validation — Phase 4 (aligned with Phase 5 policy logic).

Key fixes over the original:
  1. Uses ADAPTIVE thresholds (percentile-based) — not a fixed 0.6 cutoff.
  2. Separates a "calibration window" from the test window so thresholds
     are always fitted on data the backtest has already seen (no lookahead).
  3. Reports honest per-regime stats including win-rate and edge.
  4. Flags the Phase 4 vs Phase 5 gap by running BOTH threshold modes
     side-by-side so you can see exactly how much of the gap is threshold
     logic vs model quality.
  5. Adds a basic Kelly-fraction sizing check.
"""

import logging
import numpy as np
import pandas as pd

from training.config import (
    MOMENTUM_SL_PCT,
    MOMENTUM_TP_PCT,
    NO_TRADE_REGIMES,
    THRESHOLDS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Percentile thresholds — must match Phase 5 AdaptiveThresholdEngine exactly
# ---------------------------------------------------------------------------
REGIME_PERCENTILE_THRESHOLDS = {
    'trending_up':   0.55,
    'trending_down': 0.55,
    'mixed':         0.75,
    'ranging':       0.90,
    'unknown':       0.80,
}

BARS_PER_YEAR = 35_040          # 15-min candles: 4 * 24 * 365
INITIAL_EQUITY = 10_000.0
RISK_PER_TRADE = 0.01           # 1 % of equity risked per trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_percentile_threshold(probs: np.ndarray, regime: str) -> float:
    """Return the probability value at the regime's target percentile."""
    target = REGIME_PERCENTILE_THRESHOLDS.get(regime, 0.80)
    return float(np.percentile(probs[~np.isnan(probs)], target * 100))


def _sharpe(returns: pd.Series) -> float:
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(BARS_PER_YEAR))


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.expanding(min_periods=1).max()
    return float((equity / peak - 1).min())


def _kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Full Kelly fraction (cap at 0.25 for safety)."""
    if avg_loss == 0:
        return 0.0
    b = avg_win / avg_loss
    f = (b * win_rate - (1 - win_rate)) / b
    return min(max(f, 0.0), 0.25)


# ---------------------------------------------------------------------------
# Single-pass simulation
# ---------------------------------------------------------------------------

def _simulate(
    df: pd.DataFrame,
    use_adaptive: bool,
    calib_probs: np.ndarray | None = None,
) -> dict:
    """
    Simulate one pass over df.

    Parameters
    ----------
    df           : test DataFrame with model outputs already present.
    use_adaptive : if True, compute per-regime adaptive thresholds from
                   calib_probs; if False, use fixed THRESHOLDS['momentum_action_threshold'].
    calib_probs  : training-set momentum probabilities needed for adaptive mode.
    """
    if use_adaptive and calib_probs is None:
        raise ValueError("calib_probs required when use_adaptive=True")

    results = []

    for _, row in df.iterrows():
        regime = row.get('regime_label', 'unknown')
        prob   = float(row.get('momentum_probability', 0.0))
        label  = int(row.get('label_momentum', 0))
        risk   = row.get('label_risk', 'MEDIUM_RISK')
        behav  = int(row.get('label_behavioral', 0))

        # --- hard blocks (same as Phase 5) ---
        if risk == 'NO_TRADE':
            continue
        if behav == 1:
            continue
        if regime in NO_TRADE_REGIMES:
            continue

        # --- threshold ---
        if use_adaptive:
            threshold = _compute_percentile_threshold(calib_probs, regime)
        else:
            threshold = THRESHOLDS['momentum_action_threshold']   # 0.6 fixed

        if prob < threshold:
            continue

        # --- PnL (same mechanics as Phase 4 original) ---
        pnl = MOMENTUM_TP_PCT if label == 1 else -MOMENTUM_SL_PCT
        results.append({
            'regime':    regime,
            'prob':      prob,
            'label':     label,
            'threshold': threshold,
            'pnl':       pnl,
        })

    if not results:
        return _empty_result()

    trades = pd.DataFrame(results)
    equity = (1 + trades['pnl']).cumprod() * INITIAL_EQUITY

    wins      = (trades['pnl'] > 0).sum()
    win_rate  = wins / len(trades)
    avg_win   = trades.loc[trades['pnl'] > 0, 'pnl'].mean() if wins > 0 else 0.0
    avg_loss  = abs(trades.loc[trades['pnl'] < 0, 'pnl'].mean()) if (len(trades) - wins) > 0 else MOMENTUM_SL_PCT

    regime_stats = {}
    for r in trades['regime'].unique():
        r_df = trades[trades['regime'] == r]
        r_wr = (r_df['pnl'] > 0).mean()
        r_ret = r_df['pnl'].sum()
        regime_stats[r] = {
            'trades':    len(r_df),
            'win_rate':  r_wr,
            'total_ret': r_ret,
            'avg_prob':  r_df['prob'].mean(),
            'threshold': r_df['threshold'].iloc[0],
        }

    return {
        'n_trades':      len(trades),
        'trade_freq':    len(trades) / len(df),
        'total_return':  float(equity.iloc[-1] / INITIAL_EQUITY - 1),
        'sharpe':        _sharpe(trades['pnl']),
        'max_drawdown':  _max_drawdown(equity),
        'win_rate':      win_rate,
        'avg_win':       avg_win,
        'avg_loss':      avg_loss,
        'kelly':         _kelly_fraction(win_rate, avg_win, avg_loss),
        'regime_stats':  regime_stats,
    }


def _empty_result() -> dict:
    return {
        'n_trades': 0, 'trade_freq': 0.0, 'total_return': 0.0,
        'sharpe': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0,
        'avg_win': 0.0, 'avg_loss': 0.0, 'kelly': 0.0,
        'regime_stats': {},
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_backtest_validation(test_df: pd.DataFrame, train_df: pd.DataFrame | None = None) -> dict:
    """
    Run backtest validation on test_df.

    Parameters
    ----------
    test_df  : labeled test set with model output columns present.
    train_df : labeled training set — used to calibrate adaptive thresholds.
               If None, adaptive mode is skipped.

    Returns
    -------
    dict with keys 'fixed' and optionally 'adaptive', each containing
    the metrics dict from _simulate().
    """
    logger.info("=" * 50)
    logger.info("BACKTEST VALIDATION (TEST SET)")
    logger.info("=" * 50)

    if 'momentum_probability' not in test_df.columns:
        logger.warning("No model outputs found in test set. Skipping backtest.")
        return {}

    output = {}

    # ── Mode 1 : fixed threshold (original Phase 4 logic) ──────────────────
    fixed = _simulate(test_df, use_adaptive=False)
    output['fixed'] = fixed

    _log_result("FIXED THRESHOLD (original Phase 4 logic)", fixed, test_df, THRESHOLDS.get('backtest_sharpe_min', 0.5))

    # ── Mode 2 : adaptive threshold (Phase 5 logic) ────────────────────────
    if train_df is not None and 'momentum_probability' in train_df.columns:
        calib_probs = train_df['momentum_probability'].dropna().values
        adaptive = _simulate(test_df, use_adaptive=True, calib_probs=calib_probs)
        output['adaptive'] = adaptive

        _log_result("ADAPTIVE THRESHOLD (Phase 5 policy logic)", adaptive, test_df, THRESHOLDS.get('backtest_sharpe_min', 0.5))

        # ── Gap analysis ──────────────────────────────────────────────────
        logger.info("  GAP ANALYSIS: Fixed vs Adaptive")
        logger.info(f"    Trade count : {fixed['n_trades']:4d} fixed  →  {adaptive['n_trades']:4d} adaptive")
        logger.info(f"    Total return: {fixed['total_return']*100:+6.2f}%  →  {adaptive['total_return']*100:+6.2f}%")
        logger.info(f"    Sharpe      : {fixed['sharpe']:+6.2f}         →  {adaptive['sharpe']:+6.2f}")
    else:
        logger.info("  Skipping adaptive mode (no train_df supplied).")

    return output


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_result(title: str, r: dict, df: pd.DataFrame, sharpe_min: float):
    n = len(df)
    logger.info(f"\n  [{title}]")
    logger.info(f"    Total Trades  : {r['n_trades']} ({r['trade_freq']*100:.1f}% of bars)")
    logger.info(f"    Total Return  : {r['total_return']*100:.2f}%")
    logger.info(f"    Sharpe Ratio  : {r['sharpe']:.2f}  (min: {sharpe_min})")
    logger.info(f"    Max Drawdown  : {r['max_drawdown']*100:.2f}%")
    logger.info(f"    Win Rate      : {r['win_rate']*100:.1f}%")
    logger.info(f"    Kelly Fraction: {r['kelly']:.3f}")

    if r['n_trades'] == 0:
        logger.info("    !! Zero trades — all bars blocked or below threshold !!")
        return

    logger.info("    Performance by Regime:")
    for regime, s in r['regime_stats'].items():
        logger.info(
            f"      {regime:18s}: {s['trades']:4d} trades | "
            f"WR: {s['win_rate']*100:.0f}% | "
            f"Ret: {s['total_ret']*100:+6.2f}% | "
            f"AvgProb: {s['avg_prob']:.3f} | "
            f"Threshold: {s['threshold']:.3f}"
        )