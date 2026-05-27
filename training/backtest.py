"""
Backtest Validation — Phase 4 (aligned with Phase 5 policy logic).

Key fixes over original:
  1. REGIME_PERCENTILE_THRESHOLDS removed — no longer hardcoded here.
     Thresholds are read from training.config, same source as threshold_engine.py.
     The original hardcoded 0.55 for trending_up was the main reason adaptive
     mode fired 577 trades and lost 62%.
  2. MIN_ABSOLUTE_THRESHOLD imported from threshold_engine — same floor as live.
  3. _compute_percentile_threshold floors the result at MIN_ABSOLUTE_THRESHOLD,
     matching exactly what Phase 5 does.
  4. Calibration window is separated from the test window (no lookahead).
  5. Gap analysis still runs both fixed and adaptive modes for diagnostic clarity.
"""

import logging
import numpy as np
import pandas as pd

from training.config import (
    MOMENTUM_SL_PCT,
    MOMENTUM_TP_PCT,
    NO_TRADE_REGIMES,
    THRESHOLDS,
    REGIME_PERCENTILE_THRESHOLDS,
)
from inference.threshold_engine import MIN_ABSOLUTE_THRESHOLD

logger = logging.getLogger(__name__)

BARS_PER_YEAR  = 35_040        # 15-min candles: 4 × 24 × 365
INITIAL_EQUITY = 10_000.0
RISK_PER_TRADE = 0.01          # 1% of equity risked per trade


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compute_percentile_threshold(probs: np.ndarray, regime: str) -> float:
    """
    Return the probability value at the regime's target percentile,
    floored at MIN_ABSOLUTE_THRESHOLD.

    This must match AdaptiveThresholdEngine.get_threshold() exactly.
    """
    target = REGIME_PERCENTILE_THRESHOLDS.get(regime, 0.90)
    raw = float(np.percentile(probs[~np.isnan(probs)], target * 100))
    return max(raw, MIN_ABSOLUTE_THRESHOLD)


def _sharpe(returns: pd.Series) -> float:
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(BARS_PER_YEAR))


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.expanding(min_periods=1).max()
    return float((equity / peak - 1).min())


def _kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Full Kelly fraction, capped at 0.25."""
    if avg_loss == 0:
        return 0.0
    b = avg_win / avg_loss
    f = (b * win_rate - (1 - win_rate)) / b
    return min(max(f, 0.0), 0.25)


# ── Single-pass simulation ────────────────────────────────────────────────────

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
    use_adaptive : True → percentile thresholds from calib_probs;
                   False → fixed THRESHOLDS['momentum_action_threshold'].
    calib_probs  : training-set momentum probabilities (required when use_adaptive=True).
    """
    if use_adaptive and calib_probs is None:
        raise ValueError("calib_probs required when use_adaptive=True")

    # Pre-compute per-regime thresholds once (not per-row) for adaptive mode
    regime_thresholds = {}
    if use_adaptive:
        for reg in REGIME_PERCENTILE_THRESHOLDS:
            regime_thresholds[reg] = _compute_percentile_threshold(calib_probs, reg)
        # fallback for unknown regimes
        regime_thresholds['unknown'] = _compute_percentile_threshold(calib_probs, 'unknown')

    results = []

    for _, row in df.iterrows():
        regime = row.get('regime_label', 'unknown')
        prob   = float(row.get('momentum_probability', 0.0))
        risk   = row.get('label_risk', 'MEDIUM_RISK')
        behav  = int(row.get('label_behavioral', 0))

        # Skip unlabeled tail rows (NaN label)
        raw_label = row.get('label_momentum', float('nan'))
        if raw_label != raw_label:
            continue
        label = int(raw_label)

        # Hard blocks (same as Phase 5)
        if risk == 'NO_TRADE':
            continue
        if behav == 1:
            continue
        if regime in NO_TRADE_REGIMES:
            continue

        # Threshold
        if use_adaptive:
            threshold = regime_thresholds.get(
                regime,
                regime_thresholds.get('unknown', MIN_ABSOLUTE_THRESHOLD),
            )
        else:
            threshold = THRESHOLDS['momentum_action_threshold']  # 0.60 fixed

        if prob < threshold:
            continue

        # PnL
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

    wins     = (trades['pnl'] > 0).sum()
    win_rate = wins / len(trades)
    avg_win  = trades.loc[trades['pnl'] > 0, 'pnl'].mean()  if wins > 0               else 0.0
    avg_loss = abs(trades.loc[trades['pnl'] < 0, 'pnl'].mean()) if (len(trades) - wins) > 0 else MOMENTUM_SL_PCT

    regime_stats = {}
    for r in trades['regime'].unique():
        r_df = trades[trades['regime'] == r]
        regime_stats[r] = {
            'trades':    len(r_df),
            'win_rate':  float((r_df['pnl'] > 0).mean()),
            'total_ret': float(r_df['pnl'].sum()),
            'avg_prob':  float(r_df['prob'].mean()),
            'threshold': float(r_df['threshold'].iloc[0]),
        }

    return {
        'n_trades':     len(trades),
        'trade_freq':   len(trades) / len(df),
        'total_return': float(equity.iloc[-1] / INITIAL_EQUITY - 1),
        'sharpe':       _sharpe(trades['pnl']),
        'max_drawdown': _max_drawdown(equity),
        'win_rate':     win_rate,
        'avg_win':      avg_win,
        'avg_loss':     avg_loss,
        'kelly':        _kelly_fraction(win_rate, avg_win, avg_loss),
        'regime_stats': regime_stats,
    }


def _empty_result() -> dict:
    return {
        'n_trades': 0, 'trade_freq': 0.0, 'total_return': 0.0,
        'sharpe': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0,
        'avg_win': 0.0, 'avg_loss': 0.0, 'kelly': 0.0,
        'regime_stats': {},
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_backtest_validation(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
) -> dict:
    """
    Run backtest validation on test_df.

    Parameters
    ----------
    test_df  : labeled test set with model output columns present.
    train_df : labeled training set for calibrating adaptive thresholds.
               If None, adaptive mode is skipped.

    Returns
    -------
    dict with keys 'fixed' and optionally 'adaptive'.
    """
    logger.info("=" * 50)
    logger.info("BACKTEST VALIDATION (TEST SET)")
    logger.info("=" * 50)

    if 'momentum_probability' not in test_df.columns:
        logger.warning("No model outputs found in test set. Skipping backtest.")
        return {}

    output = {}
    sharpe_min = THRESHOLDS.get('backtest_sharpe_min', 0.5)

    # Mode 1: fixed threshold (original Phase 4 logic)
    fixed = _simulate(test_df, use_adaptive=False)
    output['fixed'] = fixed
    _log_result("FIXED THRESHOLD (original Phase 4 logic)", fixed, test_df, sharpe_min)

    # Mode 2: adaptive threshold (Phase 5 logic)
    if train_df is not None and 'momentum_probability' in train_df.columns:
        calib_probs = train_df['momentum_probability'].dropna().values
        adaptive = _simulate(test_df, use_adaptive=True, calib_probs=calib_probs)
        output['adaptive'] = adaptive
        _log_result("ADAPTIVE THRESHOLD (Phase 5 policy logic)", adaptive, test_df, sharpe_min)

        # Gap analysis
        logger.info("  GAP ANALYSIS: Fixed vs Adaptive")
        logger.info(
            "    Trade count : %4d fixed  →  %4d adaptive",
            fixed['n_trades'], adaptive['n_trades'],
        )
        logger.info(
            "    Total return: %+6.2f%%  →  %+6.2f%%",
            fixed['total_return'] * 100, adaptive['total_return'] * 100,
        )
        logger.info(
            "    Sharpe      : %+6.2f         →  %+6.2f",
            fixed['sharpe'], adaptive['sharpe'],
        )
    else:
        logger.info("  Skipping adaptive mode (no train_df supplied).")

    return output


# ── Logging helper ────────────────────────────────────────────────────────────

def _log_result(title: str, r: dict, df: pd.DataFrame, sharpe_min: float):
    logger.info("\n  [%s]", title)
    logger.info("    Total Trades  : %d (%.1f%% of bars)", r['n_trades'], r['trade_freq'] * 100)
    logger.info("    Total Return  : %.2f%%", r['total_return'] * 100)
    logger.info("    Sharpe Ratio  : %.2f  (min: %.1f)", r['sharpe'], sharpe_min)
    logger.info("    Max Drawdown  : %.2f%%", r['max_drawdown'] * 100)
    logger.info("    Win Rate      : %.1f%%", r['win_rate'] * 100)
    logger.info("    Kelly Fraction: %.3f", r['kelly'])

    if r['n_trades'] == 0:
        logger.info("    !! Zero trades — all bars blocked or below threshold !!")
        return

    be_winrate = MOMENTUM_SL_PCT / (MOMENTUM_TP_PCT + MOMENTUM_SL_PCT)
    edge = r['win_rate'] - be_winrate
    logger.info(
        "    Break-even WR : %.1f%%  |  Edge: %+.1f%%",
        be_winrate * 100, edge * 100,
    )

    logger.info("    Performance by Regime:")
    for regime, s in r['regime_stats'].items():
        logger.info(
            "      %-18s: %4d trades | WR: %3.0f%% | Ret: %+6.2f%% | "
            "AvgProb: %.3f | Threshold: %.3f",
            regime,
            s['trades'],
            s['win_rate'] * 100,
            s['total_ret'] * 100,
            s['avg_prob'],
            s['threshold'],
        )