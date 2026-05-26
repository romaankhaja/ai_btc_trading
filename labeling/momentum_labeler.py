"""
Momentum Labeler — Forward-looking ATR-multiple TP/SL label.

Binary classification target:
  1 = TP hit before SL within N candles (favorable momentum)
  0 = SL hit first or neither hit (unfavorable)

Features:
- Replaces all OHLCV close/high/low calculations with mark_price, mark_high, mark_low
- Implements regime-conditional barrier multipliers:
  - ranging          : quiet market, use conservative fallback barriers
  - trending_up      : bullish continuation regime
  - trending_down    : bearish continuation regime
  - mixed            : catch-all fallback regime
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def label_momentum(df, n_candles=16):
    """
    Generate momentum labels for the entire DataFrame.
    
    For each candle i:
      - Determine direction from ema_20_slope (>0 = LONG, <0 = SHORT)
      - Get multipliers based on regime_label
      - Set SL = entry +/- atr_14 * atr_mult_sl
      - Set TP = entry +/- atr_14 * atr_mult_tp
      - Look at next n_candles mark price HIGH/LOW to check if TP or SL hit first
      - Label = 1 if TP hit before SL, else 0
    
    Last n_candles rows get NaN (no future data available).
    
    Args:
        df: DataFrame with columns: mark_close/close, mark_high/high, mark_low/low, atr_14, ema_20_slope, regime_label
        n_candles: Forward-looking horizon (default 16 = 4 hours on 15m)
    
    Returns:
        List of labels (1/0/None)
    """
    close_col = 'mark_close' if 'mark_close' in df.columns else 'close'
    high_col = 'mark_high' if 'mark_high' in df.columns else 'high'
    low_col = 'mark_low' if 'mark_low' in df.columns else 'low'
    
    closes = df[close_col].values
    highs = df[high_col].values
    lows = df[low_col].values
    atrs = df['atr_14'].values
    slopes = df['ema_20_slope'].values
    if 'regime_label' in df.columns:
        regimes = df['regime_label'].values
    elif 'regime_state' in df.columns:
        regimes = pd.Series(df['regime_state']).fillna('mixed').astype(str).values
    else:
        regimes = np.full(len(df), 'mixed')
    
    n = len(df)
    labels = np.full(n, np.nan)
    
    for i in range(n - n_candles):
        regime = regimes[i]
        
        # Regime-conditional multipliers
        if regime == 'trending_up':
            atr_mult_tp, atr_mult_sl = 5.0, 2.4
        elif regime == 'trending_down':
            atr_mult_tp, atr_mult_sl = 5.0, 2.4
        elif regime == 'ranging':
            atr_mult_tp, atr_mult_sl = 2.0, 1.6
        else:
            atr_mult_tp, atr_mult_sl = 2.5, 2.0
            
        entry = closes[i]
        atr = atrs[i]
        direction = 1 if slopes[i] > 0 else -1
        
        if np.isnan(atr) or atr <= 0:
            labels[i] = 0.0
            continue
            
        sl = entry - direction * atr * atr_mult_sl
        tp = entry + direction * atr * atr_mult_tp
        
        tp_hit = False
        sl_hit = False
        
        for j in range(i + 1, i + 1 + n_candles):
            if direction == 1:  # LONG
                if lows[j] <= sl:
                    sl_hit = True
                    break
                if highs[j] >= tp:
                    tp_hit = True
                    break
            else:  # SHORT
                if highs[j] >= sl:
                    sl_hit = True
                    break
                if lows[j] <= tp:
                    tp_hit = True
                    break
        
        labels[i] = 1.0 if tp_hit and not sl_hit else 0.0
    
    logger.info(
        f"Momentum labels: {int(np.nansum(labels))} positive / "
        f"{int(np.nansum(labels == 0))} negative / "
        f"{int(np.isnan(labels).sum())} unlabeled (tail/skipped)"
    )
    
    return labels
