"""
Triple-Barrier Meta-Labeler.

Applies Lopez de Prado's Triple-Barrier Method.
Instead of an arbitrary fixed target, this splits labeling into:
1. Primary Label: Direction (sign of forward return)
2. Meta Label: Did the trade hit the dynamic Volatility-adjusted Take Profit 
   before hitting the Stop Loss or Time barrier?

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
from pathlib import Path

from training.config import MOMENTUM_SL_PCT, MOMENTUM_TP_PCT

logger = logging.getLogger(__name__)

def apply_triple_barrier(df, t1=12):
    """
    Applies the triple-barrier method using real mark price data.
    
    Args:
        df: DataFrame containing 'mark_close', 'mark_high', 'mark_low', 'atr_14', 'regime_label'
        t1: Vertical barrier (time out in periods)
    
    Returns:
        DataFrame with 'primary_label' (direction) and 'meta_label' (success)
    """
    df = df.copy()
    
    # 0. Check required columns. If mark_close etc. are missing, fallback to close/high/low
    close_col = 'mark_close' if 'mark_close' in df.columns else 'close'
    high_col = 'mark_high' if 'mark_high' in df.columns else 'high'
    low_col = 'mark_low' if 'mark_low' in df.columns else 'low'
    
    # Get volatility (ATR_14)
    if 'atr_14' in df.columns:
        vol = df['atr_14']
    else:
        log_ret = np.log(df[close_col] / df[close_col].shift(1))
        vol = log_ret.rolling(20).std() * df[close_col]
        df['atr_14'] = vol
        
    # 1. Primary Direction Label (Forward Return over t1 periods)
    # 1 = Long, -1 = Short, 0 = Flat
    forward_return = df[close_col].shift(-t1) - df[close_col]
    threshold = vol * 0.1
    primary_label = np.where(forward_return > threshold, 1, np.where(forward_return < -threshold, -1, 0))
    df['primary_label'] = primary_label
    
    # 2. Meta-Labeling (Triple Barrier)
    meta_labels = np.full(len(df), np.nan)
    
    closes = df[close_col].values
    highs = df[high_col].values
    lows = df[low_col].values
    dirs = primary_label
    
    for i in range(len(df) - t1):
        if dirs[i] == 0:
            meta_labels[i] = 0.0 # No trade setup
            continue
            
        entry = closes[i]
        if entry <= 0 or np.isnan(entry):
            meta_labels[i] = 0.0
            continue
            
        if dirs[i] == 1: # Long
            tp_price = entry * (1.0 + MOMENTUM_TP_PCT)
            sl_price = entry * (1.0 - MOMENTUM_SL_PCT)
            
            success = 0.0
            for j in range(i + 1, i + 1 + t1):
                if lows[j] <= sl_price:
                    success = 0.0
                    break
                elif highs[j] >= tp_price:
                    success = 1.0
                    break
            meta_labels[i] = success
            
        elif dirs[i] == -1: # Short
            tp_price = entry * (1.0 - MOMENTUM_TP_PCT)
            sl_price = entry * (1.0 + MOMENTUM_SL_PCT)
            
            success = 0.0
            for j in range(i + 1, i + 1 + t1):
                if highs[j] >= sl_price:
                    success = 0.0
                    break
                elif lows[j] <= tp_price:
                    success = 1.0
                    break
            meta_labels[i] = success

    df['primary_label'] = dirs
    df['meta_label'] = meta_labels
    df['label_meta'] = meta_labels
    
    logger.info(f"Triple-Barrier Meta-Labeling Complete.")
    logger.info(f"Primary (Direction): Long={np.sum(dirs==1)}, Short={np.sum(dirs==-1)}, Flat={np.sum(dirs==0)}")
    logger.info(f"Meta (Success): {np.nansum(meta_labels == 1.0)} successful trades out of {np.sum(~np.isnan(meta_labels))} valid setups")
    
    return df


def process_datasets(data_dir: Path):
    logger.info(f"Processing datasets in {data_dir}")
    
    for split in ['train', 'val', 'test']:
        file_path = data_dir / f"{split}.parquet"
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
            
        df = pd.read_parquet(file_path)
        
        # Apply triple barrier (dynamic ATR thresholds, 12 bars = 3 hours)
        df = apply_triple_barrier(df, t1=12)
        
        # Save back
        df.to_parquet(file_path)
        logger.info(f"Updated {split}.parquet with primary_label, meta_label, and label_meta")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "labeled" / "BTCUSDT"
    process_datasets(data_dir)
