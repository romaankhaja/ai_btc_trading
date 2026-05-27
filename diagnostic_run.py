import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.model_ensemble import ModelEnsemble
from training.config import MIN_CONFIDENCE, MOMENTUM_FEATURES, MOMENTUM_HORIZON_BARS, NO_TRADE_REGIMES
from training.evaluate import get_feature_importance


def _safe_bucket_counts(series):
    return series.value_counts().sort_index().to_dict()


def main():
    print("Loading test data...")
    data_dir = PROJECT_ROOT / "data" / "labeled" / "BTCUSDT"
    test_df = pd.read_parquet(data_dir / 'test.parquet')
    
    print("Loading models...")
    models_dir = PROJECT_ROOT / "models"
    ensemble = ModelEnsemble(str(models_dir))
    ensemble.load()

    # Get XGBoost feature importances
    try:
        fi = get_feature_importance(ensemble.momentum_model, MOMENTUM_FEATURES)
        print("\n--- TOP 5 XGBOOST FEATURES (global momentum model) ---")
        for _, row in fi.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']}")
    except Exception as e:
        print(f"Could not get feature importances: {e}")

    print("\nRunning inference...")
    results = []
    
    # Match the configured momentum horizon used in labeling and execution.
    horizon = MOMENTUM_HORIZON_BARS
    close_col = 'mark_close' if 'mark_close' in test_df.columns else 'close'
    closes = test_df[close_col].values
    timestamps = test_df['open_time'].values if 'open_time' in test_df.columns else test_df.index.values

    # Pre-calculate features to dictionary list for speed
    columns = list(test_df.columns)
    
    for i in range(len(test_df) - horizon):
        if i % 1000 == 0:
            print(f"Processed {i} / {len(test_df)} bars")
            
        row = test_df.iloc[i]
        features = {col: float(row[col]) if isinstance(row[col], (int, float, np.integer, np.floating)) else row[col] for col in columns}
        
        output = ensemble.predict(features)
        
        # Calculate 16-bar forward return
        current_close = closes[i]
        future_close = closes[i + horizon]
        actual_price_change = future_close - current_close
        
        predicted_dir = output.predicted_direction # 1 for LONG, -1 for SHORT
        
        # If predicted_direction == 1, correct if change > 0
        # If predicted_direction == -1, correct if change < 0
        if predicted_dir == 1 and actual_price_change > 0:
            correct = True
        elif predicted_dir == -1 and actual_price_change < 0:
            correct = True
        else:
            correct = False
            
        results.append({
            'timestamp': timestamps[i],
            'predicted_direction': predicted_dir,
            'actual_price_change_16bars': actual_price_change,
            'was_prediction_correct': correct,
            'regime': output.regime_label,
            'meta_probability': output.meta_probability,
            'directional_confidence': (
                output.meta_probability if predicted_dir == 1 else 1.0 - output.meta_probability
            ),
        })

    res_df = pd.DataFrame(results)
    
    # Overall Accuracy
    acc_overall = res_df['was_prediction_correct'].mean()
    print(f"\n--- RAW DIRECTIONAL ACCURACY OVERALL ---")
    print(f"Accuracy: {acc_overall:.2%}")
    print(f"Total evaluated bars: {len(res_df)}")

    actionable = res_df[
        (res_df['directional_confidence'] >= MIN_CONFIDENCE)
        & (~res_df['regime'].isin(NO_TRADE_REGIMES))
    ]
    if not actionable.empty:
        action_acc = actionable['was_prediction_correct'].mean()
        print(f"\n--- ACTIONABLE DIRECTIONAL ACCURACY ---")
        print(f"Accuracy: {action_acc:.2%}")
        print(f"Bars above confidence and regime gates: {len(actionable)}")

    # Accuracy per regime
    print(f"\n--- RAW DIRECTIONAL ACCURACY PER REGIME ---")
    regime_acc = res_df.groupby('regime')['was_prediction_correct'].mean()
    regime_counts = res_df.groupby('regime').size()
    for reg in regime_acc.index:
        print(f"  {reg}: {regime_acc[reg]:.2%} ({regime_counts[reg]} bars)")

    # Probability Distribution Buckets
    print(f"\n--- PROBABILITY DISTRIBUTION BUCKETS ---")
    bins = [0.0, 0.50, 0.55, 0.60, 0.65, 1.0]
    labels = ['<0.50', '0.50-0.55', '0.55-0.60', '0.60-0.65', '0.65+']
    res_df['prob_bucket'] = pd.cut(res_df['meta_probability'], bins=bins, labels=labels)
    bucket_counts = _safe_bucket_counts(res_df['prob_bucket'])
    for bucket in labels:
        count = bucket_counts.get(bucket, 0)
        print(f"  {bucket}: {count} bars")

    pred_sign = (res_df['predicted_direction'] == 1).astype(int)
    actual_sign = (res_df['actual_price_change_16bars'] > 0).astype(int)
    tp = int(((pred_sign == 1) & (actual_sign == 1)).sum())
    fp = int(((pred_sign == 1) & (actual_sign == 0)).sum())
    tn = int(((pred_sign == 0) & (actual_sign == 0)).sum())
    fn = int(((pred_sign == 0) & (actual_sign == 1)).sum())

    print(f"\n--- CONFUSION MATRIX ---")
    print(f"  TP: {tp}")
    print(f"  FP: {fp}")
    print(f"  TN: {tn}")
    print(f"  FN: {fn}")

if __name__ == '__main__':
    main()
