import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.model_ensemble import ModelEnsemble

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
        primary_model = ensemble.regime_models['sideways']['primary']
        booster = primary_model.get_booster()
        importance = booster.get_score(importance_type='weight')
        sorted_importances = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("\n--- TOP 10 XGBOOST FEATURES (sideways primary model) ---")
        for feat, score in sorted_importances[:10]:
            print(f"  {feat}: {score}")
    except Exception as e:
        print(f"Could not get feature importances: {e}")

    print("\nRunning inference...")
    results = []
    
    # We need 4 bars forward to check actual price change
    # If using 'close', forward change is close_in_4_bars - close_now
    close_col = 'mark_close' if 'mark_close' in test_df.columns else 'close'
    closes = test_df[close_col].values
    timestamps = test_df['open_time'].values if 'open_time' in test_df.columns else test_df.index.values

    # Pre-calculate features to dictionary list for speed
    columns = list(test_df.columns)
    
    for i in range(len(test_df) - 4):
        if i % 1000 == 0:
            print(f"Processed {i} / {len(test_df)} bars")
            
        row = test_df.iloc[i]
        features = {col: float(row[col]) if isinstance(row[col], (int, float, np.integer, np.floating)) else row[col] for col in columns}
        
        output = ensemble.predict(features)
        
        # Calculate 4-bar forward return
        current_close = closes[i]
        future_close = closes[i+4]
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
            'actual_price_change_4bars': actual_price_change,
            'was_prediction_correct': correct,
            'regime': output.regime_label,
            'meta_probability': output.meta_probability
        })

    res_df = pd.DataFrame(results)
    
    # Overall Accuracy
    acc_overall = res_df['was_prediction_correct'].mean()
    print(f"\n--- RAW DIRECTIONAL ACCURACY OVERALL ---")
    print(f"Accuracy: {acc_overall:.2%}")
    print(f"Total evaluated bars: {len(res_df)}")

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
    bucket_counts = res_df['prob_bucket'].value_counts().sort_index()
    for bucket, count in bucket_counts.items():
        print(f"  {bucket}: {count} bars")

if __name__ == '__main__':
    main()
