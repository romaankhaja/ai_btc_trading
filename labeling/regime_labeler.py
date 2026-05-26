"""
Regime Labeler — Unsupervised KMeans Clustering on normalized features.

Fits KMeans ONLY on training data to prevent leakage.
Assigns regime_state and regime_label to all splits.
Saves scaler + kmeans model to models/regime/.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

# Per roadmap Section 5.2 — cluster ONLY on normalized ratio/percentile features
REGIME_FEATURES = [
    'atr_expansion_ratio',
    'volatility_percentile',
    'ema_20_slope',
    'trend_alignment_score',
    'volume_ratio',
    'bb_width_percentile'
]

DEFAULT_N_CLUSTERS = 3


def auto_map_clusters(km, scaler, feature_names):
    """
    Automatically map cluster IDs to regime names based on centroid characteristics.
    
    Logic:
    - Low abs(ema_20_slope) + low volatility -> ranging
    - Positive ema_20_slope + high volatility -> trending_up
    - Negative ema_20_slope + high volatility -> trending_down
    - Catch-all -> mixed
    """
    # Get centroids in original feature space
    centroids_scaled = km.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    
    centroid_df = pd.DataFrame(centroids, columns=feature_names)
    
    mapping = {}
    for cluster_id, row in centroid_df.iterrows():
        ema_slope = float(row['ema_20_slope'])
        vol_pct = float(row['volatility_percentile'])

        if abs(ema_slope) < 0.05 and vol_pct < 0.4:
            mapping[cluster_id] = 'ranging'
        elif ema_slope > 0.05 and vol_pct >= 0.4:
            mapping[cluster_id] = 'trending_up'
        elif ema_slope < -0.05 and vol_pct >= 0.4:
            mapping[cluster_id] = 'trending_down'
        else:
            mapping[cluster_id] = 'mixed'

    # Ensure regime names remain unique if more than one cluster lands
    # in the same semantic bucket. Keep the best-matching centroid on the
    # base label and suffix lower-confidence duplicates.
    def _confidence_score(cluster_id, name):
        row = centroid_df.loc[cluster_id]
        slope = float(row['ema_20_slope'])
        vol = float(row['volatility_percentile'])
        if name == 'ranging':
            return max(0.0, 0.4 - vol) + max(0.0, 0.05 - abs(slope))
        if name == 'trending_up':
            return max(0.0, slope - 0.05) + max(0.0, vol - 0.4)
        if name == 'trending_down':
            return max(0.0, -slope - 0.05) + max(0.0, vol - 0.4)
        return 0.0

    by_name = {}
    for cluster_id, name in mapping.items():
        by_name.setdefault(name, []).append(cluster_id)

    unique_mapping = {}
    for name, cluster_ids in by_name.items():
        if len(cluster_ids) == 1:
            unique_mapping[cluster_ids[0]] = name
            continue

        ranked = sorted(
            cluster_ids,
            key=lambda cid: (_confidence_score(cid, name), -cid),
            reverse=True,
        )
        unique_mapping[ranked[0]] = name
        for idx, cluster_id in enumerate(ranked[1:], start=1):
            unique_mapping[cluster_id] = f"{name}_{idx}"

    mapping = unique_mapping
    
    return mapping, centroid_df


def fit_regime_model(train_df, n_clusters=DEFAULT_N_CLUSTERS, random_state=42):
    """
    Fit KMeans on training data ONLY.
    
    Returns:
        scaler, kmeans model, cluster-to-name mapping, centroid DataFrame
    """
    X_train = train_df[REGIME_FEATURES].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
        max_iter=500
    )
    km.fit(X_scaled)
    
    # Evaluate cluster quality
    sil_score = silhouette_score(X_scaled, km.labels_, sample_size=min(5000, len(X_scaled)))
    logger.info(f"Regime KMeans silhouette score: {sil_score:.4f}")
    
    # Auto-map clusters to regime names
    mapping, centroid_df = auto_map_clusters(km, scaler, REGIME_FEATURES)
    
    return scaler, km, mapping, centroid_df, sil_score


def assign_regime_labels(df, scaler, km, mapping):
    """
    Assign regime_state and regime_label to a DataFrame.
    Uses scaler.transform() (NOT fit_transform) to prevent leakage.
    """
    X = df[REGIME_FEATURES].values
    X_scaled = scaler.transform(X)
    
    df = df.copy()
    df['regime_state'] = km.predict(X_scaled)
    df['regime_label'] = df['regime_state'].map(mapping)
    
    # Regime confidence = inverse of distance to nearest centroid
    distances = km.transform(X_scaled).min(axis=1)
    df['regime_confidence'] = 1.0 / (1.0 + distances)
    
    return df


def save_regime_model(scaler, km, mapping, output_dir):
    """Save scaler, kmeans, and mapping to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(scaler, output_dir / 'regime_scaler.pkl')
    joblib.dump(km, output_dir / 'regime_kmeans.pkl')
    joblib.dump(mapping, output_dir / 'regime_mapping.pkl')
    
    logger.info(f"Saved regime model to {output_dir}")
