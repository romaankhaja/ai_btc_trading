"""
Training Configuration — Feature lists, hyperparameters, thresholds.

Central configuration for all Phase 4 model training.
All feature lists map to actual column names in our labeled dataset.
"""

# ============================================================
# FEATURE SETS (Minimum Institutional Set)
# ============================================================

MOMENTUM_FEATURES = [
    'ema_20_slope', 'ema_20_slope_5m', 'ema_20_slope_1h',
    'rsi_velocity', 'rsi_velocity_5m',
    'volume_delta', 'atr_expansion_ratio',
    'vwap_distance', 'trade_imbalance',
    'trend_alignment_score'
]

# Meta-model features: primary signal plus orthogonal volatility, liquidity,
# and strategy-health context. Derivatives inputs are intentionally excluded
# until live ingestion is wired end-to-end.
META_FEATURES = MOMENTUM_FEATURES + [
    'atr_14', 'realized_volatility', 'atr_velocity',
    'volatility_percentile', 'bb_width', 'bb_width_percentile',
    'volume_ratio', 'amihud_illiquidity', 'liquidity_pressure_score',
    'volume_spike_score', 'regime_confidence'
]

VOLATILITY_FEATURES = [
    'atr_14', 'atr_expansion_ratio', 'atr_velocity',
    'bb_width', 'bb_width_percentile',
    'volume_ratio', 'amihud_illiquidity', 'realized_volatility'
]

RISK_FEATURES = [
    'regime_state', 'regime_confidence',
    'momentum_probability',       # injected from Model 1
    'predicted_volatility',       # injected from Model 2
    'atr_expansion_ratio', 'volatility_percentile',
    'amihud_illiquidity'
]

BEHAVIORAL_FEATURES = [
    'oversized_trade_score',
    'overtrading_score',
    'strategy_recent_accuracy',
    'strategy_avg_rr',
    'last_5_trade_winrate',
    'strategy_health_score',
    'revenge_trade_score',
    'discipline_score',
    'panic_exit_score',
    'fomo_score'
]

# ============================================================
# LABEL COLUMNS
# ============================================================

LABEL_MOMENTUM = 'label_momentum'
LABEL_VOLATILITY = 'label_volatility'
LABEL_RISK = 'label_risk'
LABEL_BEHAVIORAL = 'label_behavioral'

# Risk label encoding
RISK_CLASSES = ['LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK', 'NO_TRADE']

# ============================================================
# HYPERPARAMETERS
# ============================================================

MOMENTUM_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 5,
    'random_state': 42,
}

VOLATILITY_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
}

RISK_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 5,
    'random_state': 42,
    'objective': 'multi:softprob',
    'num_class': 4,
}

# ============================================================
# VALIDATION THRESHOLDS (Minimum Acceptable)
# ============================================================

THRESHOLDS = {
    'momentum_auc_min': 0.56,
    'momentum_action_threshold': 0.60,
    'volatility_rmse_vs_baseline': 1.0,   # must be < 1.0 (beat EWMA)
    'risk_no_trade_precision_min': 0.70,
    'behavioral_f1_min': 0.65,
    'regime_silhouette_min': 0.30,
    'backtest_sharpe_min': 0.5,
}

# Live execution thresholds and regime routing rules.
MIN_CONFIDENCE = 0.55
MAX_HOLDING_BARS = 12
REGIME_MAX_BARS = {
    'trending_up': MAX_HOLDING_BARS,
    'trending_down': MAX_HOLDING_BARS,
    'mixed': round(MAX_HOLDING_BARS * 0.7),
    'ranging': round(MAX_HOLDING_BARS * 0.5),
}
REGIME_KELLY_MULTIPLIER = {
    'trending_up': 0.5,
    'trending_down': 0.5,
    'mixed': 0.3,
    'ranging': 0.1,
}
NO_TRADE_REGIMES = ['trending_down']

# Critical features that should trigger retraining quickly when they drift.
CRITICAL_FEATURES = [
    'regime_state',
    'volatility_regime',
    'funding_rate_zscore',
    'OI_momentum_4h',
    'atr_14',
    'amihud_illiquidity',
]

# ============================================================
# WALK-FORWARD CV
# ============================================================

WF_N_SPLITS = 5

# ============================================================
# MLFLOW
# ============================================================

MLFLOW_TRACKING_URI = 'mlruns'
MLFLOW_EXPERIMENT_MOMENTUM = 'momentum_v1'
MLFLOW_EXPERIMENT_VOLATILITY = 'volatility_v1'
MLFLOW_EXPERIMENT_RISK = 'risk_v1'
MLFLOW_EXPERIMENT_BEHAVIORAL = 'behavioral_v1'
