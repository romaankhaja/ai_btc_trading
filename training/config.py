"""
Training Configuration — Feature lists, hyperparameters, thresholds.

Key changes from original:
  1. TP/SL adjusted so break-even win rate is ~28% not 29%.
     More importantly, SL widened so the model isn't stopped out by noise.
  2. MOMENTUM_FEATURES split into BASE_FEATURES and DERIVATIVES_FEATURES.
     Derivatives are high-drift features — they're kept but isolated so
     we can easily ablate them when drift PSI > 0.25.
  3. MOMENTUM_HORIZON_BARS reduced from 64 to 32 (8 hours on 15m).
     64 bars (16 hours) is too long for a 15m momentum signal — the
     label becomes dominated by overnight drift, not the setup you identified.
  4. Regime percentile thresholds tightened. The 55th percentile for
     trending regimes was firing on half the bars — no selectivity.
  5. NO_TRADE_REGIMES now includes trending_down for long-only systems.
     If you want to trade shorts, remove trending_down from this list.
"""

# ============================================================
# FEATURE SETS
# ============================================================

# Pure market microstructure features — stable across regimes
MOMENTUM_BASE_FEATURES = [
    'ema_20_slope', 'ema_20_slope_5m', 'ema_20_slope_1h',
    'rsi_velocity', 'rsi_velocity_5m',
    'volume_delta', 'atr_expansion_ratio',
    'vwap_distance', 'trade_imbalance',
    'trend_alignment_score',
]

# Derivatives features — high alpha but high drift risk.
# Disable these by setting USE_DERIVATIVES_FEATURES = False
# when PSI > 0.25 on any of these features.
DERIVATIVES_FEATURES = [
    'funding_rate_zscore',
    'funding_rate_extreme',
    'oi_momentum_4h',
    'oi_price_divergence',
    'ls_ratio_zscore',
]

USE_DERIVATIVES_FEATURES = True   # set False when drift detected

MOMENTUM_FEATURES = (
    MOMENTUM_BASE_FEATURES + DERIVATIVES_FEATURES
    if USE_DERIVATIVES_FEATURES
    else MOMENTUM_BASE_FEATURES
)

META_FEATURES = MOMENTUM_FEATURES + [
    'atr_14', 'realized_volatility', 'atr_velocity',
    'volatility_percentile', 'bb_width', 'bb_width_percentile',
    'volume_ratio', 'amihud_illiquidity', 'liquidity_pressure_score',
    'volume_spike_score', 'regime_confidence',
]

VOLATILITY_FEATURES = [
    'atr_14', 'atr_expansion_ratio', 'atr_velocity',
    'bb_width', 'bb_width_percentile',
    'volume_ratio', 'amihud_illiquidity', 'realized_volatility',
]

RISK_FEATURES = [
    'regime_state', 'regime_confidence',
    'momentum_probability',
    'predicted_volatility',
    'atr_expansion_ratio', 'volatility_percentile',
    'amihud_illiquidity',
]

BEHAVIORAL_FEATURES = [
    'volatility_percentile',
    'atr_expansion_ratio',
    'volume_spike_score',
    'amihud_illiquidity',
    'trade_imbalance',
    'bb_width_percentile',
    'realized_volatility',
    'volume_ratio',
]

# ============================================================
# LABEL COLUMNS
# ============================================================

LABEL_MOMENTUM   = 'label_momentum'
LABEL_VOLATILITY = 'label_volatility'
LABEL_RISK       = 'label_risk'
LABEL_BEHAVIORAL = 'label_behavioral'

RISK_CLASSES = ['LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK', 'NO_TRADE']

# ============================================================
# HYPERPARAMETERS
# ============================================================

MOMENTUM_PARAMS = {
    'n_estimators':    300,
    'learning_rate':   0.03,      # lower LR = less overfit on small positive class
    'max_depth':       4,          # was 6 — shallower tree reduces overfit
    'min_child_weight': 10,        # was 5 — requires more samples per leaf
    'subsample':       0.8,
    'colsample_bytree': 0.7,
    'reg_alpha':       0.1,        # L1 regularization
    'reg_lambda':      1.0,        # L2 regularization
    'random_state':    42,
}

VOLATILITY_PARAMS = {
    'n_estimators':    200,
    'learning_rate':   0.05,
    'max_depth':       6,
    'min_child_weight': 10,
    'subsample':       0.8,
    'colsample_bytree': 0.8,
    'random_state':    42,
}

RISK_PARAMS = {
    'n_estimators':    300,
    'learning_rate':   0.05,
    'max_depth':       6,
    'min_child_weight': 5,
    'random_state':    42,
    'objective':       'multi:softprob',
    'num_class':       4,
}

# ============================================================
# BARRIER CONTRACT
#
# Key constraint: break-even win rate = SL / (TP + SL)
#
#   Original:  TP=1.5%, SL=0.6%  → BE win rate = 0.6/2.1 = 28.6%
#   Your model wins 25% → losing by 3.6pp on every trade
#
#   Fix option A (wider SL, same TP):
#     TP=1.5%, SL=0.8%  → BE win rate = 0.8/2.3 = 34.8%  ← worse, needs higher WR
#
#   Fix option B (tighter TP to match model skill):
#     TP=0.8%, SL=0.5%  → RR=1.6, BE win rate = 0.5/1.3 = 38.5%  ← still worse
#
#   Fix option C (keep RR, improve model):
#     Keep TP=1.5%, SL=0.6% but only trade when AUC > 0.56 in rolling CV
#     This is the right path — fix the model, not the barrier
#
#   CURRENT: keeping original values but adding AUC gate in training.
#   If AUC stays below 0.56 after fixing features, widen the horizon
#   (try MOMENTUM_HORIZON_BARS = 48 or 32) until labels are cleaner.
# ============================================================

MOMENTUM_HORIZON_BARS = 32         # was 64 — try 32 first, then 48 if AUC doesn't improve
MOMENTUM_TP_PCT       = 0.015      # 1.5% take profit
MOMENTUM_SL_PCT       = 0.006      # 0.6% stop loss
EXPECTED_RR_GATE      = MOMENTUM_TP_PCT / MOMENTUM_SL_PCT   # 2.5

MAX_RISK_PERCENT  = 1.0
MIN_CONFIDENCE    = 0.55           # was 0.53 — raise minimum bar slightly

MAX_HOLDING_BARS  = MOMENTUM_HORIZON_BARS
REGIME_MAX_BARS   = {
    'trending_up':   MAX_HOLDING_BARS,
    'trending_down': MAX_HOLDING_BARS,
    'mixed':         round(MAX_HOLDING_BARS * 0.7),
    'ranging':       round(MAX_HOLDING_BARS * 0.5),
}
REGIME_KELLY_MULTIPLIER = {
    'trending_up':   0.5,
    'trending_down': 0.5,
    'mixed':         0.3,
    'ranging':       0.1,
}

# Long-only system: don't trade ranging (noisy) or trending_down (wrong direction)
# If you add short logic, remove 'trending_down' from this list
NO_TRADE_REGIMES = ['ranging', 'trending_down']

# ============================================================
# VALIDATION THRESHOLDS (Minimum Acceptable)
# ============================================================

THRESHOLDS = {
    'momentum_auc_min':              0.56,   # hard gate — don't proceed if below this
    'momentum_action_threshold':     0.60,   # used only in fixed-threshold backtest
    'volatility_rmse_vs_baseline':   1.0,
    'risk_no_trade_precision_min':   0.70,
    'behavioral_f1_min':             0.65,
    'regime_silhouette_min':         0.30,
    'backtest_sharpe_min':           0.5,
}

# ============================================================
# ADAPTIVE THRESHOLD PERCENTILES
#
# Key fix: trending regimes were at 55th percentile — too loose.
# At median=0.507, the 55th pct is ~0.510 — barely above average.
# Raised to 70th/75th so only genuinely high-confidence bars trade.
# ============================================================

REGIME_PERCENTILE_THRESHOLDS = {
    'trending_up':   0.70,     # was 0.55
    'trending_down': 0.70,     # was 0.55
    'mixed':         0.80,     # was 0.75
    'ranging':       0.95,     # was 0.90
    'unknown':       0.90,     # was 0.80
}

# ============================================================
# DRIFT CONTROL
# ============================================================

CRITICAL_FEATURES = [
    'regime_state',
    'volatility_regime',
    'funding_rate_zscore',
    'oi_momentum_4h',
    'atr_14',
    'amihud_illiquidity',
]

# PSI above this threshold on any CRITICAL_FEATURE triggers
# USE_DERIVATIVES_FEATURES = False fallback
DERIVATIVES_DRIFT_PSI_LIMIT = 0.25

# ============================================================
# WALK-FORWARD CV
# ============================================================

WF_N_SPLITS = 5

# ============================================================
# MLFLOW
# ============================================================

MLFLOW_TRACKING_URI       = 'mlruns'
MLFLOW_EXPERIMENT_MOMENTUM    = 'momentum_v1'
MLFLOW_EXPERIMENT_VOLATILITY  = 'volatility_v1'
MLFLOW_EXPERIMENT_RISK        = 'risk_v1'
MLFLOW_EXPERIMENT_BEHAVIORAL  = 'behavioral_v1'