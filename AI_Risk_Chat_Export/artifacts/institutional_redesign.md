# Institutional Architecture Redesign: Adaptive AI Risk Management

Based on a critical quantitative audit of the Phase 5 architecture, the following document outlines a comprehensive, institutional-grade redesign. The current system successfully implements robust capital protection (downside gating) but suffers from weak expectancy (Sharpe 0.62) due to path-dependent labeling, poor probability calibration, Euclidean regime clustering, and model forcing (asking a trend model to trade sideways regimes).

---

## 1. Root-Cause Diagnosis & Weak Subsystem Ranking

1. **Labeling System (CRITICAL FLAW):** Fixed `1.5 TP / 1.0 SL` ATR barriers force the model to learn path-dependent noise rather than directional alpha. In high-volatility sideways regimes, the 1.0 ATR stop is frequently triggered before the macro trend resolves, punishing the model for correct directional forecasts that had noisy paths.
2. **Calibration Engine (SEVERE):** Isotonic regression on highly overlapping financial data with a low base rate flattens the probability space into a tight 0.28–0.35 band. This destroys rank-ordering and renders absolute probabilities useless for sizing.
3. **Regime Detection (MODERATE):** KMeans relies on Euclidean distance, which is highly sensitive to the heavy tails of financial returns. Furthermore, it lacks temporal awareness (transition probabilities), causing `choppy_high_vol` to rapidly oscillate with `sideways_low_vol`.
4. **Execution/Sizing (MODERATE):** Static ATR multipliers (e.g., `SL=2.0` in `trending_high_vol`) ignore the continuously changing predicted volatility and the model's edge (probability).

---

## 2. Labeling Redesign: Triple-Barrier Meta-Labeling

**Current:** Binary classification (1 = TP hit, 0 = SL hit).
**Institutional Redesign:** 
Separate the prediction of *direction* from the prediction of *edge/magnitude*.

1.  **Primary Model (Directional):**
    *   **Target:** Predict the sign of the forward volatility-adjusted return (e.g., `sign(Return_8 / Realized_Vol_8)`).
    *   **Execution:** Generates raw directional signals (LONG/SHORT).
2.  **Secondary Model (Meta-Labeling):**
    *   **Target:** Apply Lopez de Prado’s Triple-Barrier Method. Set dynamic TP/SL barriers based on rolling volatility, plus a time barrier (e.g., 12 bars). 
    *   **Label:** 1 if the Primary Model's direction hits the TP barrier first; 0 if it hits SL or times out.
    *   **Execution:** Predicts the *probability that the primary model is correct*. This separates alpha generation from risk management.

---

## 3. Regime Engine Redesign: Hidden Markov Models (HMM)

**Current:** KMeans clustering on normalized features.
**Institutional Redesign:**
Replace KMeans with a **Gaussian Hidden Markov Model (HMM)**.
*   **Why:** Financial regimes are states in a Markov chain. A sideways market has a high probability of transitioning to a trending market (breakout), and a trending market eventually transitions to high-volatility choppy (exhaustion). HMMs explicitly model this Transition Probability Matrix, providing immense temporal stability.
*   **Features:** Fit the HMM on continuous, stationary series: log returns and realized volatility (not moving average slopes). 
*   **Output:** The HMM outputs the posterior probability of being in state $S_t$ given the sequence of observations, providing a probabilistic regime vector (e.g., `[0.1, 0.8, 0.05, 0.05]`) rather than a hard cluster ID.

---

## 4. Calibration & Signal Ranking Redesign

**Current:** Isotonic Calibration → Adaptive Percentile Thresholds.
**Institutional Redesign:**
*   **Drop Isotonic:** Isotonic regression is inappropriate for low-SNR financial data. 
*   **Move to Ranking:** Instead of forcing absolute probability thresholds, use the raw log-odds (margins) from the Meta-Labeling XGBoost model. 
*   **Cross-Sectional/Temporal Ranking:** Maintain a rolling window of the last 1000 raw predictions. Standardize the current prediction (Z-score). Only execute trades when the Z-score exceeds a statistically significant threshold (e.g., $Z > 1.64$ for the 95th percentile). This inherently adapts to distribution shifts without requiring hacky percentile thresholds.

---

## 5. Feature Engineering & Orthogonalization

**Current Redundancies:** `ema_20_slope`, `ema_50_slope`, `rsi_velocity`, `trend_acceleration` are highly collinear representations of price momentum.
**Institutional Redesign:**
1.  **Orthogonalization:** Perform Principal Component Analysis (PCA) on highly correlated feature subsets (e.g., all momentum oscillators). Feed the first 2-3 principal components into XGBoost. This prevents tree dilution where importance is randomly split across collinear features.
2.  **Cross-Sectional Interactions:** Tree models struggle with division. Explicitly engineer ratio features: 
    *   `vol_adjusted_momentum = rsi_velocity / realized_volatility`
    *   `liquidity_adjusted_trend = ema_20_slope * volume_delta`

---

## 6. Risk Sizing Redesign: Fractional Kelly

**Current:** Static risk percentages and static ATR multipliers per regime.
**Institutional Redesign:**
Implement dynamic exposure scaling based on the **Kelly Criterion**, modulated by predicted volatility.
*   **Edge ($p$):** Probability from the Meta-Labeling model.
*   **Odds ($b$):** Predicted Take-Profit distance / Stop-Loss distance.
*   **Kelly Fraction ($f^*$):** $f^* = p - (1-p)/b$
*   **Fractional Scaling:** $Risk\_Percent = f^* \times \text{Half-Kelly Scalar} \times \text{Regime Penalty}$
*   **Volatility Targeting:** Scale the absolute position size so that the expected portfolio variance remains constant, regardless of the underlying asset's current ATR.

---

## 7. Ensemble & Architecture Redesign

**Current:** A single trend model forced to trade all regimes.
**Institutional Redesign: The Hierarchical Mixture of Experts**
Do not force a trend model to trade sideways markets.
1.  **Regime Router (HMM):** Determines the probability distribution of current states.
2.  **Expert 1 (Trend-Following):** Meta-labeled model optimized for trending regimes.
3.  **Expert 2 (Mean-Reversion):** Meta-labeled model (e.g., Bollinger Band fade) optimized for sideways regimes.
4.  **Gating Network:** The final signal is a weighted average of the Experts, weighted by the HMM's regime probabilities. If `choppy_high_vol` is dominant, the gating network applies a weight of 0 to all experts (effectively a dynamic NO_TRADE).

---

## 8. Statistical Diagnostics to Run Next

Before rewriting code, run these diagnostics to quantify the flaws:
1.  **Feature Collinearity Matrix:** Compute the Spearman rank correlation of all features. Drop or PCA any pairs with $\rho > 0.85$.
2.  **Calibration Curve (Reliability Diagram):** Plot the binned predicted probabilities vs. actual hit rates for the Isotonic model to visualize the flatlining. 
3.  **Path Dependency Analysis:** Calculate the percentage of trades in the training set where price *eventually* hit the 1.5 TP, but hit the 1.0 SL first due to noise. This quantifies the noise-trap of the current labeling system.

---

## Priority Order of Fixes

1.  **URGENT:** Rip out fixed ATR momentum labels. Implement Continuous Target (forward Sharpe) or Triple-Barrier Meta-Labeling.
2.  **HIGH:** Remove Isotonic Calibration. Switch execution logic to Z-score ranking of raw model margins over a rolling window.
3.  **HIGH:** Implement Hierarchical Ensemble. Stop executing the trend model in sideways regimes; either hard-block it or build a dedicated mean-reversion expert.
4.  **MEDIUM:** Replace static risk sizing with Fractional Kelly.
5.  **MEDIUM:** Replace KMeans with a Gaussian HMM on returns/volatility.
