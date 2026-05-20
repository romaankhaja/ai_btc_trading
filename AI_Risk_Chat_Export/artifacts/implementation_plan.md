# Phase 6 Pipeline Audit — Complete Bug Report & Fix Plan

## Summary of Findings

I audited every file in the Phase 6 inference pipeline. Below are **7 distinct issues** — 3 critical bugs, 2 design flaws, and 2 structural weaknesses — ranked by severity.

---

## BUG 1 — CRITICAL: HMM Single-Point Inference Always Returns `trending_low_vol`

**File:** [model_ensemble.py](file:///c:/Users/sathwik.kusuri/Documents/AI_Risk_Management/inference/model_ensemble.py) — Line 99

**Root Cause:** The HMM Viterbi algorithm requires a **sequence** to compute state transitions. `model.predict(single_row)` with a 1-element array has no temporal context, so the HMM defaults to the state with the highest stationary probability — always state 1 (`trending_low_vol`).

**Evidence:**
- Batch prediction (full test sequence): 4 regimes — `sideways: 1688, trending_low: 1544, choppy: 1272, trending_high: 734`
- Single-point prediction (current code): `trending_low_vol: 5238/5238` (100%)

**Impact:** The Policy Engine's regime routing (lines 134–158) is completely dead. All 205 trades use the `trending_low_vol` parameters. The `choppy_high_vol` 70% risk reduction, `sideways_low_vol` 50% reduction, and `crash_mode` hard block never fire.

**Fix:** Add a rolling `deque(maxlen=50)` buffer for HMM features. Pass the full buffer to `model.predict()` each time. Use the last element as the current regime.

---

## BUG 2 — CRITICAL: `meta_labeler.py` pt_sl Parameter Order Mismatch

**File:** [meta_labeler.py](file:///c:/Users/sathwik.kusuri/Documents/AI_Risk_Management/labeling/meta_labeler.py) — Lines 18 & 122

**Root Cause:** The function signature is `pt_sl=[1.5, 1.0]` where index 0 = **Profit Taking** and index 1 = **Stop Loss**. The call on line 122 passes `pt_sl=[1.0, 2.0]`, meaning TP=1.0×ATR, SL=2.0×ATR.

But in `risk_sizer.py` and `policy_engine.py`, the parameters are `sl_multiplier` and `tp_multiplier` — and the **defaults in `PolicyDecision`** (line 25-26) are `sl_multiplier=1.5, tp_multiplier=2.7`, which are **completely different** from both the labeler AND the regime routing values.

**Impact:** If the regime routing code is bypassed (which it currently is — see Bug 1), the system falls through to the `PolicyDecision` defaults: SL=1.5, TP=2.7. These do NOT match what the model was trained on (TP=1.0, SL=2.0), creating a **train/inference mismatch**.

**Fix:** Align `PolicyDecision` defaults to match the meta-labeler: `sl_multiplier=2.0, tp_multiplier=1.0`. Or better — change to symmetric R:R (see Fix 4).

---

## BUG 3 — MODERATE: PnL Calculation Asymmetry on Fees

**File:** [paper_trader.py](file:///c:/Users/sathwik.kusuri/Documents/AI_Risk_Management/execution/paper_trader.py) — Lines 125-129

**Root Cause:** The code computes:
```python
slippage = open_trade.exit_price * self.SLIPPAGE_PCT
fee = open_trade.exit_price * self.FEE_PCT
raw_pnl = direction * (exit_price - entry_price)
pnl_per_unit = raw_pnl - slippage - fee
```

The `slippage` is always **subtracted** regardless of direction. But slippage on a TP_HIT for a LONG should already be baked into the TP price (which is an optimistic fill). The real issue: `fee` is only applied on exit, but in reality fees are charged on **both entry and exit**. Entry slippage is applied (line 172) but entry fees are not.

**Impact:** Every trade is under-charged by approximately $0.04% × position_size on the entry leg. Over 205 trades this understates total fee drag.

**Fix:** Add entry fee to the PnL calculation:
```python
entry_fee = open_trade.entry_price * self.FEE_PCT
exit_fee = open_trade.exit_price * self.FEE_PCT
pnl_per_unit = raw_pnl - slippage - entry_fee - exit_fee
```

---

## BUG 4 — DESIGN: Inverse R:R Requires 76% Win Rate to Break Even

**Files:** [policy_engine.py](file:///c:/Users/sathwik.kusuri/Documents/AI_Risk_Management/inference/policy_engine.py) Lines 136-137, [meta_labeler.py](file:///c:/Users/sathwik.kusuri/Documents/AI_Risk_Management/labeling/meta_labeler.py) Line 122

**Root Cause:** Current configuration: TP = 1.0×ATR, SL = 2.0×ATR. Each SL loss ($218.93 avg) costs 3.1× each TP win ($69.65 avg). Breakeven requires 75.9% win rate. Current win rate is 69.8% — a 6-point gap.

**Fix:** Switch to symmetric R:R across the entire pipeline:
- **meta_labeler.py:** `pt_sl=[1.5, 1.5]` (symmetric)
- **policy_engine.py:** Regime routing uses symmetric multipliers
- **risk_sizer.py:** Defaults to `sl_multiplier=1.5, tp_multiplier=1.5`

With 1:1 R:R, breakeven drops to 50%. Even if win rate drops from 69.8% to ~55% due to tighter TP, the system is profitable.

---

## BUG 5 — DESIGN: SHORT Trades Bleed in Bullish Markets (No Trend Filter)

**File:** [trade_decision.py](file:///c:/Users/sathwik.kusuri/Documents/AI_Risk_Management/inference/trade_decision.py) — Line 114-115

**Root Cause:** The test period is +8.73% bullish ($71,214 → $77,430). The model takes 121 SHORT trades (59%) vs 84 LONG (41%). SHORT win rate is 67.8% vs LONG 72.6%. SHORT PnL = -$1,950. There is **no macro trend awareness** — the system blindly follows the Primary Model's direction.

**Fix:** Add a soft trend-alignment filter in `policy_engine.py`. If direction opposes the macro trend (`ema_50_slope`), apply a 50% risk reduction. This doesn't block counter-trend trades; it sizes them smaller.

---

## BUG 6 — STRUCTURAL: Meta-Model AUC is Near-Random (0.50–0.51)

**File:** [train_meta_ensemble.py](file:///c:/Users/sathwik.kusuri/Documents/AI_Risk_Management/training/train_meta_ensemble.py) — Lines 72-76

**Root Cause:** The Meta-Model is trained on the **same features** (`MOMENTUM_FEATURES`) as the Primary Model. It receives no additional information about *why* a trade might succeed or fail. The Meta-Model should consume features the Primary Model does NOT see — like volatility regime, liquidity conditions, and volume dynamics.

**Evidence:** Val AUC = 0.5172, Test AUC = 0.5061 — effectively random.

**Impact:** The `meta_probability` and `meta_margin` used for Z-Score ranking are nearly noise. The Z-Score threshold is filtering on random fluctuations, not genuine signal quality.

**Fix:** Create a separate `META_FEATURES` list that includes `MOMENTUM_FEATURES` PLUS volatility/liquidity signals: `atr_14`, `realized_volatility`, `volume_ratio`, `amihud_illiquidity`, `regime_cluster`, `regime_confidence`. This gives the Meta-Model orthogonal information to assess trade quality.

---

## BUG 7 — STRUCTURAL: `bars_held` Uninitialized on First Candle

**File:** [paper_trader.py](file:///c:/Users/sathwik.kusuri/Documents/AI_Risk_Management/execution/paper_trader.py) — Line 143

**Root Cause:** `bars_held` is only initialized when a trade is opened (line 185) or closed (lines 141, 164). If the very first candle has an open trade from warm-up, `bars_held` is referenced before assignment (line 143).

**Impact:** Currently doesn't crash because the warm-up loop doesn't open trades. But in a live system, this would cause a `NameError`.

**Fix:** Initialize `bars_held = 0` at the top of the `run()` method, alongside `open_trade = None`.

---

## Proposed Changes Summary

| # | Severity | File | Fix |
| :--- | :--- | :--- | :--- |
| 1 | **CRITICAL** | `model_ensemble.py` | Rolling HMM buffer (50 bars) |
| 2 | **CRITICAL** | `policy_engine.py` | Fix `PolicyDecision` defaults to match labeler |
| 3 | **MODERATE** | `paper_trader.py` | Add entry fee to PnL calculation |
| 4 | **DESIGN** | `meta_labeler.py`, `policy_engine.py`, `risk_sizer.py` | Switch to symmetric 1:1 R:R |
| 5 | **DESIGN** | `policy_engine.py` | Add trend-alignment soft filter |
| 6 | **STRUCTURAL** | `train_meta_ensemble.py`, `config.py` | Create `META_FEATURES` with orthogonal signals |
| 7 | **STRUCTURAL** | `paper_trader.py` | Initialize `bars_held = 0` at method start |

## Verification Plan

### Automated Tests
1. Re-run `run_phase6.py` end-to-end after all fixes.
2. Verify regime breakdown shows **multiple** regimes (not just `trending_low_vol`).
3. Compare Sharpe, Total Return, and SHORT PnL against current baseline.
4. Confirm Meta-Model AUC improves above 0.52 with expanded feature set.

### Acceptance Criteria
- Regime distribution in paper trading results matches batch HMM prediction (~4 states).
- SHORT trades no longer dominate the trade count.
- Total Return improves (target: > -10%, ideally positive).
