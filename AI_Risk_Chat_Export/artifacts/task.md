# Phase 6 Pipeline Fix — Task Tracker

- `[x]` Bug 1: Rolling HMM buffer in `model_ensemble.py`
- `[x]` Bug 2: Fix `PolicyDecision` defaults in `policy_engine.py`
- `[x]` Bug 3: Add entry fee to PnL in `paper_trader.py`
- `[x]` Bug 4: Symmetric R:R across `meta_labeler.py`, `policy_engine.py`, `risk_sizer.py`
- `[x]` Bug 5: Trend-alignment soft filter in `policy_engine.py`
- `[x]` Bug 6: Create `META_FEATURES` in `config.py` and `train_meta_ensemble.py`
- `[x]` Bug 7: Initialize `bars_held` in `paper_trader.py`
- `[x]` Re-run meta_labeler with new symmetric R:R
- `[x]` Re-run `run_phase6.py` and validate results
- `[x]` Block choppy_high_vol regime (39% WR)
- `[x]` Block sideways_low_vol regime (54% WR - can't overcome fees)
