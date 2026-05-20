import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from training.config import MOMENTUM_FEATURES, META_FEATURES, VOLATILITY_FEATURES, RISK_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "labeled" / "BTCUSDT"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def check(name, fn, results):
    try:
        fn()
        results.append((name, True, "PASS"))
    except Exception as exc:
        results.append((name, False, f"FAIL: {exc}"))


def main():
    results = []

    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")

    def schema_check(df, split_name):
        def _fn():
            required = {"regime_state", "regime_label", "regime_confidence", "label_momentum", "label_meta"}
            missing = sorted(required - set(df.columns))
            if missing:
                raise AssertionError(f"{split_name}: missing columns {missing}")
            usable = df[df["regime_label"] != "choppy_high_vol"]
            for col in ["label_momentum", "label_meta"]:
                na_ratio = usable[col].isna().mean()
                if na_ratio > 0.05:
                    raise AssertionError(f"{split_name}: {col} NaN ratio {na_ratio:.2%} > 5%")
        return _fn

    check("train schema", schema_check(train_df, "train"), results)
    check("val schema", schema_check(val_df, "val"), results)
    check("test schema", schema_check(test_df, "test"), results)

    def model_shape_check():
        sample = train_df.iloc[[0]].copy()
        for col in set(MOMENTUM_FEATURES + VOLATILITY_FEATURES + RISK_FEATURES):
            if col not in sample.columns:
                sample[col] = 0.0

        primary_model = None
        meta_model = None
        for regime_dir in (MODELS_DIR / "momentum").glob("*"):
            primary_candidate = regime_dir / "primary_model.pkl"
            meta_candidate = regime_dir / "calibrated_meta_model.pkl"
            if primary_candidate.exists() and meta_candidate.exists():
                primary_model = joblib.load(primary_candidate)
                meta_model = joblib.load(meta_candidate)
                break
        if primary_model is None or meta_model is None:
            raise AssertionError("No complete momentum model pair found")

        vol_model = joblib.load(MODELS_DIR / "volatility" / "volatility_model.pkl")
        risk_model = joblib.load(MODELS_DIR / "risk" / "risk_model.pkl")

        x_mom = sample[MOMENTUM_FEATURES]
        x_meta = sample[META_FEATURES]
        x_vol = sample[VOLATILITY_FEATURES]
        x_risk = sample[RISK_FEATURES]

        primary_pred = primary_model.predict(x_mom)
        if len(primary_pred) != 1:
            raise AssertionError("Primary momentum model returned unexpected prediction length")

        mom_proba = meta_model.predict_proba(x_meta)
        if getattr(mom_proba, "shape", (0, 0))[1] != 2:
            raise AssertionError(f"Momentum meta model predict_proba shape {mom_proba.shape} != (1, 2)")

        vol_pred = vol_model.predict(x_vol)
        if len(vol_pred) != 1:
            raise AssertionError("Volatility model returned unexpected prediction length")

        risk_pred = risk_model.predict(x_risk)
        if len(risk_pred) != 1:
            raise AssertionError("Risk model returned unexpected prediction length")

    check("model shape compatibility", model_shape_check, results)

    def artifact_check():
        expected = [
            MODELS_DIR / "regime" / "nhhmm_regime_detector.pkl",
            MODELS_DIR / "volatility" / "volatility_model.pkl",
            MODELS_DIR / "risk" / "risk_model.pkl",
            MODELS_DIR / "behavioral" / "behavioral_iforest.pkl",
        ]
        for regime in ["trending_low_vol", "trending_high_vol", "sideways_low_vol", "crash_mode"]:
            expected.append(MODELS_DIR / "momentum" / regime / "primary_model.pkl")
            expected.append(MODELS_DIR / "momentum" / regime / "calibrated_meta_model.pkl")
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise AssertionError(f"Missing model artifacts: {missing}")

    check("model artifacts", artifact_check, results)

    def regime_detector_check():
        detector = joblib.load(MODELS_DIR / "regime" / "nhhmm_regime_detector.pkl")
        labeled = detector.assign_labels(train_df.head(50))
        if "regime_state" not in labeled.columns:
            raise AssertionError("NHHMM detector did not emit regime_state")
    check("regime detector", regime_detector_check, results)

    def trades_check():
        trades_file = RESULTS_DIR / "test_trades.csv"
        if not trades_file.exists():
            raise AssertionError("results/test_trades.csv missing")
        trades_df = pd.read_csv(trades_file)
        if len(trades_df) <= 0:
            raise AssertionError("results/test_trades.csv is empty")
    check("trade log", trades_check, results)

    def performance_report_check():
        perf_file = RESULTS_DIR / "performance_report.json"
        if not perf_file.exists():
            raise AssertionError("results/performance_report.json missing")
        with open(perf_file, "r", encoding="utf-8") as f:
            report = json.load(f)
        for key in [
            "total_trades",
            "win_rate",
            "sharpe_ratio",
            "max_drawdown",
            "total_pnl",
            "avg_trade_pnl",
            "ece_by_model",
            "circuit_breaker_activations",
        ]:
            if key not in report:
                raise AssertionError(f"Missing performance report key: {key}")
        cb_log = RESULTS_DIR / "circuit_breaker_log.csv"
        if not cb_log.exists():
            raise AssertionError("results/circuit_breaker_log.csv missing")
    check("performance report", performance_report_check, results)

    def ece_gate_check():
        perf_file = RESULTS_DIR / "performance_report.json"
        with open(perf_file, "r", encoding="utf-8") as f:
            report = json.load(f)
        ece_by_model = report.get("ece_by_model", {})
        if not ece_by_model:
            raise AssertionError("Missing ece_by_model in performance report")
        failures = {k: v for k, v in ece_by_model.items() if float(v) > 0.08}
        if failures:
            raise AssertionError(f"ECE gate failed: {failures}")

    check("ece gate", ece_gate_check, results)

    def circuit_breaker_log_check():
        cb_log = RESULTS_DIR / "circuit_breaker_log.csv"
        df = pd.read_csv(cb_log)
        required = {"timestamp", "event_type", "detail", "consecutive_sl_count", "cooldown_remaining_bars"}
        missing = required - set(df.columns)
        if missing:
            raise AssertionError(f"circuit_breaker_log missing columns {sorted(missing)}")
        if len(df) <= 0:
            raise AssertionError("circuit_breaker_log.csv is empty")

    check("circuit breaker log", circuit_breaker_log_check, results)

    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed

    print("\nVERIFICATION RESULTS")
    for name, ok, message in results:
        print(f"{name:28s} : {message}")
    print(f"\nSummary: {passed} passed, {failed} failed")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
