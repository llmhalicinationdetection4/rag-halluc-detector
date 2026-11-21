"""Phase 4: calibration, abstention and latency.

* A logistic-regression calibrator is fitted on the TRAIN features from Phase 3 (no class weights, so the outputs
  are real probabilities) and evaluated on TEST: ROC-AUC, Brier score and expected calibration error (ECE).
* The abstention margin ('uncertain' when |p - 0.5| < margin) is chosen on cross-validated TRAIN probabilities
  and then reported on TEST with its coverage and accuracy.
* If the Phase 1 HHEM train scores are still in results/cache, HHEM is calibrated the same way.
* Latency is measured one response at a time (like the dashboard) on real test responses.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from halludetect.calibration import (
    LogisticCalibrator,
    brier_score,
    choose_margin,
    coverage_accuracy_curve,
    expected_calibration_error,
)
from halludetect.eval.run_phase3 import H_COLUMNS
from halludetect.eval.threshold import safe_auc

FEATURES = list(H_COLUMNS.values())


def _cv_probabilities(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Out-of-fold TRAIN probabilities, so the abstention margin is not chosen on data the calibrator has memorised."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    folds = int(min(5, y.sum(), len(y) - y.sum()))
    if folds < 2:
        return LogisticCalibrator.fit(X, y, FEATURES[: X.shape[1]]).predict_proba(X)
    return cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=folds, method="predict_proba")[:, 1]


def _calibration_rows(model: str, tasks: np.ndarray, y: np.ndarray, p: np.ndarray, margin) -> list[dict]:
    rows = []
    for task in ["POOLED"] + sorted(set(tasks)):
        sel = np.ones(len(y), dtype=bool) if task == "POOLED" else tasks == task
        ece, _ = expected_calibration_error(y[sel], p[sel])
        row = {"model": model, "task": task, "n": int(sel.sum()), "auc": safe_auc(y[sel], p[sel]),
               "brier": brier_score(y[sel], p[sel]), "ece": ece, "abstain_margin": margin}
        if margin is not None:
            keep = np.abs(p[sel] - 0.5) >= margin
            row["coverage"] = float(keep.mean())
            row["accuracy_when_answering"] = (float(((p[sel][keep] >= 0.5).astype(int) == y[sel][keep]).mean())
                                              if keep.any() else float("nan"))
        rows.append(row)
    return rows


def _hhem_calibration(cfg, out_dir: Path, data_dir, predictions_path: Path, target_accuracy: float):
    """Calibrate HHEM when its Phase 1 TRAIN scores are still cached. Returns (rows, margin, message)."""
    from halludetect.data.ragtruth import load_ragtruth

    files = sorted((out_dir / "cache").glob("ragtruth_all_hhem_*_train.npy"))
    if not files or not predictions_path.exists():
        return [], None, "HHEM calibration skipped: Phase 1 train scores not found in results/cache (run `eval --dataset ragtruth`)"
    train = load_ragtruth(data_dir or cfg.data.ragtruth_dir, "train", None, cfg.seed)
    scores = np.load(files[0])
    if len(scores) != len(train):
        return [], None, f"HHEM calibration skipped: cached train scores ({len(scores)}) do not match the train split ({len(train)})"
    y_tr = np.array([int(e.label_hallucinated) for e in train])
    X_tr = (1.0 - scores).reshape(-1, 1)
    cal = LogisticCalibrator.fit(X_tr, y_tr, ["h_hhem"])
    cal.save(out_dir / "phase4_hhem_calibrator.json")
    margin = choose_margin(coverage_accuracy_curve(y_tr, _cv_probabilities(X_tr, y_tr)), target_accuracy)
    pred = pd.read_csv(predictions_path)
    y_te = pred["label_hallucinated"].to_numpy(dtype=int)
    p_te = cal.predict_proba((1.0 - pred["hhem_score"].to_numpy(dtype=float)).reshape(-1, 1))
    return _calibration_rows("hhem", pred["task_type"].to_numpy(), y_te, p_te, margin), margin, "HHEM calibrated"


def measure_latency(engines: dict, examples, n: int = 100, warmup: int = 3, device: str = "") -> pd.DataFrame:
    """Time each engine on `n` real responses, one at a time. `engines` maps a name to f(context, answer)."""
    def sync():
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

    rows = []
    for name, fn in engines.items():
        for ex in examples[:warmup]:
            fn(ex.context, ex.answer)
        times = []
        for ex in examples[warmup:warmup + n]:
            sync()
            t0 = time.perf_counter()
            fn(ex.context, ex.answer)
            sync()
            times.append((time.perf_counter() - t0) * 1000.0)
        if times:
            rows.append({"engine": name, "device": device, "n": len(times), "p50_ms": float(np.percentile(times, 50)),
                         "p95_ms": float(np.percentile(times, 95)), "mean_ms": float(np.mean(times))})
    return pd.DataFrame(rows).round(1)


def _default_engines(cfg, device: str | None):
    from halludetect.config import get_device
    from halludetect.models.hhem import HHEMModel
    from halludetect.models.nli import NLIModel
    from halludetect.verifiers.claim_level import ClaimLevelVerifier

    dev = get_device(device or cfg.device)
    m = cfg.models
    nli = NLIModel(m.nli, device=dev, max_length=m.max_length, batch_size=m.batch_size, fallback_name=m.nli_fallback)
    hhem = HHEMModel(m.hhem, device=dev, batch_size=m.batch_size)
    claim = ClaimLevelVerifier(nli=nli, device=dev)
    return ({"hhem": lambda c, a: hhem.predict_consistency([(c, a)]),
             "nli_whole_context": lambda c, a: nli.predict_proba([(c, a)]),
             "claim_level": lambda c, a: claim.verify(c, a)}, dev)


def run_phase4(cfg=None, out_dir=None, latency_n: int = 100, target_accuracy: float = 0.80, device: str | None = None,
               engines: dict | None = None, latency_examples=None, data_dir=None, predictions_path=None) -> dict:
    from halludetect.config import load_config

    cfg = cfg or load_config()
    out_dir = Path(out_dir or cfg.eval.results_dir)
    for name in ("phase3_features_train.csv", "phase3_features_test.csv"):
        if not (out_dir / name).exists():
            raise FileNotFoundError(f"{out_dir / name} not found. Run `python -m halludetect phase3` first.")
    train = pd.read_csv(out_dir / "phase3_features_train.csv")
    test = pd.read_csv(out_dir / "phase3_features_test.csv")
    X_tr, y_tr = train[FEATURES].to_numpy(dtype=float), train["y"].to_numpy(dtype=int)
    X_te, y_te = test[FEATURES].to_numpy(dtype=float), test["y"].to_numpy(dtype=int)

    cal = LogisticCalibrator.fit(X_tr, y_tr, FEATURES)
    cal.save(out_dir / "phase4_calibrator.json")
    p_te = cal.predict_proba(X_te)
    margin = choose_margin(coverage_accuracy_curve(y_tr, _cv_probabilities(X_tr, y_tr)), target_accuracy)
    coverage_accuracy_curve(y_te, p_te).round(4).to_csv(out_dir / "phase4_abstention_curve.csv", index=False)
    _, reliability = expected_calibration_error(y_te, p_te)
    reliability.round(4).to_csv(out_dir / "phase4_reliability.csv", index=False)

    rows = _calibration_rows("claim_level", test["task"].to_numpy(), y_te, p_te, margin)
    hhem_rows, hhem_margin, hhem_message = _hhem_calibration(
        cfg, out_dir, data_dir, Path(predictions_path or out_dir / "phase1_predictions.csv"), target_accuracy)
    rows += hhem_rows
    calibration = pd.DataFrame(rows).round(4)
    calibration.to_csv(out_dir / "phase4_calibration.csv", index=False)

    latency = pd.DataFrame()
    if latency_n > 0:
        dev_label = device or cfg.device
        if engines is None:
            from halludetect.data.ragtruth import load_ragtruth

            engines, dev_label = _default_engines(cfg, device)
            latency_examples = load_ragtruth(data_dir or cfg.data.ragtruth_dir, "test", latency_n + 3, cfg.seed)
        latency = measure_latency(engines, latency_examples, latency_n, device=dev_label)
        latency.to_csv(out_dir / "phase4_latency.csv", index=False)

    summary = {"abstain_margin": margin, "target_accuracy": target_accuracy, "features": FEATURES,
               "hhem_abstain_margin": hhem_margin, "hhem_note": hhem_message,
               "note": "calibrator and margin fitted on TRAIN only; metrics are from the TEST split"}
    (out_dir / "phase4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nCALIBRATION (test split)")
    print(calibration.to_string(index=False))
    print(f"\nAbstain margin chosen on train: {margin} (target accuracy {target_accuracy:.0%})")
    print(hhem_message)
    if len(latency):
        print("\nLATENCY (one response at a time, real test responses)")
        print(latency.to_string(index=False))
    print(f"\nSaved phase4_*.csv/json in {out_dir}")
    return {"calibration": calibration, "latency": latency, "summary": summary}
