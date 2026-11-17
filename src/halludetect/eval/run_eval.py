"""Phase 1 evaluation: whole-context NLI baseline vs HHEM.

Protocol: tune tau on the train split, report on the test split (never tune on test).
Writes results/phase1_metrics.csv, phase1_predictions.csv and phase1_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from halludetect.config import get_device, set_seed
from halludetect.data.demo import load_demo
from halludetect.data.ragtruth import load_ragtruth
from halludetect.eval.metrics import best_threshold, classification_metrics


def load_splits(cfg, dataset: str, max_examples: int | None):
    if dataset == "demo":
        return (load_demo(cfg.data.demo_path, "train"), load_demo(cfg.data.demo_path, "test"))
    if dataset == "ragtruth":
        return (
            load_ragtruth(cfg.data.ragtruth_dir, "train", max_examples, cfg.seed),
            load_ragtruth(cfg.data.ragtruth_dir, "test", max_examples, cfg.seed),
        )
    raise ValueError("dataset must be 'demo' or 'ragtruth'")


def _pairs(examples):
    return [(e.context, e.answer) for e in examples]


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text))


def _cached(cache_dir: Path | None, key: str, n: int, compute):
    """Run compute() once and keep the result on disk, so a crash later does not lose finished work."""
    if cache_dir is None:
        return compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_slug(key)}.npy"
    if path.exists():
        arr = np.load(path)
        if arr.shape[0] == n:
            print(f"[cache] loaded {path.name}")
            return arr
    arr = compute()
    np.save(path, arr)
    print(f"[cache] saved {path.name}")
    return arr


def run_phase1(
    cfg,
    dataset: str = "demo",
    max_examples: int | None = None,
    use_hhem: bool = True,
    out_dir: str | Path | None = None,
    nli=None,
    hhem=None,
) -> dict:
    """nli / hhem can be injected (used by tests); otherwise real models are loaded."""
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out_dir = Path(out_dir or cfg.eval.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if max_examples is None:
        max_examples = cfg.eval.max_examples

    train, test = load_splits(cfg, dataset, max_examples)
    if not train or not test:
        raise ValueError(f"Empty split (train={len(train)}, test={len(test)}). Check the dataset path and split names.")
    y_train = np.array([e.label_hallucinated for e in train], dtype=int)
    y_test = np.array([e.label_hallucinated for e in test], dtype=int)

    use_cache = nli is None and hhem is None  # tests inject stand-in models and skip the cache
    cache_dir = out_dir / "cache" if use_cache else None
    tag = f"{dataset}_{max_examples if max_examples is not None else 'all'}"

    if nli is None:
        from halludetect.models.nli import NLIModel

        nli = NLIModel(cfg.models.nli, device=device, max_length=cfg.models.max_length,
                       batch_size=cfg.models.batch_size, fallback_name=cfg.models.nli_fallback)

    nli_tag = f"{tag}_nli_{getattr(nli, 'model_name', 'x')}"
    print(f"[1/4] NLI on train ({len(train)} examples)", flush=True)
    probs_train = _cached(cache_dir, f"{nli_tag}_train", len(train), lambda: nli.predict_proba(_pairs(train)))
    print(f"[2/4] NLI on test ({len(test)} examples)", flush=True)
    probs_test = _cached(cache_dir, f"{nli_tag}_test", len(test), lambda: nli.predict_proba(_pairs(test)))
    score_sets = {"nli_whole_context": (probs_train[:, 0], probs_test[:, 0])}

    hhem_status = "skipped (use_hhem=False)"
    hhem_test_scores = None
    if use_hhem:
        try:
            if hhem is None:
                from halludetect.models.hhem import HHEMModel

                hhem = HHEMModel(cfg.models.hhem, device=device, batch_size=cfg.models.batch_size)
            hhem_tag = f"{tag}_hhem_{getattr(hhem, 'model_name', 'x')}"
            print(f"[3/4] HHEM on train ({len(train)} examples)", flush=True)
            hhem_train = _cached(cache_dir, f"{hhem_tag}_train", len(train),
                                 lambda: hhem.predict_consistency(_pairs(train)))
            print(f"[4/4] HHEM on test ({len(test)} examples)", flush=True)
            hhem_test_scores = _cached(cache_dir, f"{hhem_tag}_test", len(test),
                                       lambda: hhem.predict_consistency(_pairs(test)))
            score_sets["hhem"] = (hhem_train, hhem_test_scores)
            hhem_status = "ok"
        except Exception as err:  # noqa: BLE001
            hhem_status = f"FAILED: {type(err).__name__}: {err}"
            print(f"[warn] HHEM failed and was skipped -> {hhem_status}")

    rows, taus = [], {}
    for name, (s_train, s_test) in score_sets.items():
        tuned_tau, train_f1 = best_threshold(y_train, s_train)
        taus[name] = {"tuned_tau": tuned_tau, "train_f1_at_tuned_tau": train_f1}
        for setting, tau in (("fixed_0.5", cfg.baseline.tau), ("tuned_on_train", tuned_tau)):
            m = classification_metrics(y_test, s_test, tau)
            rows.append({"model": name, "setting": setting, **m})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "phase1_metrics.csv", index=False)

    preds = pd.DataFrame({
        "id": [e.id for e in test],
        "task_type": [e.task_type for e in test],
        "label_hallucinated": y_test,
        "nli_p_entail": probs_test[:, 0],
        "nli_p_neutral": probs_test[:, 1],
        "nli_p_contra": probs_test[:, 2],
    })
    if hhem_test_scores is not None:
        preds["hhem_score"] = hhem_test_scores
    preds.to_csv(out_dir / "phase1_predictions.csv", index=False)

    summary = {
        "dataset": dataset, "n_train": len(train), "n_test": len(test), "device": device,
        "seed": cfg.seed, "nli_model": getattr(nli, "model_name", "injected"),
        "hhem_status": hhem_status, "thresholds": taus,
    }
    with open(out_dir / "phase1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    show = ["model", "setting", "tau", "precision", "recall", "f1", "roc_auc", "n"]
    print(metrics[show].round(3).to_string(index=False))
    return {"metrics": metrics, "predictions": preds, "summary": summary}
