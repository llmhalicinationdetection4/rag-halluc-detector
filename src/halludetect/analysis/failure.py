"""Phase 2: failure analysis.

1. Injected errors: score each (original, edited) pair with both verifiers and measure, per error type, how
   often the edit lowers the score (paired accuracy) and how often it is newly flagged.
2. Real errors: break the Phase 1 test predictions down by RAGTruth's own annotation (label type, response
   quality), so the false positives caused by refusals and truncated answers become visible.

Scores are consistency scores: higher = more consistent with the context. A response is flagged
(predicted hallucinated) when its score is below the verifier's tau from Phase 1.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from halludetect.config import get_device
from halludetect.eval.run_eval import _cached
from halludetect.inject.injectors import ERROR_TYPES, read_injections

MODELS = ("nli_whole_context", "hhem")


def _pairs_key(pairs) -> str:
    h = hashlib.sha1()
    for context, answer in pairs:
        h.update(context.encode("utf-8", "replace"))
        h.update(b"\x1e")
        h.update(answer.encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def score_injections(cfg, injections, use_hhem: bool = True, nli=None, hhem=None, out_dir=None) -> pd.DataFrame:
    """One row per injection with original and edited scores for each model."""
    out_dir = Path(out_dir or cfg.eval.results_dir)
    use_cache = nli is None and hhem is None
    cache_dir = out_dir / "cache" if use_cache else None
    device = get_device(cfg.device)

    originals: dict[str, tuple[str, str]] = {}
    for inj in injections:
        originals.setdefault(inj.example_id, (inj.context, inj.original))
    orig_ids = sorted(originals)
    orig_pairs = [originals[i] for i in orig_ids]
    edit_pairs = [(inj.context, inj.edited) for inj in injections]
    print(f"{len(edit_pairs)} edited responses and {len(orig_pairs)} originals to score", flush=True)

    df = pd.DataFrame({
        "example_id": [inj.example_id for inj in injections],
        "task_type": [inj.task_type for inj in injections],
        "error_type": [inj.error_type for inj in injections],
        "description": [inj.description for inj in injections],
    })
    id_to_row = {i: k for k, i in enumerate(orig_ids)}
    orig_index = [id_to_row[inj.example_id] for inj in injections]

    def add(model_name, scorer, tag):
        orig = _cached(cache_dir, f"phase2_{tag}_orig_{_pairs_key(orig_pairs)}", len(orig_pairs), lambda: scorer(orig_pairs))
        edit = _cached(cache_dir, f"phase2_{tag}_edit_{_pairs_key(edit_pairs)}", len(edit_pairs), lambda: scorer(edit_pairs))
        df[f"orig_{model_name}"] = np.asarray(orig)[orig_index]
        df[f"edit_{model_name}"] = np.asarray(edit)

    if nli is None:
        from halludetect.models.nli import NLIModel

        nli = NLIModel(cfg.models.nli, device=device, max_length=cfg.models.max_length,
                       batch_size=cfg.models.batch_size, fallback_name=cfg.models.nli_fallback)
    print("[1/2] NLI on originals and edits", flush=True)
    add("nli_whole_context", lambda pairs: nli.predict_proba(pairs)[:, 0], f"nli_{getattr(nli, 'model_name', 'x')}")

    if use_hhem:
        try:
            if hhem is None:
                from halludetect.models.hhem import HHEMModel

                hhem = HHEMModel(cfg.models.hhem, device=device, batch_size=cfg.models.batch_size)
            print("[2/2] HHEM on originals and edits", flush=True)
            add("hhem", lambda pairs: hhem.predict_consistency(pairs), f"hhem_{getattr(hhem, 'model_name', 'x')}")
        except Exception as err:  # noqa: BLE001
            print(f"[warn] HHEM failed and was skipped -> {type(err).__name__}: {err}")
    return df


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_metrics(df: pd.DataFrame, taus: dict[str, float]) -> pd.DataFrame:
    """Per (model, task, error type):
    pair_acc      share of pairs where the edit lowered the score (ties count half); 0.5 = no signal
    mean_drop     average score drop caused by the edit
    flagged_orig  share of the ORIGINAL responses flagged at tau (these are false alarms: they were faithful)
    flagged_edit  share of the edited responses flagged at tau
    caught        among originals that passed, the share whose edit became flagged
    """
    rows = []
    tasks = ["ALL"] + sorted(df["task_type"].unique())
    types = ["ALL_TYPES"] + [t for t in ERROR_TYPES if t in set(df["error_type"])]
    for model in MODELS:
        if f"orig_{model}" not in df.columns:
            continue
        tau = taus[model]
        for task in tasks:
            for et in types:
                sub = df
                if task != "ALL":
                    sub = sub[sub["task_type"] == task]
                if et != "ALL_TYPES":
                    sub = sub[sub["error_type"] == et]
                if sub.empty:
                    continue
                o = sub[f"orig_{model}"].to_numpy(dtype=float)
                e = sub[f"edit_{model}"].to_numpy(dtype=float)
                wins = (e < o).astype(float) + 0.5 * (e == o)
                lo, hi = _bootstrap_ci(wins)
                passing = o >= tau
                rows.append({
                    "model": model, "task_type": task, "error_type": et, "n": len(sub), "tau": tau,
                    "pair_acc": wins.mean(), "pair_acc_ci_low": lo, "pair_acc_ci_high": hi,
                    "mean_drop": float((o - e).mean()),
                    "flagged_orig": float((o < tau).mean()), "flagged_edit": float((e < tau).mean()),
                    "n_orig_pass": int(passing.sum()),
                    "caught": float((e[passing] < tau).mean()) if passing.any() else float("nan"),
                })
    return pd.DataFrame(rows).round(3)


def buckets_for(label_hallucinated: bool, meta: dict) -> list[str]:
    """Which analysis groups a real RAGTruth response belongs to."""
    if label_hallucinated:
        types = sorted(set(meta.get("label_types") or [])) or ["unknown type"]
        return ["hallucinated: ALL"] + [f"hallucinated: {t}" for t in types]
    return ["faithful: ALL", f"faithful (quality={meta.get('quality') or 'unknown'})"]


def real_error_breakdown(pred_df: pd.DataFrame, examples_by_id: dict, taus: dict[str, float]) -> pd.DataFrame:
    """flagged_rate is recall for 'hallucinated: ...' groups and the false-positive rate for 'faithful ...' groups.
    A response with several error types counts once in each of its types."""
    score_cols = {"nli_whole_context": "nli_p_entail", "hhem": "hhem_score"}
    records = []
    for row in pred_df.itertuples(index=False):
        ex = examples_by_id.get(str(row.id))
        if ex is None:
            continue
        for bucket in buckets_for(bool(row.label_hallucinated), ex.meta):
            records.append({"task_type": row.task_type, "group": bucket,
                            **{m: getattr(row, c) for m, c in score_cols.items() if hasattr(row, c)}})
    recs = pd.DataFrame(records)
    rows = []
    for model in MODELS:
        if model not in recs.columns:
            continue
        for task in ["ALL"] + sorted(recs["task_type"].unique()):
            sub_task = recs if task == "ALL" else recs[recs["task_type"] == task]
            for group, g in sub_task.groupby("group"):
                s = g[model].to_numpy(dtype=float)
                rows.append({"model": model, "task_type": task, "group": group, "n": len(g),
                             "flagged_rate": float((s < taus[model]).mean()), "mean_score": float(s.mean()),
                             "kind": "recall" if group.startswith("hallucinated") else "false_positive_rate"})
    return pd.DataFrame(rows).round(3)


def _load_taus(summary_path: Path, cfg) -> dict[str, float]:
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)["thresholds"]
        return {m: float(thresholds[m]["tuned_tau"]) for m in MODELS if m in thresholds}
    print(f"[warn] {summary_path} not found: using the fixed threshold {cfg.baseline.tau} for both models")
    return {m: float(cfg.baseline.tau) for m in MODELS}


def run_failure_analysis(cfg, injected_path=None, predictions_path=None, summary_path=None, out_dir=None,
                         use_hhem: bool = True, nli=None, hhem=None, examples=None) -> dict:
    out_dir = Path(out_dir or cfg.eval.results_dir)
    injected_path = Path(injected_path or out_dir / "phase2_injected.jsonl")
    predictions_path = Path(predictions_path or out_dir / "phase1_predictions.csv")
    summary_path = Path(summary_path or out_dir / "phase1_summary.json")
    out_dir.mkdir(parents=True, exist_ok=True)

    taus = _load_taus(summary_path, cfg)
    injections = read_injections(injected_path)
    if not injections:
        raise ValueError(f"{injected_path} is empty. Run `python -m halludetect inject` first.")

    scores = score_injections(cfg, injections, use_hhem=use_hhem, nli=nli, hhem=hhem, out_dir=out_dir)
    scores.to_csv(out_dir / "phase2_injected_scores.csv", index=False)
    metrics = paired_metrics(scores, taus)
    metrics.to_csv(out_dir / "phase2_injected_metrics.csv", index=False)

    show = ["model", "error_type", "n", "pair_acc", "pair_acc_ci_low", "pair_acc_ci_high", "mean_drop",
            "flagged_orig", "caught"]
    print("\nINJECTED ERRORS, all tasks (pair_acc 0.5 = no signal, 1.0 = every edit lowered the score)")
    print(metrics[metrics["task_type"] == "ALL"][show].to_string(index=False))
    by_task = metrics[(metrics["task_type"] != "ALL") & (metrics["error_type"] != "ALL_TYPES")]
    if not by_task.empty:
        print("\nINJECTED ERRORS, pair_acc by task")
        print(by_task.pivot_table(index="error_type", columns=["model", "task_type"], values="pair_acc").round(3).to_string())

    real = None
    if predictions_path.exists():
        if examples is None:
            from halludetect.data.ragtruth import load_ragtruth

            examples = load_ragtruth(cfg.data.ragtruth_dir, "test", None, cfg.seed)
        by_id = {str(e.id): e for e in examples}
        pred_df = pd.read_csv(predictions_path)
        real = real_error_breakdown(pred_df, by_id, taus)
        real.to_csv(out_dir / "phase2_real_error_types.csv", index=False)
        print("\nREAL RAGTRUTH ERRORS, all tasks (recall for hallucinated groups, false-positive rate for faithful groups)")
        print(real[real["task_type"] == "ALL"][["model", "group", "n", "kind", "flagged_rate", "mean_score"]].to_string(index=False))
    else:
        print(f"[warn] {predictions_path} not found: skipped the real-error breakdown")
    print(f"\nSaved phase2_injected_scores.csv, phase2_injected_metrics.csv and phase2_real_error_types.csv in {out_dir}")
    return {"scores": scores, "metrics": metrics, "real": real, "taus": taus}
