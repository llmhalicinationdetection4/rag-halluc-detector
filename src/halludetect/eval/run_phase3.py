"""Phase 3 evaluation of the claim-level verifier on RAGTruth.

Protocol (nothing is tuned on the test split):
  1. Score a seeded random subset of the TRAIN split and the TEST split with the claim-level verifier.
  2. Pick the answer-level aggregation with the best TRAIN ROC-AUC, then its threshold by TRAIN F1.
  3. Report the TEST split per task and pooled: ROC-AUC with a bootstrap interval, F1 next to the 'flag everything'
     F1, and a paired comparison with the Phase 1 HHEM and NLI scores on the very same responses.
  4. Claim-level labels come from RAGTruth's span annotations; the claim threshold is also tuned on TRAIN.
Scores are cached per example (results/cache), so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from halludetect.eval.span_mapping import label_claims
from halludetect.eval.threshold import auc_with_ci, best_f1_threshold, f1_at, flag_all_f1, paired_auc_diff_ci, safe_auc
from halludetect.verifiers.claim_level import AGGREGATIONS, hallucination_score

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional
    def tqdm(iterable, **kwargs):
        return iterable

H_COLUMNS = {name: f"h_{name}" for name in AGGREGATIONS}


def _cache_key(signature: dict, ex) -> str:
    h = hashlib.sha1(json.dumps(signature, sort_keys=True).encode("utf-8"))
    h.update(ex.context.encode("utf-8", "replace"))
    h.update(b"\x1e")
    h.update(ex.answer.encode("utf-8", "replace"))
    return h.hexdigest()


def score_examples(examples, verifier, cache_path, desc: str = "claim-level") -> list[dict]:
    """One record per example: id, task, label, claims (with offsets and scores) and answer-level scores."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cached[rec["key"]] = rec
    signature = verifier.signature()
    records = []
    with open(cache_path, "a", encoding="utf-8") as out:
        for ex in tqdm(examples, desc=desc, mininterval=20):
            key = _cache_key(signature, ex)
            rec = cached.get(key)
            if rec is None:
                result = verifier.verify(ex.context, ex.answer, ex.task_type)
                claims = [{k: v for k, v in c.items() if k != "evidence"} for c in result["claims"]]
                rec = {"key": key, "id": ex.id, "task": ex.task_type, "y": int(ex.label_hallucinated),
                       "claims": claims, "aggs": result["aggs"]}
                out.write(json.dumps(rec) + "\n")
                out.flush()
            records.append({**rec, "spans": ex.meta.get("label_spans", [])})
    return records


def h_scores(records: list[dict]) -> dict[str, np.ndarray]:
    return {name: np.array([hallucination_score(r["aggs"], name) for r in records], dtype=float) for name in AGGREGATIONS}


def _features_frame(records: list[dict]) -> pd.DataFrame:
    h = h_scores(records)
    df = pd.DataFrame({"id": [r["id"] for r in records], "task": [r["task"] for r in records],
                       "y": [r["y"] for r in records]})
    for name, col in H_COLUMNS.items():
        df[col] = h[name]
    return df


def _claim_table(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-claim hallucination label (from the annotated spans), hallucination score (1 - P(entail)) and task."""
    labels, scores, tasks = [], [], []
    for r in records:
        for claim, label in zip(r["claims"], label_claims(r["claims"], r["spans"])):
            labels.append(label)
            scores.append(1.0 - claim["p_entail"])
            tasks.append(r["task"])
    return np.array(labels, dtype=int), np.array(scores, dtype=float), tasks


def _load_phase1(predictions_path: Path):
    if not predictions_path.exists():
        return None
    df = pd.read_csv(predictions_path)
    df["id"] = df["id"].astype(str)
    return df.set_index("id")


def run_phase3(cfg=None, verifier=None, max_train: int | None = 3000, max_test: int | None = None, out_dir=None,
               predictions_path=None, data_dir=None) -> dict:
    from halludetect.config import load_config
    from halludetect.data.ragtruth import load_ragtruth

    cfg = cfg or load_config()
    out_dir = Path(out_dir or cfg.eval.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = data_dir or cfg.data.ragtruth_dir
    if verifier is None:
        from halludetect.verifiers.claim_level import ClaimLevelVerifier

        verifier = ClaimLevelVerifier(device=cfg.device)

    train = load_ragtruth(data_dir, "train", max_train, cfg.seed)
    test = load_ragtruth(data_dir, "test", max_test, cfg.seed)
    if not train or not test:
        raise ValueError(f"Empty split (train={len(train)}, test={len(test)}). Check {data_dir}.")
    print(f"Scoring {len(train)} train and {len(test)} test responses with the claim-level verifier", flush=True)
    rec_train = score_examples(train, verifier, out_dir / "cache" / "phase3_train.jsonl", "train")
    rec_test = score_examples(test, verifier, out_dir / "cache" / "phase3_test.jsonl", "test")
    _features_frame(rec_train).to_csv(out_dir / "phase3_features_train.csv", index=False)
    _features_frame(rec_test).to_csv(out_dir / "phase3_features_test.csv", index=False)

    # ---- choose the aggregation and its threshold on TRAIN only
    y_tr = np.array([r["y"] for r in rec_train])
    h_tr = h_scores(rec_train)
    train_auc = {name: safe_auc(y_tr, h_tr[name]) for name in AGGREGATIONS}
    best = max(train_auc, key=lambda n: -1.0 if np.isnan(train_auc[n]) else train_auc[n])
    tau, train_f1 = best_f1_threshold(y_tr, h_tr[best])
    cy_tr, cs_tr, _ = _claim_table(rec_train)
    claim_tau = best_f1_threshold(cy_tr, cs_tr)[0] if len(np.unique(cy_tr)) > 1 else None

    # ---- evaluate on TEST
    predictions_path = Path(predictions_path or out_dir / "phase1_predictions.csv")
    phase1 = _load_phase1(predictions_path)
    y_te = np.array([r["y"] for r in rec_test])
    h_te = h_scores(rec_test)[best]
    ids = np.array([str(r["id"]) for r in rec_test])
    tasks_te = np.array([r["task"] for r in rec_test])
    baselines = None
    if phase1 is not None:
        matched = np.array([i in phase1.index for i in ids])
        if matched.mean() >= 0.99:
            baselines = {"hhem": 1.0 - phase1.loc[ids[matched], "hhem_score"].to_numpy(dtype=float),
                         "nli": 1.0 - phase1.loc[ids[matched], "nli_p_entail"].to_numpy(dtype=float)}
            if not matched.all():
                print(f"[warn] {int((~matched).sum())} test responses have no Phase 1 scores and are left out of the baseline comparison")
        else:
            print(f"[warn] only {matched.mean():.0%} of the test responses match results/phase1_predictions.csv ids: "
                  "the HHEM/NLI comparison is skipped")
    else:
        print(f"[warn] {predictions_path} not found: the HHEM/NLI comparison is skipped")

    rows, claim_rows = [], []
    cy_te, cs_te, ctasks = _claim_table(rec_test)
    ctasks = np.array(ctasks)
    for task in ["POOLED"] + sorted(set(tasks_te)):
        sel = np.ones(len(y_te), dtype=bool) if task == "POOLED" else tasks_te == task
        y, h = y_te[sel], h_te[sel]
        auc, lo, hi = auc_with_ci(y, h)
        precision, recall, f1 = f1_at(y, h >= tau)
        row = {"task": task, "n": int(sel.sum()), "hallucinated_rate": float(y.mean()), "aggregation": best,
               "tau": tau, "precision": precision, "recall": recall, "f1": f1, "flag_all_f1": flag_all_f1(y),
               "flagged_rate": float((h >= tau).mean()), "claim_auc": auc, "claim_auc_ci_low": lo, "claim_auc_ci_high": hi}
        if baselines is not None and matched[sel].all():
            sel_m = sel[matched]
            for name, scores in baselines.items():
                row[f"{name}_auc"] = safe_auc(y, scores[sel_m])
                diff, dlo, dhi = paired_auc_diff_ci(y, h, scores[sel_m])
                row[f"auc_diff_vs_{name}"], row[f"auc_diff_vs_{name}_ci_low"], row[f"auc_diff_vs_{name}_ci_high"] = diff, dlo, dhi
        rows.append(row)
        if claim_tau is not None:
            csel = np.ones(len(cy_te), dtype=bool) if task == "POOLED" else ctasks == task
            if csel.any():
                p, r, f = f1_at(cy_te[csel], cs_te[csel] >= claim_tau)
                claim_rows.append({"task": task, "n_claims": int(csel.sum()), "claim_hallucinated_rate": float(cy_te[csel].mean()),
                                   "claim_tau": claim_tau, "precision": p, "recall": r, "f1": f, "flag_all_f1": flag_all_f1(cy_te[csel])})

    by_task = pd.DataFrame(rows).round(4)
    by_task.to_csv(out_dir / "phase3_by_task.csv", index=False)
    claim_df = pd.DataFrame(claim_rows).round(4)
    claim_df.to_csv(out_dir / "phase3_claim_level.csv", index=False)
    summary = {"best_aggregation": best, "best_threshold": tau, "train_f1_at_threshold": train_f1,
               "train_auc_by_aggregation": {k: (None if np.isnan(v) else round(v, 4)) for k, v in train_auc.items()},
               "claim_threshold": claim_tau, "n_train": len(rec_train), "n_test": len(rec_test), "seed": cfg.seed,
               "verifier": verifier.signature(),
               "note": "aggregation and thresholds were tuned on a seeded random subset of the train split only"}
    (out_dir / "phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nAggregation chosen on TRAIN: {best} (train AUC {train_auc[best]:.3f}), tau {tau:.4f}")
    print("\nTEST, answer level")
    print(by_task.to_string(index=False))
    if len(claim_df):
        print("\nTEST, claim level (labels from RAGTruth span annotations)")
        print(claim_df.to_string(index=False))
    print(f"\nSaved phase3_by_task.csv, phase3_claim_level.csv, phase3_summary.json and the feature files in {out_dir}")
    return {"by_task": by_task, "claim_level": claim_df, "summary": summary}
