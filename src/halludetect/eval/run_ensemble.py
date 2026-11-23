"""Phase 4b: does combining HHEM with the claim-level scores beat HHEM alone?

Motivation: HHEM almost never notices an entity swap (Phase 2), while NLI-style scores rank those edits well, so the
two kinds of score may complement each other.

Protocol (nothing is chosen on the test split):
  * Features per response (all oriented 'higher = more likely hallucinated'): h_hhem = 1 - HHEM score,
    h_nli = 1 - P(entailment) of the whole-context NLI, and the four claim-level aggregations from Phase 3.
  * HHEM and whole-context NLI scores for the Phase 3 TRAIN subset are computed here (cached); TEST scores are the
    Phase 1 predictions.
  * A logistic regression is fitted per feature set on TRAIN. The feature set and regularisation C are chosen by
    5-fold cross-validated TRAIN AUC; the decision threshold is tuned on out-of-fold TRAIN probabilities.
  * On TEST: ROC-AUC per task with a paired bootstrap difference against HHEM, and F1 next to the 'flag everything' F1.
  * On the Phase 2 injected set: how often each error lowers the score (pair accuracy), for HHEM, the claim-level score
    alone and the combined model.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from halludetect.calibration import LogisticCalibrator
from halludetect.eval.run_eval import _cached
from halludetect.eval.run_phase3 import H_COLUMNS, _features_frame, score_examples
from halludetect.eval.threshold import auc_with_ci, best_f1_threshold, f1_at, flag_all_f1, paired_auc_diff_ci, safe_auc
from halludetect.verifiers.claim_level import AGGREGATIONS, hallucination_score

CLAIM = list(H_COLUMNS.values())
FEATURE_SETS = {
    "hhem": ["h_hhem"],
    "claim": CLAIM,
    "hhem+claim": ["h_hhem"] + CLAIM,
    "hhem+nli+claim": ["h_hhem", "h_nli"] + CLAIM,
}
CANDIDATES = ["hhem+claim", "hhem+nli+claim"]  # the sets that are actually combinations; 'hhem' and 'claim' are references
C_GRID = (0.3, 1.0, 10.0)
TASKS = ["POOLED", "QA", "Summary", "Data2txt"]


def _cv_probabilities(X: np.ndarray, y: np.ndarray, C: float, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    folds = int(min(5, y.sum(), len(y) - y.sum()))
    if folds < 2:
        return LogisticCalibrator.fit(X, y, [f"f{i}" for i in range(X.shape[1])], C=C).predict_proba(X)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(LogisticRegression(C=C, max_iter=2000), X, y, cv=cv, method="predict_proba")[:, 1]


def _train_scores(cfg, out_dir: Path, data_dir, ids: list[str], hhem, nli, device: str | None):
    """HHEM consistency scores and NLI P(entailment) for the TRAIN responses with these ids (cached)."""
    import hashlib

    from halludetect.data.ragtruth import load_ragtruth

    train = {str(e.id): e for e in load_ragtruth(data_dir or cfg.data.ragtruth_dir, "train", None, cfg.seed)}
    missing = [i for i in ids if i not in train]
    if missing:
        raise ValueError(f"{len(missing)} train ids from phase3_features_train.csv are not in the RAGTruth train split")
    pairs = [(train[i].context, train[i].answer) for i in ids]
    key = hashlib.sha1("|".join(ids).encode()).hexdigest()[:16]
    cache_dir = out_dir / "cache" if (hhem is None and nli is None) else None
    if hhem is None:
        from halludetect.config import get_device
        from halludetect.models.hhem import HHEMModel

        hhem = HHEMModel(cfg.models.hhem, device=get_device(device or cfg.device), batch_size=cfg.models.batch_size)
    if nli is None:
        from halludetect.config import get_device
        from halludetect.models.nli import NLIModel

        m = cfg.models
        nli = NLIModel(m.nli, device=get_device(device or cfg.device), max_length=m.max_length,
                       batch_size=m.batch_size, fallback_name=m.nli_fallback)
    print(f"Scoring {len(pairs)} train responses with HHEM and the whole-context NLI", flush=True)
    hhem_scores = _cached(cache_dir, f"ensemble_train_hhem_{key}", len(pairs), lambda: hhem.predict_consistency(pairs))
    nli_scores = _cached(cache_dir, f"ensemble_train_nli_{key}", len(pairs), lambda: np.asarray(nli.predict_proba(pairs))[:, 0])
    return np.asarray(hhem_scores, dtype=float), np.asarray(nli_scores, dtype=float)


def _pair_acc(p_orig: np.ndarray, p_edit: np.ndarray, n_boot: int = 500, seed: int = 42):
    wins = (p_edit > p_orig).astype(float) + 0.5 * (p_edit == p_orig)
    rng = np.random.default_rng(seed)
    means = [wins[rng.integers(0, len(wins), len(wins))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(wins.mean()), float(lo), float(hi)


def _injected_table(cfg, out_dir: Path, model: LogisticCalibrator, tau: float, verifier, injected_path, scores_path) -> pd.DataFrame | None:
    from halludetect.inject.injectors import read_injections

    injected_path = Path(injected_path or out_dir / "phase2_injected.jsonl")
    scores_path = Path(scores_path or out_dir / "phase2_injected_scores.csv")
    if not injected_path.exists() or not scores_path.exists():
        print(f"[warn] {injected_path.name} or {scores_path.name} not found: the injected-error comparison is skipped "
              "(run `inject` and `analyze` first)")
        return None
    injections = read_injections(injected_path)
    scores = pd.read_csv(scores_path)
    aligned = (len(scores) == len(injections)
               and not (scores["example_id"].astype(str).to_numpy() != np.array([str(i.example_id) for i in injections])).any()
               and not (scores["description"].astype(str).to_numpy() != np.array([str(i.description) for i in injections])).any())
    if not aligned:
        print("[warn] phase2_injected_scores.csv does not line up with phase2_injected.jsonl (ids or edit descriptions differ; "
              "re-run `analyze`): the injected-error comparison is skipped")
        return None
    if verifier is None:
        from halludetect.verifiers.claim_level import ClaimLevelVerifier

        verifier = ClaimLevelVerifier(device=cfg.device)

    def examples(pairs):
        return [SimpleNamespace(id=f"{i}", context=c, answer=a, task_type=t, label_hallucinated=False, meta={})
                for i, (c, a, t) in enumerate(pairs)]

    orig_unique = {}
    for inj in injections:
        orig_unique.setdefault(str(inj.example_id), (inj.context, inj.original, inj.task_type))
    orig_ids = list(orig_unique)
    print(f"Claim-level scores for {len(orig_ids)} originals and {len(injections)} edited responses", flush=True)
    rec_o = score_examples(examples([orig_unique[i] for i in orig_ids]), verifier, out_dir / "cache" / "ensemble_inj_orig.jsonl", "originals")
    rec_e = score_examples(examples([(i.context, i.edited, i.task_type) for i in injections]), verifier, out_dir / "cache" / "ensemble_inj_edit.jsonl", "edits")
    pos = {i: k for k, i in enumerate(orig_ids)}
    claim_o = _features_frame(rec_o)[CLAIM].to_numpy()[[pos[str(i.example_id)] for i in injections]]
    claim_e = _features_frame(rec_e)[CLAIM].to_numpy()

    def design(h_hhem, h_nli, claim):
        cols = {"h_hhem": h_hhem, "h_nli": h_nli, **{c: claim[:, k] for k, c in enumerate(CLAIM)}}
        return np.column_stack([cols[name] for name in model.feature_names])

    X_o = design(1 - scores["orig_hhem"].to_numpy(), 1 - scores["orig_nli_whole_context"].to_numpy(), claim_o)
    X_e = design(1 - scores["edit_hhem"].to_numpy(), 1 - scores["edit_nli_whole_context"].to_numpy(), claim_e)
    p_o, p_e = model.predict_proba(X_o), model.predict_proba(X_e)
    h_o, h_e = 1 - scores["orig_hhem"].to_numpy(), 1 - scores["edit_hhem"].to_numpy()
    summary_path = out_dir / "phase3_summary.json"
    best = (json.loads(summary_path.read_text(encoding="utf-8")).get("best_aggregation") if summary_path.exists() else None) or "min_entail"
    if best not in AGGREGATIONS:
        best = "min_entail"
    c_o, c_e = claim_o[:, CLAIM.index(H_COLUMNS[best])], claim_e[:, CLAIM.index(H_COLUMNS[best])]
    types = np.array([i.error_type for i in injections])
    rows = []
    for et in ["ALL_TYPES"] + [t for t in dict.fromkeys(types)]:
        sel = np.ones(len(types), dtype=bool) if et == "ALL_TYPES" else types == et
        ens = _pair_acc(p_o[sel], p_e[sel])
        passing = p_o[sel] < tau
        rows.append({"error_type": et, "n": int(sel.sum()),
                     "hhem_pair_acc": _pair_acc(h_o[sel], h_e[sel])[0], "claim_pair_acc": _pair_acc(c_o[sel], c_e[sel])[0],
                     "ensemble_pair_acc": ens[0], "ensemble_ci_low": ens[1], "ensemble_ci_high": ens[2],
                     "ensemble_caught": float((p_e[sel][passing] >= tau).mean()) if passing.any() else float("nan")})
    return pd.DataFrame(rows).round(4)


def run_ensemble(cfg=None, out_dir=None, data_dir=None, predictions_path=None, hhem=None, nli=None, verifier=None,
                 device: str | None = None, injected: bool = True, injected_path=None, scores_path=None, n_boot: int = 1000) -> dict:
    from halludetect.config import load_config

    cfg = cfg or load_config()
    out_dir = Path(out_dir or cfg.eval.results_dir)
    for name in ("phase3_features_train.csv", "phase3_features_test.csv"):
        if not (out_dir / name).exists():
            raise FileNotFoundError(f"{out_dir / name} not found. Run `python -m halludetect phase3` first.")
    predictions_path = Path(predictions_path or out_dir / "phase1_predictions.csv")
    if not predictions_path.exists():
        raise FileNotFoundError(f"{predictions_path} not found (the Phase 1 test predictions).")

    train = pd.read_csv(out_dir / "phase3_features_train.csv")
    test = pd.read_csv(out_dir / "phase3_features_test.csv")
    train["id"], test["id"] = train["id"].astype(str), test["id"].astype(str)
    p1 = pd.read_csv(predictions_path)
    p1["id"] = p1["id"].astype(str)
    test = test.merge(p1[["id", "hhem_score", "nli_p_entail"]], on="id", how="left")
    if test["hhem_score"].isna().any():
        raise ValueError(f"{int(test['hhem_score'].isna().sum())} test responses have no Phase 1 scores in {predictions_path.name}")
    test["h_hhem"], test["h_nli"] = 1 - test["hhem_score"], 1 - test["nli_p_entail"]

    hhem_tr, nli_tr = _train_scores(cfg, out_dir, data_dir, train["id"].tolist(), hhem, nli, device)
    train["h_hhem"], train["h_nli"] = 1 - hhem_tr, 1 - nli_tr
    y_tr, y_te = train["y"].to_numpy(dtype=int), test["y"].to_numpy(dtype=int)

    # ---- choose the feature set and C on TRAIN only
    cv_rows, best_per_set = [], {}
    for name, feats in FEATURE_SETS.items():
        X = train[feats].to_numpy(dtype=float)
        scored = [(safe_auc(y_tr, _cv_probabilities(X, y_tr, C, cfg.seed)), C) for C in C_GRID]
        auc, C = max(scored, key=lambda t: (t[0], -t[1]))
        best_per_set[name] = (auc, C)
        cv_rows.append({"feature_set": name, "n_features": len(feats), "cv_auc": auc, "C": C})
    chosen = max(CANDIDATES, key=lambda n: (best_per_set[n][0], -len(FEATURE_SETS[n])))
    print("\nTRAIN 5-fold cross-validated AUC:")
    print(pd.DataFrame(cv_rows).round(4).to_string(index=False))
    print(f"Chosen on TRAIN: {chosen} (C = {best_per_set[chosen][1]})")

    # ---- fit, tune the threshold on out-of-fold TRAIN probabilities, evaluate on TEST
    fitted, taus = {}, {}
    for name, feats in FEATURE_SETS.items():
        C = best_per_set[name][1]
        X = train[feats].to_numpy(dtype=float)
        fitted[name] = LogisticCalibrator.fit(X, y_tr, feats, C=C)
        taus[name] = best_f1_threshold(y_tr, _cv_probabilities(X, y_tr, C, cfg.seed))[0]
    h_hhem_te = test["h_hhem"].to_numpy()
    tasks_te = test["task"].to_numpy()
    rows = []
    for name, feats in FEATURE_SETS.items():
        p = fitted[name].predict_proba(test[feats].to_numpy(dtype=float))
        for task in TASKS:
            sel = np.ones(len(y_te), dtype=bool) if task == "POOLED" else tasks_te == task
            auc, lo, hi = auc_with_ci(y_te[sel], p[sel], n_boot=n_boot)
            diff, dlo, dhi = paired_auc_diff_ci(y_te[sel], p[sel], h_hhem_te[sel], n_boot=n_boot)
            precision, recall, f1 = f1_at(y_te[sel], p[sel] >= taus[name])
            rows.append({"feature_set": name, "task": task, "n": int(sel.sum()), "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi,
                         "auc_minus_hhem": diff, "diff_ci_low": dlo, "diff_ci_high": dhi, "tau": taus[name],
                         "precision": precision, "recall": recall, "f1": f1, "flag_all_f1": flag_all_f1(y_te[sel]),
                         "chosen": name == chosen})
    by_task = pd.DataFrame(rows).round(4)
    by_task.to_csv(out_dir / "ensemble_by_task.csv", index=False)
    fitted[chosen].save(out_dir / "ensemble_model.json")
    summary = {"chosen_feature_set": chosen, "features": FEATURE_SETS[chosen], "C": best_per_set[chosen][1], "tau": taus[chosen],
               "cv_auc_by_feature_set": {r["feature_set"]: round(r["cv_auc"], 4) for r in cv_rows}, "n_train": len(train),
               "note": "feature set, C and tau were chosen on the train subset only (cross-validated); test results are in ensemble_by_task.csv"}
    (out_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTEST (pooled and per task); auc_minus_hhem is a paired bootstrap difference against HHEM's own score")
    show = by_task[by_task["feature_set"].isin(["hhem", chosen])]
    print(show[["feature_set", "task", "auc", "auc_minus_hhem", "diff_ci_low", "diff_ci_high", "f1", "flag_all_f1"]].to_string(index=False))

    inj = None
    if injected:
        inj = _injected_table(cfg, out_dir, fitted[chosen], taus[chosen], verifier, injected_path, scores_path)
        if inj is not None:
            inj.to_csv(out_dir / "ensemble_injected.csv", index=False)
            print("\nINJECTED ERRORS: pair accuracy (0.5 = no signal)")
            print(inj[["error_type", "n", "hhem_pair_acc", "claim_pair_acc", "ensemble_pair_acc", "ensemble_ci_low", "ensemble_ci_high", "ensemble_caught"]].to_string(index=False))
    print(f"\nSaved ensemble_by_task.csv, ensemble_summary.json, ensemble_model.json" + (", ensemble_injected.csv" if inj is not None else "") + f" in {out_dir}")
    return {"by_task": by_task, "summary": summary, "injected": inj, "cv": pd.DataFrame(cv_rows)}
