"""Phase 3 ablations: which parts of the claim-level verifier matter?

Every variant is scored on the same fixed test subset (`per_task` responses of each task) with the aggregation chosen
in Phase 3, and reported as ROC-AUC per task and pooled. No thresholds are involved, so nothing is tuned on test data.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from halludetect.eval.run_phase3 import h_scores, score_examples
from halludetect.eval.threshold import auc_with_ci

CONFIGS = [
    ("baseline", {}),
    ("top_k_1", {"top_k": 1}),
    ("top_k_5", {"top_k": 5}),
    ("chunk_128", {"chunk_size": 128, "stride": 64}),
    ("chunk_512", {"chunk_size": 512, "stride": 256}),
    ("retrieval_off", {"use_retrieval": False}),
    ("claim_split_off", {"split_answer": False}),
]


def run_ablation(cfg=None, verifier=None, per_task: int = 300, out_dir=None, data_dir=None) -> pd.DataFrame:
    from halludetect.config import load_config
    from halludetect.data.ragtruth import load_ragtruth

    cfg = cfg or load_config()
    out_dir = Path(out_dir or cfg.eval.results_dir)
    summary_path = out_dir / "phase3_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found. Run `python -m halludetect phase3` first.")
    aggregation = json.loads(summary_path.read_text(encoding="utf-8"))["best_aggregation"]
    if verifier is None:
        from halludetect.verifiers.claim_level import ClaimLevelVerifier

        verifier = ClaimLevelVerifier(device=cfg.device)

    test = load_ragtruth(data_dir or cfg.data.ragtruth_dir, "test", None, cfg.seed)
    by_task: dict[str, list] = defaultdict(list)
    for ex in sorted(test, key=lambda e: e.id):
        by_task[ex.task_type].append(ex)
    rng = random.Random(cfg.seed)
    subset = []
    for task in sorted(by_task):
        pool = by_task[task][:]
        rng.shuffle(pool)
        subset.extend(pool[:per_task])
    print(f"Ablation on {len(subset)} test responses ({per_task} per task), aggregation = {aggregation}", flush=True)

    rows = []
    for name, options in CONFIGS:
        records = score_examples(subset, verifier.with_options(**options), out_dir / "cache" / f"ablation_{name}.jsonl", name)
        y = np.array([r["y"] for r in records])
        h = h_scores(records)[aggregation]
        tasks = np.array([r["task"] for r in records])
        for task in ["POOLED"] + sorted(set(tasks)):
            sel = np.ones(len(y), dtype=bool) if task == "POOLED" else tasks == task
            auc, lo, hi = auc_with_ci(y[sel], h[sel], n_boot=500)
            rows.append({"config": name, "task": task, "n": int(sel.sum()), "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(out_dir / "phase3_ablation.csv", index=False)
    print(df.pivot(index="config", columns="task", values="auc").reindex([n for n, _ in CONFIGS]).round(3).to_string())
    print(f"\nSaved phase3_ablation.csv in {out_dir}")
    return df
