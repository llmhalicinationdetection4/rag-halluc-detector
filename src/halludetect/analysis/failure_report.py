from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_subset_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "error_rate": 0.0, "count": 0}

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    errors = sum(yt != yp for yt, yp in zip(y_true, y_pred))
    error_rate = errors / float(len(y_true))

    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "error_rate": float(error_rate),
        "count": len(y_true),
    }


def generate_failure_breakdown(
    records: List[Dict[str, Any]], output_dir: str | Path = "results"
) -> Dict[str, Any]:
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    bucket_keys = ["context_length_bucket", "has_numbers_or_dates", "paraphrase_level", "error_type"]

    report: Dict[str, Any] = {}

    for b_key in bucket_keys:
        if b_key not in df.columns and "meta" in df.columns:
            # Flatten error_type if nested
            df[b_key] = df["meta"].apply(lambda m: m.get(b_key) if isinstance(m, dict) else None)

        if b_key not in df.columns:
            continue

        group_metrics = {}
        for val, group in df.groupby(b_key):
            y_t = group["label_hallucinated"].tolist()
            y_p = group["pred_hallucinated"].tolist()
            group_metrics[str(val)] = compute_subset_metrics(y_t, y_p)

        report[b_key] = group_metrics
        m_df = pd.DataFrame(group_metrics).T
        m_df.to_csv(out_p / f"breakdown_{b_key}.csv")

    # Plot F1 by bucket
    if "context_length_bucket" in report:
        plt.figure(figsize=(6, 4))
        b_data = report["context_length_bucket"]
        labels = list(b_data.keys())
        f1_vals = [b_data[k]["f1"] for k in labels]
        plt.bar(labels, f1_vals, color="#3470a3")
        plt.title("F1 Score by Context Length Bucket")
        plt.xlabel("Bucket")
        plt.ylabel("F1 Score")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(out_p / "f1_by_bucket.png", dpi=200)
        plt.close()

    # Plot F1 by error type
    if "error_type" in report:
        plt.figure(figsize=(8, 4))
        e_data = report["error_type"]
        labels = list(e_data.keys())
        f1_vals = [e_data[k]["f1"] for k in labels]
        plt.bar(labels, f1_vals, color="#a83232")
        plt.title("F1 Score by Injected Error Type")
        plt.xlabel("Error Type")
        plt.ylabel("F1 Score")
        plt.xticks(rotation=30, ha="right")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(out_p / "f1_by_error_type.png", dpi=200)
        plt.close()

    with open(out_p / "failure_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report