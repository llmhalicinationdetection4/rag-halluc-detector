import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report, precision_recall_curve

def analyze_results():
    cache_file = Path("results/cache/phase3_predictions.jsonl")
    response_path = Path("data/RAGTruth/dataset/response.jsonl")
    
    if not cache_file.exists():
        print("Cache file not found.")
        return

    true_labels = {}
    if response_path.exists():
        with open(response_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                d = json.loads(line)
                if d.get("split") == "test":
                    rec_id = str(d.get("raw_id") or d.get("id") or f"test_{idx}")
                    is_halluc = 1 if (d.get("label") == "Hallucinated" or d.get("label") == 1 or (isinstance(d.get("labels"), list) and len(d.get("labels")) > 0)) else 0
                    true_labels[rec_id] = is_halluc

    preds = []
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                rec_id = item["id"]
                if rec_id in true_labels:
                    item["y_true"] = true_labels[rec_id]
                preds.append(item)

    y_true = np.array([p.get("y_true", 0) for p in preds])
    y_scores = np.array([p["score"] for p in preds])

    print(f"Total evaluated records: {len(preds)}")
    print(f"True distribution: Hallucinated={np.sum(y_true)}, Normal={len(y_true) - np.sum(y_true)}")

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, 1 - y_scores)
        
        # Find optimal threshold maximizing F1
        precisions, recalls, thresholds = precision_recall_curve(y_true, 1 - y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        y_pred_binary = ((1 - y_scores) >= best_threshold).astype(int)
        f1 = f1_score(y_true, y_pred_binary)
        
        print(f"\n--- Optimized Metrics ---")
        print(f"ROC-AUC: {auc:.3f}")
        print(f"Optimal Threshold: {best_threshold:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_binary))
    else:
        print("\nNote: Only one class present in y_true.")

if __name__ == "__main__":
    analyze_results()