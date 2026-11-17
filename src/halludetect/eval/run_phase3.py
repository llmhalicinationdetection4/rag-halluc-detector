import json
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from halludetect.verifiers.claim_level import ClaimLevelVerifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class RAGTruthEvaluator:
    def __init__(self, data_dir: str = "data/RAGTruth/dataset", cache_dir: str = "results/cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.cache_dir / "phase3_predictions.jsonl"
        self.verifier = None
        self.test_records = []

    def load_data(self):
        """Loads RAGTruth test split and fails loudly if data is missing."""
        source_info_path = self.data_dir / "source_info.jsonl"
        response_path = self.data_dir / "response.jsonl"
        
        if not source_info_path.exists() or not response_path.exists():
            raise FileNotFoundError(f"Missing RAGTruth files in {self.data_dir}.")
            
        sources = {}
        with open(source_info_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                sources[d["source_id"]] = d["source"]

        with open(response_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                d = json.loads(line)
                if d.get("split") == "test":
                    rec_id = str(d.get("raw_id") or d.get("id") or f"test_{idx}")
                    self.test_records.append({
                        "id": rec_id,
                        "task": d.get("task_type", "qa"),
                        "context": sources.get(d.get("source_id"), ""),
                        "answer": d.get("response", ""),
                        "hallucinated": d.get("label") == "Hallucinated"
                    })
        
        logging.info(f"Loaded {len(self.test_records)} test records from RAGTruth.")
        if len(self.test_records) == 0:
            raise ValueError("No test records found. Check dataset integrity.")

    def _load_checkpoint(self):
        """Loads already-scored records so runs can pause and resume safely."""
        completed = {}
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        completed[item["id"]] = item
        return completed

    def _bootstrap_ci(self, y_true, y_scores, metric_fn, n_bootstraps=1000, seed=42):
        """Calculates 95% Confidence Intervals."""
        rng = np.random.RandomState(seed)
        scores = []
        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(y_scores), len(y_scores))
            if len(np.unique(y_true[indices])) < 2:
                continue
            scores.append(metric_fn(y_true[indices], y_scores[indices]))
        if not scores:
            return (0.0, 0.0)
        return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

    def evaluate(self):
        if not self.test_records:
            self.load_data()
            
        completed = self._load_checkpoint()
        if completed:
            logging.info(f"Resuming from checkpoint: {len(completed)}/{len(self.test_records)} already scored.")

        remaining = [r for r in self.test_records if r["id"] not in completed]
        
        if remaining:
            if self.verifier is None:
                logging.info("Initializing ClaimLevelVerifier (Models loading...)")
                self.verifier = ClaimLevelVerifier()

            remaining.sort(key=lambda x: len(x["context"]))
            
            with open(self.checkpoint_file, "a", encoding="utf-8") as out_f:
                for record in tqdm(remaining, desc="Scoring Remaining"):
                    result = self.verifier.verify(record["context"], record["answer"], task=record["task"])
                    pred_entry = {
                        "id": record["id"],
                        "task": record["task"],
                        "y_true": 1 if record["hallucinated"] else 0,
                        "score": float(result["score"])
                    }
                    out_f.write(json.dumps(pred_entry) + "\n")
                    out_f.flush()
                    completed[record["id"]] = pred_entry

        predictions = list(completed.values())
        tasks = sorted(set(p["task"] for p in predictions)) + ["OVERALL"]
        results_by_task = {}
        
        for task in tasks:
            task_preds = predictions if task == "OVERALL" else [p for p in predictions if p["task"] == task]
            y_true = np.array([p["y_true"] for p in task_preds])
            y_scores = np.array([p["score"] for p in task_preds])
            y_pred_binary = (y_scores < 0.990).astype(int)
            
            try:
                auc = roc_auc_score(y_true, 1 - y_scores)
                auc_ci = self._bootstrap_ci(y_true, 1 - y_scores, roc_auc_score)
                f1 = f1_score(y_true, y_pred_binary)
                flag_all_f1 = f1_score(y_true, np.ones_like(y_true))
                
                results_by_task[task] = {
                    "AUC": f"{auc:.3f} [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}]",
                    "F1": f"{f1:.3f} (Flag All: {flag_all_f1:.3f})",
                    "n_samples": len(task_preds)
                }
            except ValueError:
                results_by_task[task] = {"AUC": "N/A", "F1": "N/A", "n_samples": len(task_preds)}
                
        print("\n=== EVALUATION RESULTS ===")
        for task, mets in results_by_task.items():
            print(f"Task: {task:<10} | n={mets['n_samples']:<4} | AUC: {mets['AUC']:<20} | F1: {mets['F1']}")

if __name__ == "__main__":
    evaluator = RAGTruthEvaluator()
    evaluator.evaluate()