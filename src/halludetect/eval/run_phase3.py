import json
import logging
import hashlib
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from halludetect.config import load_config
from halludetect.data.ragtruth import load_ragtruth
from halludetect.verifiers.claim_level import ClaimLevelVerifier
from halludetect.eval.span_mapping import map_spans_to_claims

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def get_hash(ctx: str, ans: str) -> str:
    return hashlib.md5((ctx + ans).encode('utf-8')).hexdigest()

def get_aggregations(claims_result):
    if not claims_result: return {"max_contra": 0.0, "min_entail": 1.0, "mean_entail": 1.0}
    p_con = [c["p_contra"] for c in claims_result]
    p_ent = [c["p_entail"] for c in claims_result]
    return {
        "max_contra": max(p_con),
        "min_entail": min(p_ent),
        "mean_entail": sum(p_ent) / len(p_ent)
    }

def tune_thresholds(y_true, scores_dict):
    best_f1, best_th, best_agg = 0, 0.5, "max_contra"
    for agg, scores in scores_dict.items():
        thresholds = np.linspace(0.0, 1.0, 101)
        for th in thresholds:
            # If entailment, hallucinated when score < th. If contra, hallucinated when score >= th.
            preds = (np.array(scores) < th).astype(int) if "entail" in agg else (np.array(scores) >= th).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th, best_agg = f1, th, agg
    return best_agg, best_th, best_f1

def bootstrap_auc_diff(y_true, scores_claim, scores_hhem, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2: continue
        auc_claim = roc_auc_score(y_true[idx], scores_claim[idx])
        auc_hhem = roc_auc_score(y_true[idx], scores_hhem[idx])
        diffs.append(auc_claim - auc_hhem)
    if not diffs: return 0.0, 0.0
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

def process_split(data, verifier, cache_file):
    cache = {}
    if cache_file.exists():
        with open(cache_file, "r") as f:
            for line in f:
                d = json.loads(line)
                cache[d["hash"]] = d
                
    results = []
    with open(cache_file, "a") as f:
        for ex in tqdm(data, desc="Processing..."):
            h = get_hash(ex.context, ex.answer)
            if h in cache:
                res = cache[h]
            else:
                raw_claims = verifier.verify(ex.context, ex.answer, ex.task_type)["claims"]
                res = {
                    "id": ex.id, "hash": h, "task": ex.task_type,
                    "y_true": 1 if ex.label_hallucinated else 0,
                    "aggs": get_aggregations(raw_claims),
                    "claims": raw_claims,
                    "label_spans": ex.meta.get("label_spans", [])
                }
                f.write(json.dumps(res) + "\n")
                f.flush()
            results.append(res)
    return results

def run_evaluation(max_examples=None):
    cfg = load_config()
    verifier = ClaimLevelVerifier(device=cfg.device)
    cache_dir = Path("results/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("--- Phase 3: Train Tuning ---")
    train_data = load_ragtruth(cfg.data.ragtruth_dir, "train", max_examples, cfg.seed)
    train_res = process_split(train_data, verifier, cache_dir / "phase3_train_cache.jsonl")
    
    y_train = [r["y_true"] for r in train_res]
    scores_train = {agg: [r["aggs"][agg] for r in train_res] for agg in ["max_contra", "min_entail", "mean_entail"]}
    
    best_agg, best_th, train_f1 = tune_thresholds(y_train, scores_train)
    logging.info(f"Tuned on TRAIN: Agg={best_agg}, Tau={best_th:.3f}, F1={train_f1:.3f}")
    
    with open("results/phase3_summary.json", "w") as f:
        json.dump({"best_aggregation": best_agg, "best_threshold": best_th}, f)

    logging.info("--- Phase 3: Test Evaluation ---")
    test_data = load_ragtruth(cfg.data.ragtruth_dir, "test", max_examples, cfg.seed)
    test_res = process_split(test_data, verifier, cache_dir / "phase3_test_cache.jsonl")
    
    hhem_map = {}
    if Path("results/phase1_predictions.csv").exists():
        df = pd.read_csv("results/phase1_predictions.csv")
        hhem_map = {str(row["id"]): row["hhem_score"] for _, row in df.iterrows()}

    tasks = sorted(list(set(r["task"] for r in test_res)))
    for task in tasks + ["POOLED"]:
        subset = test_res if task == "POOLED" else [r for r in test_res if r["task"] == task]
        if not subset: continue
        
        y_true = np.array([r["y_true"] for r in subset])
        # Higher HHEM = Faithful. So AUC needs faith scores:
        scores_claim_faith = np.array([r["aggs"]["mean_entail"] if "entail" in best_agg else (1.0 - r["aggs"]["max_contra"]) for r in subset])
        hhem_scores = np.array([hhem_map.get(str(r["id"]), 0.0) for r in subset])
        
        preds = (np.array([r["aggs"][best_agg] for r in subset]) < best_th).astype(int) if "entail" in best_agg else (np.array([r["aggs"][best_agg] for r in subset]) >= best_th).astype(int)
        
        auc = roc_auc_score(y_true, scores_claim_faith)
        f1 = f1_score(y_true, preds, zero_division=0)
        flag_all_f1 = f1_score(y_true, np.ones_like(y_true))
        
        print(f"\n[{task}] n={len(subset)}")
        print(f"  Claim-Level AUC: {auc:.3f}")
        print(f"  Claim-Level F1:  {f1:.3f} (Flag-All F1: {flag_all_f1:.3f})")
        
        if len(np.unique(hhem_scores)) > 1:
            diff_ci = bootstrap_auc_diff(y_true, scores_claim_faith, hhem_scores)
            print(f"  HHEM AUC:        {roc_auc_score(y_true, hhem_scores):.3f}")
            print(f"  Claim - HHEM 95% CI: [{diff_ci[0]:.3f}, {diff_ci[1]:.3f}]")
            
        # Claim level metrics
        claim_y_true, claim_y_pred = [], []
        for r in subset:
            if not r["claims"]: continue
            # 1. Map ground truth spans to the generated claims
            true_claim_flags = map_spans_to_claims(" ".join([c["text"] for c in r["claims"]]), [c["text"] for c in r["claims"]], r["label_spans"])
            claim_y_true.extend([1 if f else 0 for f in true_claim_flags])
            # 2. Predict hallucinated if claim entails less than threshold (or contra is high)
            for c in r["claims"]:
                if "entail" in best_agg:
                    claim_y_pred.append(1 if c["p_entail"] < best_th else 0)
                else:
                    claim_y_pred.append(1 if c["p_contra"] >= best_th else 0)
                    
        if claim_y_true:
            cP = precision_score(claim_y_true, claim_y_pred, zero_division=0)
            cR = recall_score(claim_y_true, claim_y_pred, zero_division=0)
            cF1 = f1_score(claim_y_true, claim_y_pred, zero_division=0)
            print(f"  Claim-Level Granular -> P: {cP:.3f}, R: {cR:.3f}, F1: {cF1:.3f}")

if __name__ == "__main__":
    run_evaluation()