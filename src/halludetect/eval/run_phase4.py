import json
import time
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from halludetect.config import load_config
from halludetect.verifiers.claim_level import ClaimLevelVerifier

def calc_ece(y_true, probs, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs <= bins[i+1])
        if np.any(mask):
            acc = np.mean(y_true[mask] == (probs[mask] > 0.5))
            conf = np.mean(probs[mask])
            ece += (len(probs[mask]) / len(probs)) * np.abs(acc - conf)
    return ece

def run_phase4():
    train_cache = Path("results/cache/phase3_train_cache.jsonl")
    test_cache = Path("results/cache/phase3_test_cache.jsonl")
    if not train_cache.exists() or not test_cache.exists():
        print("Run phase3 first to generate caches.")
        return
        
    def load_features(path):
        X, y = [], []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                X.append([d["aggs"]["max_contra"], d["aggs"]["min_entail"], d["aggs"]["mean_entail"]])
                y.append(d["y_true"])
        return np.array(X), np.array(y)

    X_train, y_train = load_features(train_cache)
    X_test, y_test = load_features(test_cache)
    
    print("--- Phase 4: Calibration & Abstention ---")
    calibrator = LogisticRegression(class_weight="balanced")
    calibrator.fit(X_train, y_train)
    
    test_probs = calibrator.predict_proba(X_test)[:, 1]
    ece = calc_ece(y_test, test_probs)
    print(f"Test ECE: {ece:.3f}")
    
    abstain_mask = (test_probs > 0.3) & (test_probs < 0.7)
    covered = ~abstain_mask
    coverage = np.mean(covered)
    acc_covered = np.mean(y_test[covered] == (test_probs[covered] > 0.5))
    print(f"Abstention Band (0.3 - 0.7): Coverage {coverage*100:.1f}%, Accuracy on Covered {acc_covered*100:.1f}%")

    print("\n--- Latency Check ---")
    cfg = load_config()
    verifier = ClaimLevelVerifier(device=cfg.device)
    latencies = []
    with open(test_cache) as f:
        # Just run 20 for latency stats
        for line in list(f)[:20]:
            d = json.loads(line)
            # Dummy short strings if not saved in cache, or modify cache to save context/answer
            # Because Phase3 cache only saves hashes, we do a raw dummy latency test matching typical sizes
            ctx, ans = " ".join(["word"]*500), " ".join(["word"]*50)
            start = time.perf_counter()
            verifier.verify(ctx, ans)
            latencies.append(time.perf_counter() - start)
            
    print(f"Latency P50: {np.percentile(latencies, 50):.3f}s")
    print(f"Latency P95: {np.percentile(latencies, 95):.3f}s")

if __name__ == "__main__":
    run_phase4()