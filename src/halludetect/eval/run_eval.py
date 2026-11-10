from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from halludetect.analysis.buckets import tag_examples_with_buckets
from halludetect.analysis.failure_report import generate_failure_breakdown
from halludetect.config import load_config, resolve_device, set_seed
from halludetect.data.demo import load_demo_examples
from halludetect.data.ragtruth import load_ragtruth_split
from halludetect.data.schema import Example
from halludetect.eval.latency import measure_verifier_latency
from halludetect.eval.metrics import evaluate_predictions
from halludetect.inject.perturb import create_perturbed_dataset
from halludetect.models.hhem import HHEMModel
from halludetect.models.nli import NLIModel
from halludetect.text.chunking import chunk_text
from halludetect.text.claims import split_into_claims
from halludetect.text.retrieval import ChunkRetriever
from halludetect.verifiers.baseline import HHEMVerifier, WholeContextVerifier
from halludetect.verifiers.calibration import TemperatureScaler
from halludetect.verifiers.claim_level import ClaimLevelVerifier


def tune_threshold_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    best_tau = 0.5
    best_f1 = -1.0
    for tau in np.linspace(0.1, 0.9, 81):
        preds = scores < tau  # Hallucinated if score < tau
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return best_tau


def plot_roc_curves(eval_runs: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    for name, run in eval_runs.items():
        y_t = np.array(run["y_true"], dtype=int)
        scores = np.array(run["halluc_scores"], dtype=float)
        fpr, tpr, _ = roc_curve(y_t, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={run['metrics']['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Hallucination Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path / "roc.png", dpi=200)
    plt.close()


def plot_reliability_diagram(y_true: List[bool], probs: List[float], out_path: Path) -> None:
    plt.figure(figsize=(5, 5))
    bins = np.linspace(0, 1, 11)
    confs, accs = [], []
    y_arr = np.array(y_true, dtype=float)
    p_arr = np.array(probs, dtype=float)

    for i in range(10):
        in_bin = (p_arr >= bins[i]) & (p_arr < bins[i + 1])
        if np.sum(in_bin) > 0:
            confs.append(float(np.mean(p_arr[in_bin])))
            accs.append(float(np.mean(y_arr[in_bin])))

    plt.plot([0, 1], [0, 1], "k:", label="Perfect")
    if confs:
        plt.plot(confs, accs, "s-", color="#2a75d3", label="Claim-Level Model")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Observed Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "reliability.png", dpi=200)
    plt.close()


def plot_coverage_accuracy_curve(
    y_true: List[bool], confidences: List[float], is_correct: List[bool], out_path: Path
) -> None:
    alphas = np.linspace(0.5, 0.99, 20)
    coverages, accuracies = [], []

    conf_arr = np.array(confidences)
    corr_arr = np.array(is_correct)

    for alpha in alphas:
        covered = conf_arr >= alpha
        cov = float(np.mean(covered))
        acc = float(np.mean(corr_arr[covered])) if cov > 0 else 1.0
        coverages.append(cov)
        accuracies.append(acc)

    plt.figure(figsize=(6, 4))
    plt.plot(coverages, accuracies, "o-", color="#32a852")
    plt.xlabel("Coverage (Proportion of non-abstentions)")
    plt.ylabel("Accuracy on Accepted Claims")
    plt.title("Coverage vs. Accuracy (Abstention Trade-off)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path / "coverage_accuracy.png", dpi=200)
    plt.close()


def plot_latency_comparison(latencies: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    names = list(latencies.keys())
    p50s = [latencies[k]["p50_ms"] for k in names]
    p95s = [latencies[k]["p95_ms"] for k in names]

    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width / 2, p50s, width, label="p50 (ms)", color="#3470a3")
    plt.bar(x + width / 2, p95s, width, label="p95 (ms)", color="#d95f02")

    plt.xticks(x, names, rotation=15)
    plt.ylabel("Latency (ms)")
    plt.title("Verifier Latency Comparison (BS=1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "latency.png", dpi=200)
    plt.close()


def execute_evaluation(dataset_source: str = "demo", config_path: str = "config/default.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["models"]["device"])
    out_dir = Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    if dataset_source == "ragtruth":
        train_examples = load_ragtruth_split(cfg["paths"]["ragtruth_root"], split="train")
        test_examples = load_ragtruth_split(cfg["paths"]["ragtruth_root"], split="test")
        if not test_examples:
            print("RAGTruth data not detected in expected path. Falling back to Demo dataset.")
            test_examples = load_demo_examples(cfg["paths"]["demo_data"])
            train_examples = test_examples[:10]
    else:
        all_demo = load_demo_examples(cfg["paths"]["demo_data"])
        train_examples = all_demo[:10]
        test_examples = all_demo[10:] if len(all_demo) > 10 else all_demo

    # Generate injected / perturbed dataset for failure-mode analysis
    perturbed_records = create_perturbed_dataset(test_examples, cfg["paths"]["injected_data"])

    # 2. Instantiate Models
    nli = NLIModel(
        model_name=cfg["models"]["nli_model_name"],
        fallback_name=cfg["models"]["nli_fallback_name"],
        device=device,
    )
    hhem = HHEMModel(model_name=cfg["models"]["hhem_model_name"], device=device)
    retriever = ChunkRetriever(model_name=cfg["models"]["embedding_model_name"], device=device)

    # 3. Fit Calibration and Thresholds on Train Split
    train_pairs = [(e.context, e.answer) for e in train_examples]
    train_labels = np.array([e.label_hallucinated for e in train_examples], dtype=int)
    
    baseline_train_scores = nli.predict_proba(train_pairs)[:, 0]  # P(entail)
    tau_baseline = tune_threshold_f1(baseline_train_scores, train_labels)

    # Fit temperature scaler
    scaler = TemperatureScaler()
    train_logits = np.log(np.clip(nli.predict_proba(train_pairs), 1e-12, 1.0))
    train_nli_classes = np.where(train_labels == 1, 2, 0)  # Contradiction vs Entailment approximation
    scaler.fit(train_logits, train_nli_classes)

    # 4. Initialize Verifiers
    baseline_v = WholeContextVerifier(nli_model=nli, tau=tau_baseline)
    hhem_v = HHEMVerifier(hhem_model=hhem, tau=cfg["thresholds"]["tau_baseline"])
    claim_v = ClaimLevelVerifier(
        nli_model=nli,
        retriever=retriever,
        chunk_tokens=cfg["chunking"]["chunk_tokens"],
        stride=cfg["chunking"]["stride"],
        top_k=cfg["retrieval"]["top_k"],
        tau_support=cfg["thresholds"]["tau_support"],
        tau_contra=cfg["thresholds"]["tau_contra"],
        tau_answer=cfg["thresholds"]["tau_answer"],
    )
    claim_calibrated_v = ClaimLevelVerifier(
        nli_model=nli,
        retriever=retriever,
        chunk_tokens=cfg["chunking"]["chunk_tokens"],
        stride=cfg["chunking"]["stride"],
        top_k=cfg["retrieval"]["top_k"],
        tau_support=cfg["thresholds"]["tau_support"],
        tau_contra=cfg["thresholds"]["tau_contra"],
        tau_answer=cfg["thresholds"]["tau_answer"],
        alpha_abstain=cfg["thresholds"]["alpha_abstain"],
        scaler=scaler,
    )

    verifiers = {
        "Baseline (Whole Context)": baseline_v,
        "HHEM-2.1": hhem_v,
        "Claim-Level": claim_v,
        "Claim-Level + Calibrated": claim_calibrated_v,
    }

    eval_runs: Dict[str, Any] = {}
    eval_pairs = [(e.context, e.answer) for e in test_examples]
    y_true = [e.label_hallucinated for e in test_examples]

    # 5. Run Evaluations
    for name, v in verifiers.items():
        preds, scores = [], []
        for ctx, ans in eval_pairs:
            res = v.verify(ctx, ans)
            preds.append(res.hallucinated)
            scores.append(1.0 - res.score)  # Hallucination risk score = 1 - consistency

        metrics = evaluate_predictions(y_true, preds, scores)
        eval_runs[name] = {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": preds,
            "halluc_scores": scores,
        }

    # 6. Latency Benchmarks
    latencies = {name: measure_verifier_latency(v, eval_pairs[:10]) for name, v in verifiers.items()}

    # 7. Ablation Table
    ablation_rows = []
    # Baseline
    ablation_rows.append({"Setup": "Baseline (Whole Context)", **eval_runs["Baseline (Whole Context)"]["metrics"]})
    # + Claim split only (simulated by ClaimLevel with top_k large / single chunk)
    claim_split_only_v = ClaimLevelVerifier(
        nli_model=nli,
        retriever=retriever,
        chunk_tokens=4096,
        stride=4096,
        top_k=1,
    )
    cso_preds = [claim_split_only_v.verify(ctx, ans).hallucinated for ctx, ans in eval_pairs]
    cso_scores = [1.0 - claim_split_only_v.verify(ctx, ans).score for ctx, ans in eval_pairs]
    ablation_rows.append({"Setup": "+ Claim Split", **evaluate_predictions(y_true, cso_preds, cso_scores)})
    # + Chunk retrieval
    ablation_rows.append({"Setup": "+ Chunk Retrieval (Claim-Level)", **eval_runs["Claim-Level"]["metrics"]})
    # + Calibration & Abstain
    ablation_rows.append({"Setup": "+ Calibration & Abstain", **eval_runs["Claim-Level + Calibrated"]["metrics"]})

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(out_dir / "ablation_table.csv", index=False)

    # 8. Failure Analysis on Injected Perturbations
    pert_dicts = [
        {"context": p["context"], "answer": p["answer"], "label_hallucinated": True, "meta": p["meta"]}
        for p in perturbed_records
    ]
    tagged_perturbed = tag_examples_with_buckets(pert_dicts, tokenizer=nli.tokenizer)
    for item in tagged_perturbed:
        res = claim_v.verify(item["context"], item["answer"])
        item["pred_hallucinated"] = res.hallucinated

    failure_report = generate_failure_breakdown(tagged_perturbed, output_dir=out_dir)

    # 9. Plot Figures
    plot_roc_curves(eval_runs, out_dir)
    confidences = [1.0 - s for s in eval_runs["Claim-Level"]["halluc_scores"]]
    accuracies = [yt == yp for yt, yp in zip(y_true, eval_runs["Claim-Level"]["y_pred"])]
    plot_reliability_diagram(y_true, confidences, out_dir)
    plot_coverage_accuracy_curve(y_true, confidences, accuracies, out_dir)
    plot_latency_comparison(latencies, out_dir)

    # Save summary metrics
    summary = {
        name: {
            "metrics": r["metrics"],
            "latency": latencies[name],
        }
        for name, r in eval_runs.items()
    }
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation complete. All results and figures saved to: {out_dir.resolve()}")