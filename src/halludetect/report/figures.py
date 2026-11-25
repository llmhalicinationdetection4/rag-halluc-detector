"""Figures for the report, generated from the CSV/JSON files in results/ (nothing is typed in by hand).

    python -m halludetect figures            # writes PNGs to report/figures/

A figure whose input file is missing is skipped with a message, so partial results still work.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLORS = {"hhem": "#0072B2", "nli": "#E69F00", "claim": "#009E73", "combined": "#6A3D9A", "grey": "#7f7f7f"}
TASK_ORDER = ["POOLED", "QA", "Summary", "Data2txt"]
TASK_LABEL = {"POOLED": "All tasks", "ALL": "All tasks", "QA": "QA", "Summary": "Summary", "Data2txt": "Data-to-text"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight"})
    return plt


def _read(results: Path, name: str):
    path = results / name
    if not path.exists():
        print(f"[skip] {name} not found")
        return None
    return pd.read_csv(path)


def _save(plt, fig, out: Path, name: str) -> Path:
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[ok]   {path}")
    return path


def _grouped_bars(ax, groups, series, width=0.8, ylim=None, chance=None, label_values=True):
    """series: [(label, color, values, (lo_err, hi_err) or None)] over `groups`."""
    n = len(series)
    x = np.arange(len(groups))
    w = width / n
    for i, (label, color, vals, err) in enumerate(series):
        pos = x - width / 2 + w * (i + 0.5)
        yerr = None if err is None else np.array([np.asarray(vals) - np.asarray(err[0]), np.asarray(err[1]) - np.asarray(vals)])
        ax.bar(pos, vals, w * 0.92, label=label, color=color, yerr=yerr, capsize=2, error_kw={"elinewidth": 1})
        if label_values:
            tops = np.asarray(vals) if err is None else np.asarray(err[1])
            for px, v, top in zip(pos, vals, tops):
                ax.text(px, top + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    if ylim:
        ax.set_ylim(*ylim)
    if chance is not None:
        ax.axhline(chance, color=COLORS["grey"], ls="--", lw=1)
        ax.text(len(groups) - 0.45, chance + 0.005, "chance" if chance == 0.5 else "", color=COLORS["grey"], ha="right", fontsize=8)


# ------------------------------------------------------------------------------------------------ 1. pipeline
def fig_pipeline(results: Path, out: Path):
    plt = _plt()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f2f2f2", ec="#555555", fs=9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#444444"))

    box(0.1, 2.2, 1.7, 1.2, "Retrieved context\n+ model answer", fc="#e8e8e8", bold=True)
    box(2.5, 4.35, 3.0, 0.9, "Whole-context NLI\nDeBERTa-v3, first 512 tokens", fc="#fbe9c8")
    box(2.5, 2.65, 3.0, 0.9, "HHEM-2.1-Open\nreads the full context", fc="#cfe5f3")
    cl = ["Split answer\ninto claims", "Cut context into\n256-token chunks", "Match top-3 chunks\n(MiniLM)", "NLI on each\n(chunk, claim)"]
    box(2.5, 0.2, 3.0, 1.9, "", fc="#d3eee4")
    ax.text(4.0, 1.93, "Claim-level verifier", ha="center", fontsize=9, fontweight="bold")
    for i, t in enumerate(cl):
        bx, by = 2.62 + (i % 2) * 1.43, 0.3 + (1 - i // 2) * 0.72
        box(bx, by, 1.33, 0.62, t, fc="white", fs=7.2)
    box(6.2, 4.35, 1.9, 0.9, "score = P(entailment)\nflag if below tau", fc="white", fs=8.5)
    box(6.2, 2.65, 1.9, 0.9, "score = P(consistent)\nflag if below tau", fc="white", fs=8.5)
    box(6.2, 0.55, 1.9, 1.2, "score = 1 - lowest\nclaim entailment\nflag if at or above tau", fc="white", fs=8.2)
    box(8.8, 2.1, 2.0, 1.4, "Verdict:\nsupported or\nhallucinated", fc="#e8e8e8", bold=True)
    for y in (4.8, 3.1):
        arrow(1.8, 2.8, 2.5, y)
        arrow(5.5, y, 6.2, y)
        arrow(8.1, y, 8.8, 2.8)
    arrow(1.8, 2.8, 2.5, 1.2)
    arrow(5.5, 1.15, 6.2, 1.15)
    arrow(8.1, 1.15, 8.8, 2.4)
    ax.text(5.5, 5.5, "All thresholds (tau) are tuned on the RAGTruth train split; every result is measured on the test split.",
            ha="center", fontsize=9, style="italic", color="#333333")
    return _save(plt, fig, out, "fig1_pipeline.png")


# ------------------------------------------------------------------------------------------------ 2. AUC by task
def fig_auc_by_task(results: Path, out: Path):
    p1, p3, en = _read(results, "phase1_by_task.csv"), _read(results, "phase3_by_task.csv"), _read(results, "ensemble_by_task.csv")
    if p1 is None or p3 is None:
        return None
    plt = _plt()
    p1 = p1.assign(task=p1.task.replace({"ALL": "POOLED"})).set_index(["task", "model"])
    p3 = p3.set_index("task")
    groups = [TASK_LABEL[t] for t in TASK_ORDER]

    def col(model, field):
        return [p1.loc[(t, model), field] for t in TASK_ORDER]

    series = [("Whole-context NLI", COLORS["nli"], col("nli_whole_context", "roc_auc"), (col("nli_whole_context", "auc_ci_low"), col("nli_whole_context", "auc_ci_high"))),
              ("Claim-level", COLORS["claim"], [p3.loc[t, "claim_auc"] for t in TASK_ORDER], ([p3.loc[t, "claim_auc_ci_low"] for t in TASK_ORDER], [p3.loc[t, "claim_auc_ci_high"] for t in TASK_ORDER])),
              ("HHEM", COLORS["hhem"], col("hhem", "roc_auc"), (col("hhem", "auc_ci_low"), col("hhem", "auc_ci_high")))]
    if en is not None:
        chosen = en[en.chosen].feature_set.iloc[0]
        e = en[en.feature_set == chosen].set_index("task")
        series.append(("HHEM + claim-level (combined)", COLORS["combined"], [e.loc[t, "auc"] for t in TASK_ORDER], ([e.loc[t, "auc_ci_low"] for t in TASK_ORDER], [e.loc[t, "auc_ci_high"] for t in TASK_ORDER])))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    _grouped_bars(ax, groups, series, ylim=(0.4, 1.0), chance=0.5)
    ax.set_ylabel("ROC-AUC on the test split (95% bootstrap CI)")
    ax.set_title("Detection quality by task")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8.5, frameon=False)
    return _save(plt, fig, out, "fig2_auc_by_task.png")


# ------------------------------------------------------------------------------------------------ 3. ROC curves
def fig_roc(results: Path, out: Path):
    pred, summ = _read(results, "phase1_predictions.csv"), None
    if pred is None:
        return None
    import json

    from sklearn.metrics import roc_auc_score, roc_curve

    plt = _plt()
    taus = {}
    path = results / "phase1_summary.json"
    if path.exists():
        th = json.loads(path.read_text(encoding="utf-8"))["thresholds"]
        taus = {"hhem": th["hhem"]["tuned_tau"], "nli": th["nli_whole_context"]["tuned_tau"]}
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.4), sharex=True, sharey=True)
    for ax, task in zip(axes.ravel(), TASK_ORDER):
        sub = pred if task == "POOLED" else pred[pred.task_type == task]
        y = sub.label_hallucinated.to_numpy()
        for key, col, label, color in (("hhem", "hhem_score", "HHEM", COLORS["hhem"]), ("nli", "nli_p_entail", "Whole-context NLI", COLORS["nli"])):
            s = 1 - sub[col].to_numpy()
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC {roc_auc_score(y, s):.2f})")
            if key in taus:
                flag = sub[col].to_numpy() < taus[key]
                ax.plot(flag[y == 0].mean(), flag[y == 1].mean(), "o", color=color, mec="black", ms=7)
        ax.plot([0, 1], [0, 1], ls="--", color=COLORS["grey"], lw=1)
        ax.set_title(TASK_LABEL[task])
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("False-alarm rate (faithful responses flagged)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Recall (hallucinations caught)")
    fig.suptitle("ROC curves on the test split; dots mark the tuned thresholds", y=0.995)
    fig.tight_layout()
    return _save(plt, fig, out, "fig3_roc_curves.png")


# ------------------------------------------------------------------------------------------------ 4. threshold trade-off
def fig_threshold_tradeoff(results: Path, out: Path):
    pred = _read(results, "phase1_predictions.csv")
    if pred is None:
        return None
    plt = _plt()
    y, s = pred.label_hallucinated.to_numpy(), pred.hhem_score.to_numpy()
    taus = np.linspace(0, 1, 201)
    recall = [(s[y == 1] < t).mean() for t in taus]
    fpr = [(s[y == 0] < t).mean() for t in taus]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(taus, recall, color=COLORS["hhem"], lw=2, label="Hallucinations caught (recall)")
    ax.plot(taus, fpr, color="#D55E00", lw=2, label="Faithful responses falsely flagged")
    ax.axvline(0.824, color="black", ls="--", lw=1)
    ax.text(0.83, 0.05, "tuned tau = 0.824", fontsize=8.5)
    for t in (0.824,):
        ax.plot(t, (s[y == 1] < t).mean(), "o", color=COLORS["hhem"], mec="black")
        ax.plot(t, (s[y == 0] < t).mean(), "o", color="#D55E00", mec="black")
    ax.set_xlabel("HHEM threshold tau (flag when the score is below tau)")
    ax.set_ylabel("Share of test responses")
    ax.set_title("The threshold trades missed hallucinations against false alarms")
    ax.legend(loc="upper left", frameon=False)
    return _save(plt, fig, out, "fig4_threshold_tradeoff.png")


# ------------------------------------------------------------------------------------------------ 5. ablation
def fig_ablation(results: Path, out: Path):
    ab = _read(results, "phase3_ablation.csv")
    if ab is None:
        return None
    plt = _plt()
    names = {"baseline": "Baseline (top-3, 256 tokens)", "top_k_1": "Top-1 chunk", "top_k_5": "Top-5 chunks", "chunk_128": "128-token chunks",
             "chunk_512": "512-token chunks", "retrieval_off": "No retrieval (all chunks)", "claim_split_off": "No claim splitting"}
    pooled = ab[ab.task == "POOLED"].set_index("config").loc[list(names)]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    y = np.arange(len(names))[::-1]
    colors = [COLORS["claim"] if k == "baseline" else (COLORS["nli"] if k == "claim_split_off" else COLORS["grey"]) for k in names]
    ax.barh(y, pooled.auc, color=colors, xerr=[pooled.auc - pooled.auc_ci_low, pooled.auc_ci_high - pooled.auc], capsize=3)
    for yi, v in zip(y, pooled.auc):
        ax.text(0.405, yi, f"{v:.3f}", va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(list(names.values()))
    ax.axvline(0.5, color=COLORS["grey"], ls="--", lw=1)
    ax.set_xlim(0.4, 0.85)
    ax.set_xlabel("Pooled ROC-AUC (95% bootstrap CI), 300 test responses per task")
    ax.set_title("Ablation: splitting the answer into claims matters most")
    ax.grid(axis="y", alpha=0)
    return _save(plt, fig, out, "fig5_ablation.png")


# ------------------------------------------------------------------------------------------------ 6. injected errors
def fig_injected(results: Path, out: Path):
    p2, en = _read(results, "phase2_injected_metrics.csv"), _read(results, "ensemble_injected.csv")
    if p2 is None:
        return None
    plt = _plt()
    p2 = p2[p2.task_type == "ALL"]
    types = ["number_swap", "date_change", "unsupported_addition", "negation_flip", "entity_swap", "ALL_TYPES"]
    labels = ["Number\nswap", "Date\nchange", "Unsupported\nsentence", "Negation\nflip", "Entity\nswap", "All\ntypes"]

    def pick(model, field):
        sub = p2[p2.model == model].set_index("error_type")
        return [sub.loc[t, field] for t in types]

    series = [("Whole-context NLI", COLORS["nli"], pick("nli_whole_context", "pair_acc"), (pick("nli_whole_context", "pair_acc_ci_low"), pick("nli_whole_context", "pair_acc_ci_high"))),
              ("HHEM", COLORS["hhem"], pick("hhem", "pair_acc"), (pick("hhem", "pair_acc_ci_low"), pick("hhem", "pair_acc_ci_high")))]
    if en is not None:
        e = en.set_index("error_type")
        series.insert(1, ("Claim-level", COLORS["claim"], [e.loc[t, "claim_pair_acc"] for t in types], None))
        series.append(("HHEM + claim-level (combined)", COLORS["combined"], [e.loc[t, "ensemble_pair_acc"] for t in types], ([e.loc[t, "ensemble_ci_low"] for t in types], [e.loc[t, "ensemble_ci_high"] for t in types])))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    _grouped_bars(ax, labels, series, ylim=(0.4, 1.15), chance=0.5)
    ax.set_ylabel("Pair accuracy: share of edits that lower the score")
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.set_title("Injected errors: HHEM is weakest on entity swaps and negations")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8.5, frameon=False)
    return _save(plt, fig, out, "fig6_injected_errors.png")


# ------------------------------------------------------------------------------------------------ 7. real error types
def fig_real_errors(results: Path, out: Path):
    real = _read(results, "phase2_real_error_types.csv")
    if real is None:
        return None
    plt = _plt()
    real = real[real.task_type == "ALL"]
    groups = ["hallucinated: ALL", "hallucinated: Evident Baseless Info", "hallucinated: Subtle Baseless Info", "hallucinated: Evident Conflict",
              "hallucinated: Subtle Conflict", "faithful (quality=good)", "faithful (quality=incorrect_refusal)"]
    labels = ["All\nhallucinated", "Evident\nbaseless", "Subtle\nbaseless", "Evident\nconflict", "Subtle\nconflict*", "Faithful\n(false alarms)", "Refusals\n(false alarms)"]

    def pick(model):
        sub = real[real.model == model].set_index("group")
        return [sub.loc[g, "flagged_rate"] for g in groups]

    series = [("Whole-context NLI", COLORS["nli"], pick("nli_whole_context"), None), ("HHEM", COLORS["hhem"], pick("hhem"), None)]
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    _grouped_bars(ax, labels, series, ylim=(0, 1.12))
    ax.axvline(4.5, color=COLORS["grey"], ls=":", lw=1)
    ax.set_ylabel("Share flagged as hallucinated")
    ax.set_title("RAGTruth's own error types: high is good on the left, low is good on the right")
    ax.text(4.45, 1.05, "*only 15 responses", ha="right", fontsize=8, color=COLORS["grey"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    return _save(plt, fig, out, "fig7_real_error_types.png")


# ------------------------------------------------------------------------------------------------ 8. reliability
def fig_reliability(results: Path, out: Path):
    rel, cal = _read(results, "phase4_reliability.csv"), _read(results, "phase4_calibration.csv")
    if rel is None:
        return None
    plt = _plt()
    rel = rel[rel.n > 0]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", color=COLORS["grey"], label="perfectly calibrated")
    ax.plot(rel.mean_probability, rel.observed_rate, "-", color=COLORS["claim"], lw=1.5)
    sc = ax.scatter(rel.mean_probability, rel.observed_rate, s=40 + rel.n / 4, color=COLORS["claim"], edgecolor="black", zorder=3, label="bins (size = responses)")
    if cal is not None:
        row = cal[(cal.task == "POOLED") & (cal.model == "claim_level")]
        if len(row):
            ax.text(0.04, 0.9, f"ECE = {row.ece.iloc[0]:.3f}", transform=ax.transAxes, fontsize=10)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Predicted probability of hallucination")
    ax.set_ylabel("Observed share hallucinated")
    ax.set_title("Calibration of the claim-level model (test)")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    return _save(plt, fig, out, "fig8_reliability.png")


# ------------------------------------------------------------------------------------------------ 9. latency
def fig_latency(results: Path, out: Path):
    lat = _read(results, "phase4_latency.csv")
    if lat is None:
        return None
    plt = _plt()
    names = {"nli_whole_context": "Whole-context NLI", "hhem": "HHEM", "claim_level": "Claim-level"}
    lat = lat.set_index("engine").loc[[k for k in names if k in set(lat.engine)]]
    colors = {"nli_whole_context": COLORS["nli"], "hhem": COLORS["hhem"], "claim_level": COLORS["claim"]}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(lat))
    ax.bar(x, lat.p50_ms, color=[colors[k] for k in lat.index], label="median")
    ax.scatter(x, lat.p95_ms, color="black", marker="_", s=600, zorder=3, label="95th percentile")
    for xi, m, p in zip(x, lat.p50_ms, lat.p95_ms):
        ax.text(xi, p + 5, f"{m:.0f} / {p:.0f} ms", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([names[k] for k in lat.index])
    ax.set_ylabel("Milliseconds per response")
    dev = str(lat.device.iloc[0])
    ax.set_title(f"Latency per response, one at a time ({'GPU' if dev.startswith('cuda') else dev}, {int(lat.n.iloc[0])} real test responses)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(lat.p95_ms) * 1.2)
    return _save(plt, fig, out, "fig9_latency.png")


ALL_FIGURES = [fig_pipeline, fig_auc_by_task, fig_roc, fig_threshold_tradeoff, fig_ablation, fig_injected, fig_real_errors,
               fig_reliability, fig_latency]


def make_all(results_dir="results", out_dir="report/figures") -> list[Path]:
    results, out = Path(results_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in ALL_FIGURES:
        path = fn(results, out)
        if path is not None:
            made.append(path)
    print(f"{len(made)} of {len(ALL_FIGURES)} figures written to {out}")
    return made
