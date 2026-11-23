# Automated Contextual Hallucination Detection in LLM RAG Architectures

Claim-level NLI verification with failure-mode analysis. Given a retrieved **context** and an LLM **answer**,
the tool scores factual consistency and flags answers that are unsupported or contradicted by the context.

## Status
| Phase | What | State |
|---|---|---|
| 0 | Config, data loaders (demo + RAGTruth), schema | done |
| 1 | Whole-context NLI baseline, HHEM wrapper, full RAGTruth evaluation | done (results below) |
| 2 | Injected-error test set and failure analysis (`inject`, `analyze`) | done (results below) |
| 3 | Claim-level verifier, tuned on train, evaluated per task, ablations (`phase3`, `ablate`) | done (results below) |
| 4 | Calibration (ECE), abstention, latency (`phase4`) | done (results below) |
| 5 | Dashboard: FastAPI (`app.py`) + `index.html` | done; HHEM gives the verdict, the claim-level verifier is shown as experimental |

Every number in the Results section is generated from the CSV/JSON files in `results/`; none is a placeholder.

## Setup
Heavy runs are meant for Kaggle (`notebooks/run_on_kaggle.ipynb`). Python 3.12 or 3.13.
```
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install numpy pandas scikit-learn pyyaml tqdm typer pytest   # enough for most tests
pip install -e .
pytest
```
Dashboard (needs the verification models, about 1.5 GB downloaded on the first scan):
```
pip install -r requirements-app.txt
python -m halludetect serve            # opens http://127.0.0.1:8000
```
The demo scenarios in the sidebar are 4 safe cases, 2 borderline cases (a paraphrase that drops 'usually', and a statement that is true but not in the context) and 3 risk cases.
The borderline cases show the false-alarm side of the threshold and are kept on purpose. Thresholds are never tuned on them. Untick the sidebar box to skip the slower experimental claim-level check.
`python -m spacy download en_core_web_sm` is needed for `inject` and improves sentence splitting; without it the claim-level
verifier falls back to a regex splitter.

## Pipeline (in this order, on a GPU)
```
python -m halludetect eval --dataset ragtruth   # Phase 1: baselines, thresholds tuned on train, ~40 min
python -m halludetect inject                    # Phase 2: injected-error test set (needs spaCy)
python -m halludetect analyze                   # Phase 2: scores it with both verifiers, error-type breakdown
python -m halludetect phase3                    # Phase 3: claim-level verifier (train subset + full test), resumable
python -m halludetect ablate                    # Phase 3: ablations on a fixed test subset
python -m halludetect phase4                    # Phase 4: calibration, abstention, latency
python -m halludetect serve                     # Phase 5: dashboard
```
Other commands: `verify` (one pair, JSON), `eval --dataset demo` (20 handwritten examples, no downloads), `inspect-ragtruth`.
Each step writes its tables to `results/` (CSV/JSON are tracked in git, `results/cache` is not) and caches model scores, so an
interrupted run resumes. Protocol everywhere: thresholds, aggregation and calibration are fitted on the train split; only the
test split is reported.

## Data
RAGTruth: `git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git data/RAGTruth`, then check the folder that holds
`response.jsonl` and `source_info.jsonl` and set `data.ragtruth_dir` in `config/default.yaml`.
Protocol: tune thresholds on the train split, report only on the test split.

## Tests
`pytest` runs on a 4 GB laptop: all tests use a stub NLI model and download nothing.
`pytest -m slow` loads the real model and is run on Kaggle/Colab.

## Results
Both verifiers are pre-trained and used as they are; the claim-level verifier reuses the same NLI model. Thresholds, the aggregation and the
calibrator were fitted on the **train** split only; every number below is from the **test** split. One run, seed 42. Source files are in `results/`.

### Phase 1: baselines (RAGTruth test, 2,700 responses, 900 per task)
HHEM tau 0.824 and NLI tau 0.990 were tuned on the train split (`results/phase1_summary.json`).

| Task | Model | n | Hallucinated rate | F1 at tuned tau | F1 if everything is flagged | ROC-AUC (95% bootstrap CI) |
|---|---|---|---|---|---|---|
| All tasks (pooled) | HHEM | 2700 | 0.349 | 0.702 | 0.518 | 0.835 (0.820 to 0.851) |
| All tasks (pooled) | NLI whole-context | 2700 | 0.349 | 0.531 | 0.518 | 0.591 (0.569 to 0.611) |
| QA | HHEM | 900 | 0.178 | 0.567 | 0.302 | 0.851 (0.815 to 0.886) |
| QA | NLI whole-context | 900 | 0.178 | 0.320 | 0.302 | 0.574 (0.528 to 0.618) |
| Summary | HHEM | 900 | 0.227 | 0.541 | 0.370 | 0.784 (0.750 to 0.820) |
| Summary | NLI whole-context | 900 | 0.227 | 0.400 | 0.370 | 0.634 (0.592 to 0.673) |
| Data-to-text | HHEM | 900 | 0.643 | 0.791 | 0.783 | 0.631 (0.590 to 0.668) |
| Data-to-text | NLI whole-context | 900 | 0.643 | 0.759 | 0.783 | 0.504 (0.461 to 0.543) |

Where F1 is close to the "everything flagged" column (data-to-text), the detector is mostly flagging everything, not detecting.

### Phase 3: claim-level verifier against the baselines
The answer is split into sentences; each is matched to its 3 most similar 256-token context chunks and scored by the same NLI model.
Chosen on 3000 train responses: aggregation `min_entail` (lowest claim entailment; train AUC by aggregation: max contradiction
0.5932, min entailment 0.6826, mean entailment 0.6642,
share unsupported 0.6544), threshold 0.664.

| Task | Claim-level ROC-AUC | HHEM AUC | Claim-level minus HHEM (paired 95% CI) | NLI AUC | Claim-level minus NLI (paired 95% CI) | Claim-level F1 | F1 if everything is flagged |
|---|---|---|---|---|---|---|---|
| All tasks (pooled) | 0.734 (0.714 to 0.752) | 0.835 | -0.101 (-0.122 to -0.081) | 0.591 | 0.143 (0.118 to 0.167) | 0.608 | 0.518 |
| QA | 0.648 (0.602 to 0.692) | 0.851 | -0.203 (-0.246 to -0.161) | 0.574 | 0.075 (0.031 to 0.119) | 0.361 | 0.302 |
| Summary | 0.721 (0.679 to 0.760) | 0.784 | -0.063 (-0.112 to -0.018) | 0.634 | 0.086 (0.043 to 0.127) | 0.481 | 0.370 |
| Data-to-text | 0.616 (0.578 to 0.653) | 0.631 | -0.015 (-0.065 to 0.038) | 0.504 | 0.112 (0.061 to 0.165) | 0.782 | 0.783 |

Claim-level labels come from RAGTruth's span annotations (a claim is hallucinated if it overlaps an annotated span); claim threshold 0.645, tuned on train.

| Task | Claims | Hallucinated share | Precision | Recall | F1 | F1 if every claim is flagged |
|---|---|---|---|---|---|---|
| All tasks (pooled) | 17840 | 0.087 | 0.198 | 0.631 | 0.301 | 0.160 |
| QA | 5266 | 0.069 | 0.143 | 0.641 | 0.234 | 0.129 |
| Summary | 4722 | 0.055 | 0.180 | 0.488 | 0.263 | 0.104 |
| Data-to-text | 7852 | 0.119 | 0.236 | 0.667 | 0.348 | 0.212 |

**Ablations** (fixed test subset of 300 responses per task, aggregation as above, no thresholds involved):

| Variant | Pooled ROC-AUC (95% CI, n=900) | QA | Summary | Data-to-text |
|---|---|---|---|---|
| baseline | 0.729 (0.696 to 0.763) | 0.657 | 0.707 | 0.589 |
| top_k_1 | 0.693 (0.655 to 0.729) | 0.640 | 0.663 | 0.498 |
| top_k_5 | 0.720 (0.689 to 0.753) | 0.657 | 0.708 | 0.587 |
| chunk_128 | 0.738 (0.706 to 0.773) | 0.711 | 0.714 | 0.524 |
| chunk_512 | 0.702 (0.672 to 0.740) | 0.647 | 0.686 | 0.549 |
| retrieval_off | 0.720 (0.688 to 0.753) | 0.657 | 0.711 | 0.587 |
| claim_split_off | 0.586 (0.550 to 0.623) | 0.604 | 0.614 | 0.537 |

### Phase 4: calibration, abstention, latency
A logistic regression on the four claim-level scores, fitted on the train subset (no class weights), evaluated on test:

| Task | n | ROC-AUC | Brier score | ECE |
|---|---|---|---|---|
| All tasks (pooled) | 2700 | 0.725 | 0.201 | 0.065 |
| QA | 900 | 0.620 | 0.200 | 0.213 |
| Summary | 900 | 0.711 | 0.168 | 0.093 |
| Data-to-text | 900 | 0.639 | 0.234 | 0.111 |

An abstention margin was **not** chosen: no margin reached the target of 80% accuracy on cross-validated train predictions while answering at least 30% of
responses (`results/phase4_summary.json`), so the dashboard does not abstain. `results/phase4_abstention_curve.csv` shows the test-split trade-off for reference only.
The calibrator was never fitted for HHEM (its Phase 1 train scores were not cached).

Latency, one response at a time on 100 real test responses (Kaggle T4 GPU; CPU not measured):

| Engine | Device | Responses timed | Median ms | 95th percentile ms | Mean ms |
|---|---|---|---|---|---|
| hhem | cuda | 100 | 68 | 218 | 86 |
| nli_whole_context | cuda | 100 | 24 | 30 | 28 |
| claim_level | cuda | 100 | 137 | 277 | 154 |

### Phase 2: injected errors and RAGTruth's own error types
One controlled error is inserted into a faithful response (same context). *Pair accuracy* is the share of pairs where the edit lowers the score (0.5 = no signal);
*caught* is the share of originals that passed at the Phase 1 threshold whose edited version was then flagged. 2612 pairs in total.

| Injected error | Pairs | HHEM pair accuracy (95% CI) | HHEM caught | NLI pair accuracy (95% CI) | NLI caught |
|---|---|---|---|---|---|
| All error types | 2612 | 0.865 (0.853 to 0.878) | 0.505 | 0.841 (0.828 to 0.855) | 0.753 |
| Number swap | 600 | 0.980 (0.968 to 0.990) | 0.672 | 0.859 (0.831 to 0.886) | 0.750 |
| Entity swap | 586 | 0.657 (0.619 to 0.695) | 0.045 | 0.826 (0.797 to 0.855) | 0.566 |
| Negation flip | 600 | 0.783 (0.750 to 0.817) | 0.319 | 0.802 (0.772 to 0.832) | 0.792 |
| Date change | 226 | 0.996 (0.982 to 1.000) | 0.801 | 0.898 (0.858 to 0.938) | 0.836 |
| Unsupported added sentence | 600 | 0.987 (0.977 to 0.995) | 0.817 | 0.855 (0.827 to 0.882) | 0.825 |

Date changes are rare in two tasks (only 19 QA and 7 data-to-text pairs), so look at the pooled row, not per-task dates.
Six of the 586 entity swaps differ only by case or contain each other; leaving them out changes HHEM's pair accuracy from 0.657 to 0.655.

RAGTruth's own annotations (test split; the "subtle conflict" group has only 15 responses):

| RAGTruth group | n | Measure | HHEM | NLI whole-context |
|---|---|---|---|---|
| all hallucinated responses | 943 | recall | 0.845 | 0.918 |
| evident baseless info | 542 | recall | 0.895 | 0.924 |
| subtle baseless info | 141 | recall | 0.901 | 0.936 |
| evident conflict | 459 | recall | 0.828 | 0.900 |
| subtle conflict | 15 | recall | 0.667 | 0.800 |
| faithful, good quality | 1732 | false-positive rate | 0.307 | 0.824 |
| faithful, incorrect refusal | 24 | false-positive rate | 0.000 | 0.958 |

### What the results say
- **HHEM is the best detector here** (pooled ROC-AUC 0.835). The claim-level verifier does **not** beat it: it is significantly worse on QA and summaries and statistically tied on data-to-text.
- **The claim-level verifier clearly beats whole-context NLI** in every task (pooled +0.143 AUC, paired 95% CI 0.118 to 0.167).
- **Splitting the answer into claims is what matters:** without it the pooled AUC drops from 0.729 to 0.586 (non-overlapping intervals). Using the top 3 or 5 chunks or all chunks gives about the
  same AUC, so retrieval saves work without costing accuracy; one chunk per claim is lower (0.693) but the intervals overlap. Differences between chunk sizes are within noise.
- **Contradiction scores are the weakest signal** (train AUC 0.5932): most RAGTruth hallucinations are unsupported additions, not contradictions.
- **HHEM has a clear blind spot: entity swaps.** A name replaced by another name from the same context is caught only 4.5% of the time, while numbers, dates and invented sentences are caught 67% to 82%.
  The NLI model ranks entity swaps much better (pair accuracy 0.826), so the two models are complementary.
- **Whole-context NLI is not usable at its tuned threshold:** it flags 82% of faithful responses and 96% of refusals; HHEM flags 31% of faithful responses and none of the refusals.
- **Data-to-text stays near chance for every detector** (AUC 0.50 to 0.63).

## Limitations
- The NLI models read at most 512 tokens; HHEM reads the full context. Long contexts are therefore cut for the NLI baseline.
- The claim-level thresholds and calibrator were fitted on a seeded 3,000-response train subset whose hallucination rate (45.2%) is higher than the test split's (34.9%);
  the calibrator therefore over-estimates probabilities, most visibly in QA (ECE 0.21).
- Pooled numbers mix tasks with different hallucination rates; use the per-task rows.
- Responses marked `incorrect_refusal` or `truncated` are included in Phases 1 and 3 (24 refusals and 1 truncated response in the faithful test group).
- Injected errors are synthetic and rule-based: entity swaps use spaCy's small model, appended sentences come from a fixed pool of generic claims.
- One run with one seed; confidence intervals are bootstrap intervals over responses, not over training runs.
- CPU latency was not measured. The dashboard's demo scenarios are illustrations, not evidence, and thresholds are never tuned on them.

## Future scope
Knowledge-graph support for multi-hop claims, domain fine-tuning, multilingual verification,
live RAG gating, drift monitoring.
