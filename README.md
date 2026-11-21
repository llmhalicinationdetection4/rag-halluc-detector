# Automated Contextual Hallucination Detection in LLM RAG Architectures

Claim-level NLI verification with failure-mode analysis. Given a retrieved **context** and an LLM **answer**,
the tool scores factual consistency and flags answers that are unsupported or contradicted by the context.

## Status
| Phase | What | State |
|---|---|---|
| 0 | Config, data loaders (demo + RAGTruth), schema | done |
| 1 | Whole-context NLI baseline, HHEM wrapper, full RAGTruth evaluation | done (results below) |
| 2 | Injected-error test set and failure analysis (`inject`, `analyze`) | code done; run on RAGTruth pending |
| 3 | Claim-level verifier, tuned on train, evaluated per task, ablations (`phase3`, `ablate`) | code done; run on RAGTruth pending |
| 4 | Calibration (ECE), abstention, latency (`phase4`) | code done; run on RAGTruth pending |
| 5 | Dashboard: FastAPI (`app.py`) + `index.html` | done; HHEM gives the verdict, the claim-level verifier is shown as experimental |

Results for Phases 2 to 4 are added below only after they have been run; nothing in this README is a placeholder number.

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

## Results (Phase 1, RAGTruth test split)
Both verifiers are pre-trained and used as they are. Thresholds were tuned on the train split (tau 0.990 for NLI, 0.824 for HHEM, see
`results/phase1_summary.json`); every number below is from the held-out test split. Source: `results/phase1_by_task.csv`.

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

Read the per-task rows, not the pooled one: mixing tasks with different hallucination rates flatters the pooled numbers. Where F1 is close to
the "F1 if everything is flagged" column (data-to-text), the detector is mostly flagging everything, not detecting.

## Limitations (so far)
- The NLI baseline reads at most 512 tokens, so long contexts are truncated; HHEM reads the full context.
- Data-to-text contexts are structured JSON, and both verifiers are weak there.
- Responses marked `incorrect_refusal` or `truncated` in RAGTruth are not filtered out of the Phase 1 numbers.
- Results are from one run with one seed.
- The demo scenarios in the dashboard are illustrations, not evidence, and thresholds are never tuned on them.

## Future scope
Knowledge-graph support for multi-hop claims, domain fine-tuning, multilingual verification,
live RAG gating, drift monitoring.
