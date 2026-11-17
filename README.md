# Automated Contextual Hallucination Detection in LLM RAG Architectures

Claim-level NLI verification with failure-mode analysis. Given a retrieved **context** and an LLM **answer**,
the tool scores factual consistency and flags answers that are unsupported or contradicted by the context.

## Status
| Phase | What | State |
|---|---|---|
| 0 | Config, data loaders (demo + RAGTruth), schema | done |
| 1 | Whole-context NLI baseline, HHEM wrapper, full RAGTruth evaluation | done (results below) |
| 2 | Injected-error test set and failure analysis (`inject`, `analyze`) | code built and unit-tested; run on RAGTruth pending |
| 3 | Claim-level verifier | not built |
| 4 | Calibration, abstention, final evaluation, latency | not built |
| 5 | Streamlit dashboard (`app.py`) | built on the Phase 1 backend; claim-level view arrives with Phase 3 |

## Setup
Heavy evaluation runs are meant for Kaggle (see `notebooks/run_on_kaggle.ipynb`). Python 3.12 or 3.13.
```
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install numpy pandas scikit-learn pyyaml tqdm typer pytest   # enough to run the tests
pip install -e .
pytest
```
Dashboard (also needs the two verification models, about 1.2 GB downloaded on the first scan):
```
pip install -r requirements-app.txt
python -m halludetect serve            # opens http://localhost:8501
```
`requirements.txt` holds the exact pins used for the Kaggle runs; `python -m spacy download en_core_web_sm` is only needed for `inject`.

## Usage
```
python -m halludetect verify --context "The meeting is on Monday at 10 am." --answer "The meeting is on Tuesday."
python -m halludetect eval --dataset demo            # 20 handwritten examples, no downloads
python -m halludetect inspect-ragtruth               # check the RAGTruth keys
python -m halludetect eval --dataset ragtruth        # full run (Kaggle GPU: about 40 minutes), cached in results/cache
python -m halludetect inject                         # Phase 2: build the injected-error test set (needs spaCy)
python -m halludetect analyze                        # Phase 2: score it and break down real errors by type
```
Outputs go to `results/`. Models in `config/default.yaml` can be Hugging Face ids or local folders.

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
