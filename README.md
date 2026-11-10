# Automated Contextual Hallucination Detection in LLM RAG Architectures: A Claim-Level NLI Verification Approach with Failure-Mode Analysis

## Overview
Retrieval-Augmented Generation (RAG) grounds Large Language Model (LLM) responses in external knowledge sources to mitigate unsupported generations. However, long-context window saturation, lexical overlap distractors, and subtle entity/numerical shifts frequently induce contextual hallucinations. This repository provides an end-to-end Python pipeline to evaluate, diagnose, and mitigate hallucinated statements via fine-grained claim decomposition, local semantic chunk retrieval, and temperature-calibrated Natural Language Inference (NLI).

## Problem
Standard whole-context verifiers concatenate the complete retrieved passage with the full generation. Under long contexts or multi-sentence generations, NLI cross-encoders fail to isolate single ungrounded claims, dilute contradiction signals, and suffer from length-induced truncation bias.

## Method (Phases)
- **Phase 0 (Setup & Ingestion)**: Normalized schemas across RAGTruth (QA, Data2txt, Summary) and fallback demo datasets.
- **Phase 1 (Baseline Verifier)**: Full premise-hypothesis cross-encoder evaluation with threshold $\tau$ tuned on the training split. Wrapper around Vectara HHEM-2.1.
- **Phase 2 (Failure Analysis & Stress Testing)**: Deterministic error injections (numerical swaps, date shifts, named entity substitutions via spaCy, negations, distractor insertions, and antonym contradictions) categorized across length, ROUGE-L paraphrase terciles, and lexical buckets.
- **Phase 3 (Claim-Level Fix)**: Sentential claim decomposition, overlapping context windowing ($256$ tokens, stride $128$), dense cosine similarity retrieval of the top-$k$ relevant chunks via MiniLM embeddings, and localized NLI verification.
- **Phase 4 (Calibration & Abstention)**: Temperature scaling on training logits to minimize Expected Calibration Error (ECE) and confidence-based abstention when maximum class probabilities fall below $\alpha$.

## Setup
```bash
git clone <repo-url>
cd rag-halluc-detector
pip install -r requirements.txt
python -m spacy download en_core_web_sm