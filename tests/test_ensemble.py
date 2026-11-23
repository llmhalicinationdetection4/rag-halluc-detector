import json

import numpy as np
import pandas as pd

from halludetect.config import load_config
from halludetect.data.ragtruth import load_ragtruth
from halludetect.eval.run_ensemble import CANDIDATES, FEATURE_SETS, run_ensemble
from halludetect.eval.run_phase3 import run_phase3
from halludetect.inject.injectors import Injection, write_injections
from halludetect.verifiers.claim_level import ClaimLevelVerifier
from tests.fakes import FakeEmbedder, FakeHHEM, FakeNLI, write_fake_ragtruth


def _verifier():
    return ClaimLevelVerifier(nli=FakeNLI(), embedder=FakeEmbedder(), nlp=None, chunk_size=30, stride=15)


def _prepare(tmp_path):
    data = write_fake_ragtruth(tmp_path / "rt")
    out = tmp_path / "res"
    out.mkdir()
    run_phase3(load_config(), _verifier(), max_train=None, out_dir=out, data_dir=data)
    test = load_ragtruth(data, "test")
    pairs = [(e.context, e.answer) for e in test]
    pd.DataFrame({"id": [e.id for e in test], "task_type": [e.task_type for e in test],
                  "label_hallucinated": [int(e.label_hallucinated) for e in test],
                  "nli_p_entail": FakeNLI().predict_proba(pairs)[:, 0], "hhem_score": FakeHHEM().predict_consistency(pairs)}
                 ).to_csv(out / "phase1_predictions.csv", index=False)
    return data, out, test


def test_ensemble_is_chosen_on_train_and_evaluated_per_task(tmp_path):
    data, out, _ = _prepare(tmp_path)
    result = run_ensemble(load_config(), out_dir=out, data_dir=data, hhem=FakeHHEM(), nli=FakeNLI(), injected=False, n_boot=50)
    for name in ("ensemble_by_task.csv", "ensemble_summary.json", "ensemble_model.json"):
        assert (out / name).exists()
    summary = json.loads((out / "ensemble_summary.json").read_text())
    assert summary["chosen_feature_set"] in CANDIDATES and summary["n_train"] == 40 and "train" in summary["note"]
    by_task = result["by_task"]
    assert set(by_task["feature_set"]) == set(FEATURE_SETS) and set(by_task["task"]) == {"POOLED", "QA", "Summary", "Data2txt"}
    hhem_only = by_task[(by_task["feature_set"] == "hhem") & (by_task["task"] == "POOLED")].iloc[0]
    assert abs(hhem_only["auc_minus_hhem"]) < 1e-6                         # one monotone feature cannot change the ranking
    chosen = by_task[(by_task["feature_set"] == summary["chosen_feature_set"]) & (by_task["task"] == "POOLED")].iloc[0]
    assert chosen["auc"] > 0.95 and bool(chosen["chosen"]) and chosen["f1"] > 0.9


def test_ensemble_needs_phase3_and_phase1_files(tmp_path):
    try:
        run_ensemble(load_config(), out_dir=tmp_path)
    except FileNotFoundError as err:
        assert "phase3" in str(err)
    else:
        raise AssertionError("expected FileNotFoundError")
    data, out, _ = _prepare(tmp_path / "again")
    (out / "phase1_predictions.csv").unlink()
    try:
        run_ensemble(load_config(), out_dir=out, data_dir=data, hhem=FakeHHEM(), nli=FakeNLI(), injected=False)
    except FileNotFoundError as err:
        assert "predictions" in str(err)
    else:
        raise AssertionError("expected FileNotFoundError")


def _write_injected(out, test):
    faithful = [e for e in test if not e.label_hallucinated][:12]
    injections = []
    for k, e in enumerate(faithful):
        bad = e.answer + " A secret society of dragons founded it long ago."
        injections.append(Injection(e.id, e.task_type, "unsupported_addition" if k % 2 else "number_swap", e.context, e.answer, bad,
                                    "x", "b", "a"))
    write_injections(out / "phase2_injected.jsonl", injections)
    rows = []
    for inj in injections:
        o, ed = (inj.context, inj.original), (inj.context, inj.edited)
        rows.append({"example_id": inj.example_id, "task_type": inj.task_type, "error_type": inj.error_type, "description": "x",
                     "orig_nli_whole_context": FakeNLI().predict_proba([o])[0, 0], "edit_nli_whole_context": FakeNLI().predict_proba([ed])[0, 0],
                     "orig_hhem": FakeHHEM().predict_consistency([o])[0], "edit_hhem": FakeHHEM().predict_consistency([ed])[0]})
    pd.DataFrame(rows).to_csv(out / "phase2_injected_scores.csv", index=False)
    return injections


def test_ensemble_on_injected_errors(tmp_path):
    data, out, test = _prepare(tmp_path)
    _write_injected(out, test)
    result = run_ensemble(load_config(), out_dir=out, data_dir=data, hhem=FakeHHEM(), nli=FakeNLI(), verifier=_verifier(), n_boot=50)
    inj = result["injected"]
    assert (out / "ensemble_injected.csv").exists()
    assert set(inj["error_type"]) == {"ALL_TYPES", "unsupported_addition", "number_swap"}
    assert inj[inj["error_type"] == "ALL_TYPES"].iloc[0]["n"] == 12
    assert (inj["ensemble_pair_acc"] > 0.9).all() and (inj["hhem_pair_acc"] > 0.9).all()
    assert {"claim_pair_acc", "ensemble_ci_low", "ensemble_caught"} <= set(inj.columns)


def test_injected_comparison_is_skipped_when_the_files_do_not_line_up(tmp_path):
    data, out, test = _prepare(tmp_path)
    _write_injected(out, test)
    scores = pd.read_csv(out / "phase2_injected_scores.csv").iloc[::-1]
    scores.to_csv(out / "phase2_injected_scores.csv", index=False)       # a different row order than the jsonl
    result = run_ensemble(load_config(), out_dir=out, data_dir=data, hhem=FakeHHEM(), nli=FakeNLI(), verifier=_verifier(), n_boot=20)
    assert result["injected"] is None and not (out / "ensemble_injected.csv").exists()


def test_injected_comparison_is_skipped_when_the_edits_differ_even_if_the_ids_match(tmp_path):
    data, out, test = _prepare(tmp_path)
    _write_injected(out, test)
    scores = pd.read_csv(out / "phase2_injected_scores.csv")
    scores["description"] = "a different edit"                        # e.g. another spaCy version produced other swaps
    scores.to_csv(out / "phase2_injected_scores.csv", index=False)
    result = run_ensemble(load_config(), out_dir=out, data_dir=data, hhem=FakeHHEM(), nli=FakeNLI(), verifier=_verifier(), n_boot=20)
    assert result["injected"] is None
