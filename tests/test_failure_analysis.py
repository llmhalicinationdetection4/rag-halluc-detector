import json

import numpy as np
import pandas as pd

from halludetect.analysis.failure import buckets_for, paired_metrics, real_error_breakdown, run_failure_analysis
from halludetect.data.ragtruth import load_ragtruth
from halludetect.data.schema import Example
from halludetect.inject.injectors import Injection, write_injections
from tests.stubs import StubHHEM, StubNLIModel

CTX = "The cafe opened in 2015 and has 40 seats."


def _injections():
    return [
        Injection("a", "QA", "number_swap", CTX, "The cafe has 40 seats.", "The cafe has 60 seats.", "40 -> 60", "b", "a"),
        Injection("b", "QA", "unsupported_addition", CTX, "The cafe has 40 seats.",
                  "The cafe has 40 seats. Experts agree.", "Experts agree.", "b", "a"),
        Injection("c", "Summary", "number_swap", CTX, "The cafe opened in 2015.", "The cafe opened in 2015 with 90 seats.",
                  "x", "b", "a"),
    ]


def test_paired_metrics_values():
    df = pd.DataFrame({
        "task_type": ["QA", "QA", "QA", "Summary"],
        "error_type": ["number_swap"] * 4,
        "orig_hhem": [0.9, 0.9, 0.5, 0.9],
        "edit_hhem": [0.1, 0.9, 0.2, 0.95],
    })
    m = paired_metrics(df, {"hhem": 0.5}).set_index(["task_type", "error_type"])
    qa = m.loc[("QA", "number_swap")]
    assert qa["n"] == 3
    assert abs(qa["pair_acc"] - (1 + 0.5 + 1) / 3) < 1e-3          # one tie counts half
    assert abs(qa["flagged_orig"] - 0) < 1e-9 and abs(qa["flagged_edit"] - 2 / 3) < 1e-3
    assert qa["n_orig_pass"] == 3          # an original score of exactly tau is not below tau, so it passes
    assert m.loc[("Summary", "number_swap")]["pair_acc"] == 0.0
    assert ("ALL", "ALL_TYPES") in m.index


def test_buckets():
    assert buckets_for(True, {"label_types": ["Evident Conflict", "Evident Conflict", "Subtle Baseless Info"]}) == [
        "hallucinated: ALL", "hallucinated: Evident Conflict", "hallucinated: Subtle Baseless Info"]
    assert buckets_for(True, {}) == ["hallucinated: ALL", "hallucinated: unknown type"]
    assert buckets_for(False, {"quality": "incorrect_refusal"}) == ["faithful: ALL", "faithful (quality=incorrect_refusal)"]


def test_real_error_breakdown_recall_and_false_positives():
    exs = {
        "1": Example("1", "c", "a", True, "QA", {"label_types": ["Evident Conflict"]}),
        "2": Example("2", "c", "a", True, "QA", {"label_types": ["Subtle Baseless Info"]}),
        "3": Example("3", "c", "a", False, "QA", {"quality": "good"}),
        "4": Example("4", "c", "a", False, "QA", {"quality": "incorrect_refusal"}),
    }
    preds = pd.DataFrame({"id": [1, 2, 3, 4], "task_type": ["QA"] * 4, "label_hallucinated": [1, 1, 0, 0],
                          "nli_p_entail": [0.1, 0.9, 0.9, 0.1], "hhem_score": [0.1, 0.2, 0.9, 0.1]})
    out = real_error_breakdown(preds, exs, {"nli_whole_context": 0.5, "hhem": 0.5})
    pick = lambda model, group: out[(out.model == model) & (out.task_type == "ALL") & (out.group == group)].iloc[0]
    assert pick("hhem", "hallucinated: ALL")["flagged_rate"] == 1.0 and pick("hhem", "hallucinated: ALL")["kind"] == "recall"
    assert pick("nli_whole_context", "hallucinated: Subtle Baseless Info")["flagged_rate"] == 0.0
    assert pick("hhem", "faithful (quality=incorrect_refusal)")["flagged_rate"] == 1.0
    assert pick("hhem", "faithful (quality=good)")["kind"] == "false_positive_rate"


def test_loader_keeps_quality_and_label_types(tmp_path):
    (tmp_path / "source_info.jsonl").write_text(json.dumps(
        {"source_id": "s1", "task_type": "QA", "source_info": {"question": "Q", "passages": "P"}}) + "\n")
    (tmp_path / "response.jsonl").write_text(json.dumps(
        {"id": "r1", "source_id": "s1", "model": "m", "split": "test", "quality": "incorrect_refusal", "response": "No.",
         "labels": [{"start": 0, "end": 2, "label_type": "Evident Conflict"}]}) + "\n")
    ex = load_ragtruth(tmp_path, "test")[0]
    assert ex.meta["quality"] == "incorrect_refusal" and ex.meta["label_types"] == ["Evident Conflict"]


def test_end_to_end_with_stub_models(cfg, tmp_path):
    write_injections(tmp_path / "phase2_injected.jsonl", _injections())
    (tmp_path / "phase1_summary.json").write_text(json.dumps({"thresholds": {
        "nli_whole_context": {"tuned_tau": 0.5}, "hhem": {"tuned_tau": 0.5}}}))
    pd.DataFrame({"id": [1, 2], "task_type": ["QA", "QA"], "label_hallucinated": [1, 0],
                  "nli_p_entail": [0.1, 0.9], "hhem_score": [0.2, 0.9]}).to_csv(tmp_path / "phase1_predictions.csv", index=False)
    examples = [Example("1", "c", "a", True, "QA", {"label_types": ["Evident Conflict"]}),
                Example("2", "c", "a", False, "QA", {"quality": "good"})]
    out = run_failure_analysis(cfg, out_dir=tmp_path, nli=StubNLIModel(), hhem=StubHHEM(), examples=examples)
    for name in ("phase2_injected_scores.csv", "phase2_injected_metrics.csv", "phase2_real_error_types.csv"):
        assert (tmp_path / name).exists()
    m = out["metrics"]
    row = m[(m.model == "hhem") & (m.task_type == "ALL") & (m.error_type == "number_swap")].iloc[0]
    assert row["n"] == 2 and row["pair_acc"] == 1.0
    assert not (tmp_path / "cache").exists()      # stand-in models never touch the cache
    assert set(out["scores"].columns) >= {"orig_hhem", "edit_hhem", "orig_nli_whole_context", "edit_nli_whole_context"}
