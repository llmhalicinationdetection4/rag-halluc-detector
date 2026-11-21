import json

import numpy as np
import pandas as pd

from halludetect.config import load_config
from halludetect.data.ragtruth import load_ragtruth
from halludetect.eval.run_ablation import CONFIGS, run_ablation
from halludetect.eval.run_phase3 import run_phase3
from halludetect.eval.run_phase4 import measure_latency, run_phase4
from halludetect.verifiers.claim_level import AGGREGATIONS, ClaimLevelVerifier
from tests.fakes import FakeEmbedder, FakeNLI, write_fake_ragtruth


def _verifier():
    return ClaimLevelVerifier(nli=FakeNLI(), embedder=FakeEmbedder(), nlp=None, chunk_size=30, stride=15)


def _phase1_csv(path, data_dir):
    test = load_ragtruth(data_dir, "test")
    rng = np.random.default_rng(0)
    pd.DataFrame({"id": [e.id for e in test], "task_type": [e.task_type for e in test],
                  "label_hallucinated": [int(e.label_hallucinated) for e in test],
                  "nli_p_entail": rng.random(len(test)), "hhem_score": rng.random(len(test))}).to_csv(path, index=False)


def test_phase3_end_to_end_with_caching(tmp_path):
    data = write_fake_ragtruth(tmp_path / "rt")
    out = tmp_path / "res"
    out.mkdir()
    _phase1_csv(out / "phase1_predictions.csv", data)
    verifier, calls = _verifier(), []
    original = verifier.verify
    verifier.verify = lambda *a, **k: (calls.append(1), original(*a, **k))[1]

    result = run_phase3(load_config(), verifier, max_train=None, max_test=None, out_dir=out, data_dir=data)
    n_first = len(calls)
    assert n_first == 40 + 24
    for name in ("phase3_by_task.csv", "phase3_claim_level.csv", "phase3_summary.json", "phase3_features_train.csv", "phase3_features_test.csv"):
        assert (out / name).exists()
    summary = json.loads((out / "phase3_summary.json").read_text())
    assert summary["best_aggregation"] in AGGREGATIONS and summary["claim_threshold"] is not None
    by_task = result["by_task"].set_index("task")
    assert set(by_task.index) == {"POOLED", "QA", "Summary", "Data2txt"}
    assert by_task.loc["POOLED", "claim_auc"] > 0.95 and by_task.loc["POOLED", "f1"] > 0.9
    assert {"hhem_auc", "auc_diff_vs_hhem", "auc_diff_vs_nli_ci_low"} <= set(by_task.columns)
    claim = result["claim_level"].set_index("task")
    assert claim.loc["POOLED", "f1"] > 0.8                                  # claim labels line up with the annotated spans

    run_phase3(load_config(), verifier, max_train=None, max_test=None, out_dir=out, data_dir=data)
    assert len(calls) == n_first                                            # the second run used the cache only


def test_phase3_skips_the_baseline_comparison_when_ids_do_not_match(tmp_path):
    data = write_fake_ragtruth(tmp_path / "rt")
    out = tmp_path / "res"
    out.mkdir()
    pd.DataFrame({"id": ["nope"], "task_type": ["QA"], "label_hallucinated": [0], "nli_p_entail": [0.5],
                  "hhem_score": [0.5]}).to_csv(out / "phase1_predictions.csv", index=False)
    result = run_phase3(load_config(), _verifier(), max_train=None, out_dir=out, data_dir=data)
    assert "hhem_auc" not in result["by_task"].columns


def test_aggregation_and_threshold_come_from_train_not_test(tmp_path):
    data = write_fake_ragtruth(tmp_path / "rt")
    out = tmp_path / "res"
    out.mkdir()
    run_phase3(load_config(), _verifier(), max_train=None, out_dir=out, data_dir=data)
    summary = json.loads((out / "phase3_summary.json").read_text())
    assert summary["n_train"] == 40 and "train" in summary["note"]
    train = pd.read_csv(out / "phase3_features_train.csv")
    col = "h_" + summary["best_aggregation"]
    flagged_f1 = ((train[col] >= summary["best_threshold"]) & (train["y"] == 1)).sum()
    assert flagged_f1 > 0


def test_ablation_covers_every_variant(tmp_path):
    data = write_fake_ragtruth(tmp_path / "rt")
    out = tmp_path / "res"
    out.mkdir()
    verifier = _verifier()
    run_phase3(load_config(), verifier, max_train=None, out_dir=out, data_dir=data)
    df = run_ablation(load_config(), verifier, per_task=4, out_dir=out, data_dir=data)
    assert set(df["config"]) == {name for name, _ in CONFIGS}
    assert (out / "phase3_ablation.csv").exists() and df["auc"].between(0, 1).all()
    assert (df[df["task"] == "POOLED"]["n"] == 12).all()                    # 4 per task x 3 tasks


def test_ablation_needs_phase3_first(tmp_path):
    try:
        run_ablation(load_config(), _verifier(), out_dir=tmp_path, data_dir=tmp_path)
    except FileNotFoundError as err:
        assert "phase3" in str(err)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_phase4_calibration_abstention_and_latency(tmp_path):
    data = write_fake_ragtruth(tmp_path / "rt")
    out = tmp_path / "res"
    out.mkdir()
    run_phase3(load_config(), _verifier(), max_train=None, out_dir=out, data_dir=data)
    examples = load_ragtruth(data, "test")
    engines = {"fake_a": lambda c, a: None, "fake_b": lambda c, a: sum(range(1000))}
    result = run_phase4(load_config(), out_dir=out, latency_n=5, engines=engines, latency_examples=examples, data_dir=data)
    for name in ("phase4_calibrator.json", "phase4_calibration.csv", "phase4_abstention_curve.csv", "phase4_reliability.csv",
                 "phase4_latency.csv", "phase4_summary.json"):
        assert (out / name).exists()
    cal = result["calibration"]
    pooled = cal[(cal["model"] == "claim_level") & (cal["task"] == "POOLED")].iloc[0]
    assert 0 <= pooled["ece"] <= 1 and pooled["auc"] > 0.9
    assert set(result["latency"]["engine"]) == {"fake_a", "fake_b"} and (result["latency"]["n"] == 5).all()
    assert "HHEM calibration skipped" in result["summary"]["hhem_note"]     # no Phase 1 train scores in this folder


def test_phase4_needs_phase3_first(tmp_path):
    try:
        run_phase4(load_config(), out_dir=tmp_path, latency_n=0)
    except FileNotFoundError as err:
        assert "phase3" in str(err)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_latency_is_measured_per_engine_after_warmup():
    class Ex:
        context, answer = "c", "a"

    calls = []
    df = measure_latency({"x": lambda c, a: calls.append(1)}, [Ex()] * 10, n=4, warmup=2)
    assert len(calls) == 2 + 4 and df.iloc[0]["n"] == 4 and df.iloc[0]["p95_ms"] >= df.iloc[0]["p50_ms"]
