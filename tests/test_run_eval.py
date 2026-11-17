import json

from halludetect.eval.run_eval import run_phase1


def test_phase1_writes_outputs_with_stubs(cfg, stub_nli, stub_hhem, tmp_path):
    out = run_phase1(cfg, dataset="demo", use_hhem=True, out_dir=tmp_path, nli=stub_nli, hhem=stub_hhem)
    assert (tmp_path / "phase1_metrics.csv").exists()
    assert (tmp_path / "phase1_predictions.csv").exists()
    summary = json.loads((tmp_path / "phase1_summary.json").read_text())
    assert summary["n_train"] == 10 and summary["n_test"] == 10 and summary["hhem_status"] == "ok"
    models = set(out["metrics"]["model"])
    assert models == {"nli_whole_context", "hhem"}
    assert out["metrics"]["f1"].between(0, 1).all()


def test_phase1_without_hhem(cfg, stub_nli, tmp_path):
    out = run_phase1(cfg, dataset="demo", use_hhem=False, out_dir=tmp_path, nli=stub_nli)
    assert set(out["metrics"]["model"]) == {"nli_whole_context"}
