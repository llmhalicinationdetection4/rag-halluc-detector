from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from halludetect.report.figures import ALL_FIGURES, make_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_every_figure_is_made_from_the_repo_results(tmp_path):
    results = ROOT / "results"
    if not (results / "phase3_by_task.csv").exists():
        return  # results not present in this checkout
    made = make_all(results, tmp_path)
    assert len(made) == len(ALL_FIGURES)
    assert all(p.suffix == ".png" and p.stat().st_size > 5000 for p in made)


def test_missing_inputs_are_skipped_instead_of_crashing(tmp_path):
    made = make_all(tmp_path / "no_results_here", tmp_path / "out")
    assert {p.name for p in made} == {"fig1_pipeline.png"}      # the pipeline diagram needs no data
