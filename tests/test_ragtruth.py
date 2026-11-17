import json

import pytest

from halludetect.data.ragtruth import inspect_ragtruth, load_ragtruth


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def rt_dir(tmp_path):
    _write(tmp_path / "source_info.jsonl", [
        {"source_id": "s1", "task_type": "QA", "source_info": {"question": "Q?", "passages": "Passage one."}},
        {"source_id": "s2", "task_type": "Data2txt", "source_info": {"name": "Cafe", "rating": 4}},
        {"source_id": "s3", "task_type": "Summary", "source_info": "A news article."},
    ])
    _write(tmp_path / "response.jsonl", [
        {"id": "r1", "source_id": "s1", "model": "m", "labels": [], "split": "train", "response": "Fine."},
        {"id": "r2", "source_id": "s2", "model": "m", "labels": [{"start": 0}], "split": "train", "response": "Bad."},
        {"id": "r3", "source_id": "s3", "model": "m", "labels": [], "split": "test", "response": "Ok."},
    ])
    return tmp_path


def test_labels_map_to_hallucinated(rt_dir):
    train = load_ragtruth(rt_dir, "train")
    assert {e.id: e.label_hallucinated for e in train} == {"r1": False, "r2": True}


def test_split_filter_and_contexts(rt_dir):
    test = load_ragtruth(rt_dir, "test")
    assert len(test) == 1 and test[0].context == "A news article."
    qa = [e for e in load_ragtruth(rt_dir, "train") if e.task_type == "QA"][0]
    assert qa.context == "Passage one." and qa.meta["question"] == "Q?"


def test_dict_source_is_serialised(rt_dir):
    d2t = [e for e in load_ragtruth(rt_dir, "train") if e.task_type == "Data2txt"][0]
    assert "Cafe" in d2t.context


def test_max_examples_is_deterministic(rt_dir):
    a = [e.id for e in load_ragtruth(rt_dir, None, max_examples=2, seed=1)]
    b = [e.id for e in load_ragtruth(rt_dir, None, max_examples=2, seed=1)]
    assert a == b and len(a) == 2


def test_inspect_lists_keys(rt_dir):
    text = inspect_ragtruth(rt_dir, n=1)
    assert "source_id" in text and "labels" in text


def test_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ragtruth(tmp_path)
