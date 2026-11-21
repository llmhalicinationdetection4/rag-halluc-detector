import json

from halludetect.data.ragtruth import load_ragtruth
from tests.fakes import write_fake_ragtruth


def test_label_spans_come_from_the_response_labels(tmp_path):
    (tmp_path / "source_info.jsonl").write_text(json.dumps(
        {"source_id": "s1", "task_type": "QA", "source_info": {"question": "Q", "passages": "P"}}) + "\n")
    (tmp_path / "response.jsonl").write_text(json.dumps(
        {"id": "r1", "source_id": "s1", "model": "m", "split": "test", "quality": "good", "response": "The sky is green. Grass is red.",
         "labels": [{"start": 0, "end": 17, "label_type": "Evident Conflict"}]}) + "\n")
    ex = load_ragtruth(tmp_path, "test")[0]
    assert ex.meta["label_spans"] == [[0, 17]] and ex.label_hallucinated is True


def test_faithful_responses_have_no_spans(tmp_path):
    folder = write_fake_ragtruth(tmp_path)
    examples = load_ragtruth(folder, "test")
    assert len(examples) == 24
    for ex in examples:
        assert (ex.meta["label_spans"] != []) == ex.label_hallucinated
