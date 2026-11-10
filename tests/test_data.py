from pathlib import Path
from halludetect.data.demo import load_demo_examples
from halludetect.data.ragtruth import parse_ragtruth_record


def test_demo_data_loader():
    examples = load_demo_examples("data/sample/demo.jsonl")
    assert len(examples) == 20
    assert any(e.label_hallucinated for e in examples)
    assert any(not e.label_hallucinated for e in examples)


def test_ragtruth_parsing_logic():
    raw_record_clean = {
        "id": "rt-101",
        "source_info": "Paris is the capital of France.",
        "response": "Paris is France's capital city.",
        "labels": [],
        "task_type": "QA",
    }
    parsed_clean = parse_ragtruth_record(raw_record_clean)
    assert not parsed_clean.label_hallucinated
    assert parsed_clean.task_type == "QA"

    raw_record_halluc = {
        "id": "rt-102",
        "source_info": "Mars has two moons.",
        "response": "Mars has five moons.",
        "labels": [{"label": "Contradiction", "text": "five moons"}],
        "task_type": "Question Answering",
    }
    parsed_halluc = parse_ragtruth_record(raw_record_halluc)
    assert parsed_halluc.label_hallucinated
    assert parsed_halluc.task_type == "QA"