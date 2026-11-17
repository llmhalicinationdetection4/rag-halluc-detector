from halludetect.data.demo import load_demo


def test_demo_has_balanced_splits():
    train, test = load_demo(split="train"), load_demo(split="test")
    assert len(train) == 10 and len(test) == 10
    for split in (train, test):
        assert sum(e.label_hallucinated for e in split) == 5


def test_demo_fields_present():
    ex = load_demo()[0]
    assert ex.context and ex.answer and ex.id.startswith("demo-")
    assert ex.task_type in {"QA", "Summary", "Data2txt"}
