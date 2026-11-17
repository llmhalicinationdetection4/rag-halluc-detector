import random
from halludetect.data.schema import Example
from halludetect.inject.injectors import (
    ERROR_TYPES,
    build_injections,
    inject_date_change,
    inject_entity_swap,
    inject_negation_flip,
    inject_number_swap,
    inject_unsupported_addition,
    read_injections,
    write_injections,
    write_samples_csv,
)

CTX = ("The Palestinian Authority became the 123rd member of the International Criminal Court. "
       "The Palestinians signed the Rome Statute in January 2015. Prosecutor Fatou Bensouda said her office would act. "
       "The war in Gaza left more than 2,000 people dead. The court was set up in 2002. "
       "Judge Kuniko Ozaki is a vice president of the ICC. Preheat the oven to 350 degrees and bake for 45 minutes.")
ANS = ("The Palestinian Authority joined the International Criminal Court in January 2015. "
       "The war in Gaza left more than 2,000 people dead, and Prosecutor Fatou Bensouda is not opposed. "
       "The court was set up in 2002. Bake at 350 degrees for 45 minutes. Judge Kuniko Ozaki is a vice president.")


def test_number_swap_changes_a_context_number_to_a_new_one():
    for seed in range(5):
        edited, desc, before, after = inject_number_swap(CTX, ANS, random.Random(seed))
        old, new = desc.split(" -> ")
        assert old in CTX and new not in CTX and old in ANS and edited != ANS
        assert edited.replace(new, old, 1) == ANS


def test_number_swap_skips_years_and_numbers_not_in_context():
    assert inject_number_swap("It opened in 2015 with 40 seats.", "It opened in 2015.", random.Random(0)) is None
    assert inject_number_swap("There were 40 seats.", "There were 99 seats.", random.Random(0)) is None


def test_entity_swap_uses_another_name_from_the_context():
    for seed in range(5):
        edited, desc, _, _ = inject_entity_swap(CTX, ANS, random.Random(seed))
        old, new = desc.split(" -> ")
        assert old in ANS and old in CTX and new in CTX and new not in ANS
        assert edited != ANS


def test_entity_swap_none_without_names():
    assert inject_entity_swap(CTX, "nothing capitalised here at all.", random.Random(0)) is None


def test_negation_flip_adds_or_removes_a_negation():
    edited, desc, _, _ = inject_negation_flip("ctx", "The shop is open. It cannot close early.", random.Random(1))
    assert edited != "The shop is open. It cannot close early."
    assert inject_negation_flip("ctx", "Cats purr.", random.Random(0)) is None
    e2, d2, _, _ = inject_negation_flip("ctx", "He isn't ready.", random.Random(0))
    assert e2 == "He is ready."


def test_date_change_only_touches_dates_found_in_the_context():
    edited, desc, _, _ = inject_date_change(CTX, ANS, random.Random(3))
    old, new = desc.split(" -> ")
    assert old in CTX and new not in CTX and edited != ANS
    assert inject_date_change(CTX, "It was long ago.", random.Random(0)) is None
    assert inject_date_change("It was 1999.", "It was 2015.", random.Random(0)) is None


def test_unsupported_addition_appends_one_sentence():
    edited, desc, _, _ = inject_unsupported_addition(CTX, ANS, random.Random(0))
    assert edited.startswith(ANS) and edited.endswith(desc) and desc not in ANS
    assert inject_unsupported_addition(CTX, "   ", random.Random(0)) is None


def _examples():
    return [Example(id=f"r{i}", context=CTX, answer=ANS, label_hallucinated=False,
                    task_type="QA" if i % 2 else "Summary", meta={"quality": "good"}) for i in range(12)]


def test_build_injections_caps_and_is_deterministic_and_order_independent():
    exs = _examples()
    a, stats = build_injections(exs, max_per_type=3, seed=7)
    b, _ = build_injections(list(reversed(exs)), max_per_type=3, seed=7)
    assert [(i.example_id, i.error_type, i.edited) for i in a] == [(i.example_id, i.error_type, i.edited) for i in b]
    for task in ("QA", "Summary"):
        for et in ERROR_TYPES:
            assert sum(1 for i in a if i.task_type == task and i.error_type == et) <= 3
    assert all(i.edited != i.original for i in a)
    assert {s["error_type"] for s in stats} == set(ERROR_TYPES)
    c, _ = build_injections(exs, max_per_type=3, seed=8)
    assert [i.edited for i in a] != [i.edited for i in c]


def test_write_and_read_roundtrip(tmp_path):
    injs, _ = build_injections(_examples(), max_per_type=2, seed=1)
    write_injections(tmp_path / "x.jsonl", injs)
    assert read_injections(tmp_path / "x.jsonl") == injs
    write_samples_csv(tmp_path / "s.csv", injs)
    assert (tmp_path / "s.csv").read_text(encoding="utf-8").startswith("task_type,error_type")


def test_run_injection_writes_files(cfg, tmp_path):
    from halludetect.inject.injectors import run_injection

    out = run_injection(cfg, max_per_type=2, out_path=tmp_path / "inj.jsonl", examples=_examples(), allow_regex_entities=True)
    assert (tmp_path / "inj.jsonl").exists() and (tmp_path / "phase2_injection_samples.csv").exists()
    assert (tmp_path / "phase2_injection_stats.csv").exists() and len(out["injections"]) > 0


def test_run_injection_ignores_hallucinated_and_low_quality_responses(cfg, tmp_path):
    from halludetect.inject.injectors import run_injection

    exs = _examples()
    exs[0].label_hallucinated = True
    exs[1].meta["quality"] = "incorrect_refusal"
    out = run_injection(cfg, max_per_type=50, out_path=tmp_path / "inj.jsonl", examples=exs, allow_regex_entities=True)
    used = {i.example_id for i in out["injections"]}
    assert "r0" not in used and "r1" not in used


def test_street_numbers_and_address_fragments_are_not_edited():
    ctx = '{"address": "1940 Cliff Dr, Ste B-13", "city": "Santa Barbara"}'
    ans = "It is at 1940 Cliff Dr in Santa Barbara."
    assert inject_date_change(ctx, ans, random.Random(0)) is None      # 1940 is a street number, not a year
    for seed in range(10):
        out = inject_entity_swap(ctx, ans, random.Random(seed))
        if out is not None:
            assert "Ste" not in out[1]


def test_lead_in_text_is_never_edited():
    ans = "Here is a summary of the article in 80 words: The shop is open and has 40 seats."
    ctx = "The shop is open and has 40 seats. It opened in 2015."
    for seed in range(10):
        for fn in (inject_negation_flip, inject_number_swap, inject_date_change):
            out = fn(ctx, ans, random.Random(seed))
            if out is not None:
                assert out[0].startswith("Here is a summary of the article in 80 words:"), out[1]
    assert inject_negation_flip("c", "Here is a summary:\nIt rains.", random.Random(0)) is None


def test_list_markers_ranges_and_reference_labels_are_not_edited():
    ctx = "1. Wash the slide. 2. Add 3 drops. 3. Dry it. Dig 6 to 8 inches deep. See Passage 3 for details."
    ans = "1. Wash the slide. 2. Add the drops. 3. Dry it. Dig 6 to 8 inches deep. See Passage 3."
    for seed in range(10):
        assert inject_number_swap(ctx, ans, random.Random(seed)) is None
    ans2 = "Add 3 drops to the slide."
    out = inject_number_swap(ctx, ans2, random.Random(0))
    assert out is not None and out[1].startswith("3 ->")


def test_entity_swap_keeps_the_entity_type():
    import halludetect.inject.injectors as inj

    ctx = "Tom Hanks visited Paris and Rome. Acme Corp opened an office."
    ans = "Tom Hanks visited Paris last year."
    fake = {
        ctx: [(0, 9, "Tom Hanks", "PERSON"), (18, 23, "Paris", "GPE"), (28, 32, "Rome", "GPE"), (35, 44, "Acme Corp", "ORG")],
        ans: [(0, 9, "Tom Hanks", "PERSON"), (18, 23, "Paris", "GPE")],
    }
    original = inj._entity_spans
    inj._entity_spans = lambda text: fake[text]
    try:
        results = {inject_entity_swap(ctx, ans, random.Random(seed))[1] for seed in range(20)}
    finally:
        inj._entity_spans = original
    assert results == {"Paris -> Rome"}      # the only candidate with another entity of the same type in the context


def test_run_injection_demands_spacy_unless_told_otherwise(cfg, tmp_path):
    import halludetect.inject.injectors as inj
    from halludetect.inject.injectors import run_injection

    if inj.ner_available():
        return  # spaCy is installed here, so there is nothing to refuse
    try:
        run_injection(cfg, max_per_type=1, out_path=tmp_path / "x.jsonl", examples=_examples())
    except RuntimeError as err:
        assert "spacy download" in str(err)
    else:
        raise AssertionError("expected a RuntimeError about the missing spaCy model")