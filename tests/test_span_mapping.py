from halludetect.eval.span_mapping import label_claims

CLAIMS = [{"start": 0, "end": 16}, {"start": 17, "end": 30}]


def test_overlap_marks_the_right_claims():
    assert label_claims(CLAIMS, [[17, 30]]) == [0, 1]
    assert label_claims(CLAIMS, [[10, 20]]) == [1, 1]      # a span can touch two claims
    assert label_claims(CLAIMS, []) == [0, 0]


def test_touching_ends_do_not_count_as_overlap():
    assert label_claims(CLAIMS, [[16, 17]]) == [0, 0]      # [16, 17) lies between the claims
    assert label_claims(CLAIMS, [[30, 40]]) == [0, 0]
