from halludetect.analysis.buckets import (
    check_numbers_or_dates,
    classify_length_bucket,
    compute_rouge_l_terciles,
)
from halludetect.analysis.failure_report import compute_subset_metrics


def test_buckets():
    assert classify_length_bucket(100) == "<256"
    assert classify_length_bucket(300) == "256-512"
    assert classify_length_bucket(800) == "512-1024"
    assert classify_length_bucket(1500) == ">1024"

    assert check_numbers_or_dates("Launched on July 20, 1969")
    assert not check_numbers_or_dates("The ocean is very deep and vast")


def test_metrics_computation():
    y_true = [True, True, False, False]
    y_pred = [True, False, False, False]
    res = compute_subset_metrics(y_true, y_pred)
    assert res["recall"] == 0.5
    assert res["precision"] == 1.0
    assert res["error_rate"] == 0.25