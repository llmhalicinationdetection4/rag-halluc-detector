"""HHEM batching and out-of-memory handling, tested without torch or a GPU."""
import numpy as np

from halludetect.models.hhem import HHEMModel, plan_batches


def test_plan_batches_covers_every_index_once_and_respects_limits():
    lengths = [50, 4000, 300, 1000, 20, 1000, 700, 2500]
    batches = plan_batches(lengths, max_batch=16, budget=16 * 1024 * 1024)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lengths)))
    for b in batches:
        longest = max(lengths[i] for i in b)
        assert len(b) == 1 or len(b) * longest**2 <= 16 * 1024 * 1024
        assert len(b) <= 16


def test_plan_batches_puts_longest_first_and_shrinks_long_batches():
    lengths = [3000, 3000, 3000, 100, 100, 100, 100]
    batches = plan_batches(lengths, max_batch=16, budget=16 * 1024 * 1024)
    assert lengths[batches[0][0]] == 3000
    assert len(batches[0]) < 16
    assert plan_batches([], 16) == []


def _fake_model(predict):
    m = HHEMModel.__new__(HHEMModel)  # skip __init__: no torch needed
    m.batch_size = 16
    m.n_truncated = 0
    m._free_gpu_memory = lambda: None
    m._predict_batch = predict
    return m


def test_scores_come_back_in_the_original_order():
    # The score of a pair is its answer length, so we can check the ordering after length-sorting.
    m = _fake_model(lambda batch: [float(len(h)) for _, h in batch])
    pairs = [("c" * n, "h" * (i + 1)) for i, n in enumerate([5, 9000, 40, 2000, 7])]
    out = m.predict_consistency(pairs)
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_oom_splits_the_batch_instead_of_switching_to_cpu():
    succeeded = []

    def predict(batch):
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 3.00 GiB")
        succeeded.append(len(batch))
        return [0.5] * len(batch)

    m = _fake_model(predict)
    out = m.predict_consistency([("ctx", "ans")] * 8)
    assert np.allclose(out, 0.5) and len(out) == 8
    assert sum(succeeded) == 8 and max(succeeded) <= 2


def test_single_huge_context_is_shortened_and_counted():
    def predict(batch):
        premise, _ = batch[0]
        if len(premise) > 1000:
            raise RuntimeError("CUDA out of memory")
        return [0.9]

    m = _fake_model(predict)
    out = m.predict_consistency([("x" * 6000, "answer")])
    assert out.tolist() == [0.9] and m.n_truncated == 1


def test_other_runtime_errors_are_not_swallowed():
    def predict(batch):
        raise RuntimeError("Expected all tensors to be on the same device")

    m = _fake_model(predict)
    try:
        m.predict_consistency([("a", "b")])
    except RuntimeError as err:
        assert "same device" in str(err)
    else:
        raise AssertionError("a non-memory RuntimeError must be raised, not hidden")
