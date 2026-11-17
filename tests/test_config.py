import numpy as np

from halludetect.config import get_device, load_config, set_seed


def test_default_config_values():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.models.max_length == 512
    assert cfg.claim_level.top_k == 3


def test_seed_is_reproducible():
    set_seed(42)
    a = np.random.rand(3)
    set_seed(42)
    b = np.random.rand(3)
    assert (a == b).all()


def test_explicit_device_is_returned_unchanged():
    assert get_device("cpu") == "cpu"
