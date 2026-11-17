import pytest

from halludetect.config import load_config
from tests.stubs import StubHHEM, StubNLIModel


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def stub_nli():
    return StubNLIModel()


@pytest.fixture
def stub_hhem():
    return StubHHEM()
