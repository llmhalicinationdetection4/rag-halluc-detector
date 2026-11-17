"""Runs app.py end to end with a fake `streamlit` module and stand-in models (no browser, no downloads)."""
import runpy
import sys
from pathlib import Path

import halludetect.service as service
from tests.stubs import StubHHEM, StubNLIModel

APP = Path(__file__).resolve().parents[1] / "app.py"
REAL_DETECTOR = service.Detector  # keep the real class: the factory below replaces the module attribute


class _Ctx:
    def __init__(self, calls):
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        def record(*a, **k):
            self._calls.append((name, a, k))
            return _Ctx(self._calls)
        return record


class FakeStreamlit:
    def __init__(self, scenario_index=1, fast_mode=False, press=True):
        self.calls, self.scenario_index, self.fast_mode, self.press = [], scenario_index, fast_mode, press
        self.sidebar = _Sidebar(self)

    def cache_resource(self, *a, **k):
        return lambda fn: fn

    def columns(self, n):
        return [_Ctx(self.calls) for _ in range(n if isinstance(n, int) else len(n))]

    def text_area(self, label, height=None, value="", **k):
        return value

    def button(self, *a, **k):
        return self.press

    def __getattr__(self, name):
        def record(*a, **k):
            self.calls.append((name, a, k))
            return _Ctx(self.calls)
        return record

    def names(self):
        return [c[0] for c in self.calls]


class _Sidebar(_Ctx):
    def __init__(self, st):
        super().__init__(st.calls)
        self.st = st

    def radio(self, label, options, index=0, **k):
        return list(options)[self.st.scenario_index if label.startswith("Select Test") else index]

    def checkbox(self, *a, **k):
        return self.st.fast_mode


def _run(monkeypatch, **kw):
    fake = FakeStreamlit(**kw)
    monkeypatch.setitem(sys.modules, "streamlit", fake)

    def factory(use_nli, use_hhem):
        return REAL_DETECTOR(nli=StubNLIModel() if use_nli else None, hhem=StubHHEM() if use_hhem else None,
                             taus={"nli_whole_context": 0.5, "hhem": 0.5})

    monkeypatch.setattr(service, "Detector", factory)
    runpy.run_path(str(APP), run_name="__main__")
    return fake


def test_a_scan_renders_a_verdict_metrics_and_telemetry(monkeypatch):
    fake = _run(monkeypatch)
    names = fake.names()
    assert names.count("metric") == 3 and ("success" in names or "error" in names)
    assert "dataframe" in names and "progress" in names
    telemetry = [c for c in fake.calls if c[0] == "code" and c[2].get("language") == "json"]
    assert telemetry and '"primary_engine"' in telemetry[0][1][0]


def test_fast_mode_loads_only_hhem(monkeypatch):
    fake = _run(monkeypatch, fast_mode=True)
    telemetry = [c for c in fake.calls if c[0] == "code" and c[2].get("language") == "json"][0][1][0]
    assert '"hhem"' in telemetry


def test_empty_custom_scenario_only_warns(monkeypatch):
    fake = _run(monkeypatch, scenario_index=0)
    assert "warning" in fake.names() and "metric" not in fake.names()


def test_no_scan_without_a_button_press(monkeypatch):
    assert "metric" not in _run(monkeypatch, press=False).names()


def test_telemetry_is_not_hard_coded():
    text = APP.read_text(encoding="utf-8")
    assert "MiniLM" not in text and "Zero-Shot" not in text and "subprocess" not in text