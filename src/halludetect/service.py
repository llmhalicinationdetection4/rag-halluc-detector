"""Verdict engine shared by the Streamlit dashboard (app.py) and other callers.

It turns one (context, answer) pair into a verdict using the two Phase 1 verifiers:
  hhem               Vectara HHEM-2.1-Open, scored through its own predict() (the stronger detector in our evaluation)
  nli_whole_context  DeBERTa-v3 NLI over the whole context (the weaker baseline)

The thresholds are the ones tuned on the RAGTruth TRAIN split in Phase 1 and stored in results/phase1_summary.json
(tau 0.990 for the NLI baseline, 0.824 for HHEM). They are never tuned on the demo scenarios.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from halludetect.config import REPO_ROOT

# Phase 1 values, used only if results/phase1_summary.json is missing.
DEFAULT_TAUS = {"nli_whole_context": 0.990, "hhem": 0.824}
ENGINE_LABELS = {"hhem": "Vectara HHEM-2.1-Open", "nli_whole_context": "DeBERTa-v3 NLI (whole context)"}
PREFERENCE_ORDER = ("hhem", "nli_whole_context")


def _resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_taus(summary_path=None, results_dir: str = "results") -> tuple[dict, str]:
    """Return ({model: tau}, where the values came from)."""
    path = _resolve(summary_path) if summary_path else _resolve(results_dir) / "phase1_summary.json"
    taus, source = dict(DEFAULT_TAUS), "built-in Phase 1 defaults"
    if path.exists():
        try:
            thresholds = json.loads(path.read_text(encoding="utf-8"))["thresholds"]
            found = {m: float(thresholds[m]["tuned_tau"]) for m in DEFAULT_TAUS if m in thresholds}
            if found:
                taus.update(found)
                source = path.name
        except (ValueError, KeyError, TypeError):
            pass  # unreadable summary: keep the defaults
    return taus, source


class Detector:
    def __init__(self, cfg=None, use_nli: bool = True, use_hhem: bool = True, nli=None, hhem=None,
                 summary_path=None, taus: dict | None = None):
        from halludetect.config import get_device, load_config

        self.cfg = cfg or load_config()
        self.device = get_device(self.cfg.device)
        if taus is not None:
            self.taus, self.tau_source = {**DEFAULT_TAUS, **taus}, "passed in"
        else:
            self.taus, self.tau_source = load_taus(summary_path, self.cfg.eval.results_dir)
        self.load_errors: dict[str, str] = {}
        self.nli, self.hhem = nli, hhem

        if self.nli is None and use_nli:
            try:
                from halludetect.models.nli import NLIModel

                m = self.cfg.models
                self.nli = NLIModel(m.nli, device=self.device, max_length=m.max_length,
                                    batch_size=m.batch_size, fallback_name=m.nli_fallback)
            except Exception as err:  # noqa: BLE001
                self.load_errors["nli_whole_context"] = f"{type(err).__name__}: {err}"
        if self.hhem is None and use_hhem:
            try:
                from halludetect.models.hhem import HHEMModel

                m = self.cfg.models
                self.hhem = HHEMModel(m.hhem, device=self.device, batch_size=m.batch_size)
            except Exception as err:  # noqa: BLE001
                self.load_errors["hhem"] = f"{type(err).__name__}: {err}"
        if self.nli is None and self.hhem is None:
            detail = "; ".join(f"{k}: {v}" for k, v in self.load_errors.items()) or "both engines are switched off"
            raise RuntimeError(f"No verifier could be loaded ({detail})")

    @property
    def engines(self) -> list[str]:
        return [name for name in PREFERENCE_ORDER if (self.hhem if name == "hhem" else self.nli) is not None]

    def verify(self, context: str, answer: str, primary: str | None = None) -> dict:
        context, answer = (context or "").strip(), (answer or "").strip()
        if not context or not answer:
            raise ValueError("Both the context and the answer are required.")
        t0 = time.perf_counter()
        models: dict[str, dict] = {}
        warnings: list[str] = []

        if self.nli is not None:
            p_entail, p_neutral, p_contra = (float(x) for x in self.nli.predict_proba([(context, answer)])[0])
            tau = self.taus["nli_whole_context"]
            max_len = int(getattr(self.nli, "max_length", 512))
            count = getattr(self.nli, "count_tokens", lambda text: len(text.split()))
            truncated = count(context) + count(answer) + 3 > max_len
            if truncated:
                warnings.append(f"The NLI baseline reads at most {max_len} tokens, so it saw only the start of this context.")
            models["nli_whole_context"] = {
                "engine": ENGINE_LABELS["nli_whole_context"],
                "model_name": getattr(self.nli, "model_name", "nli"),
                "score": p_entail, "p_entail": p_entail, "p_neutral": p_neutral, "p_contra": p_contra,
                "nli_label": ("Supported", "Unsupported", "Hallucinated")[max(range(3), key=(p_entail, p_neutral, p_contra).__getitem__)],
                "tau": tau, "flagged": bool(p_entail < tau), "context_truncated": bool(truncated),
            }

        if self.hhem is not None:
            score = float(self.hhem.predict_consistency([(context, answer)])[0])
            tau = self.taus["hhem"]
            models["hhem"] = {
                "engine": ENGINE_LABELS["hhem"],
                "model_name": getattr(self.hhem, "model_name", "hhem"),
                "score": score, "tau": tau, "flagged": bool(score < tau),
            }

        chosen = primary if primary in models else next(n for n in PREFERENCE_ORDER if n in models)
        if primary and primary not in models:
            warnings.append(f"{primary} is not loaded, so the verdict comes from {chosen}.")
        flags = [m["flagged"] for m in models.values()]
        verdict = models[chosen]
        return {
            "hallucinated": bool(verdict["flagged"]),
            "label": "Hallucinated" if verdict["flagged"] else "Supported",
            "primary_engine": chosen,
            "score": verdict["score"],
            "threshold": verdict["tau"],
            "engines_agree": (len(set(flags)) == 1) if len(flags) > 1 else None,
            "models": models,
            "warnings": warnings,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "device": self.device,
            "tau_source": self.tau_source,
            "note": "Verdict = score below the threshold tuned on the RAGTruth train split. "
                    "The thresholds were not tuned on these demo scenarios.",
        }
