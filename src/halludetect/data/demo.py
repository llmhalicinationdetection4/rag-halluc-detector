"""Loader for the handwritten demo set (no downloads needed)."""
from __future__ import annotations

import json
from pathlib import Path

from halludetect.data.schema import Example


def load_demo(path: str | Path = "data/sample/demo.jsonl", split: str | None = None) -> list[Example]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Demo file not found: {path}")
    examples: list[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = {"split": rec.get("split"), "error_type": rec.get("error_type")}
            ex = Example(
                id=rec["id"],
                context=rec["context"],
                answer=rec["answer"],
                label_hallucinated=bool(rec["label_hallucinated"]),
                task_type=rec.get("task_type", "QA"),
                meta=meta,
            )
            if split is None or meta["split"] == split:
                examples.append(ex)
    return examples
