from __future__ import annotations

import json
from pathlib import Path
from typing import List
from halludetect.data.schema import Example


def load_demo_examples(file_path: str | Path = "data/sample/demo.jsonl") -> List[Example]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Demo file missing at: {path.resolve()}")

    examples: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            examples.append(
                Example(
                    id=str(item.get("id", f"demo-{idx}")),
                    context=str(item["context"]),
                    answer=str(item["answer"]),
                    label_hallucinated=bool(item["label_hallucinated"]),
                    task_type=str(item.get("task_type", "QA")),
                    meta=dict(item.get("meta", {})),
                )
            )
    return examples