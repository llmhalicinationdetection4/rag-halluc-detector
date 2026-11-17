"""RAGTruth loader.

IMPORTANT: this loader follows the layout described in the RAGTruth repository
(response.jsonl + source_info.jsonl, joined on source_id; a response is hallucinated when
its `labels` list is non-empty). Run `python -m halludetect inspect-ragtruth` first and compare
the printed keys with the RAGTruth README. If a key differs, fix the constants below.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from halludetect.data.schema import Example

RESPONSE_FILE = "response.jsonl"
SOURCE_FILE = "source_info.jsonl"


def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _check_files(data_dir: Path) -> tuple[Path, Path]:
    resp, src = data_dir / RESPONSE_FILE, data_dir / SOURCE_FILE
    missing = [p.name for p in (resp, src) if not p.exists()]
    if missing:
        listing = sorted(p.name for p in data_dir.iterdir()) if data_dir.exists() else "folder does not exist"
        raise FileNotFoundError(
            f"Missing {missing} in {data_dir}. Folder contents: {listing}. "
            "Clone RAGTruth and set data.ragtruth_dir in config/default.yaml."
        )
    return resp, src


def inspect_ragtruth(data_dir: str | Path, n: int = 3) -> str:
    """Return the keys of the first n records of each file, so you can verify the schema."""
    data_dir = Path(data_dir)
    resp, src = _check_files(data_dir)
    lines = []
    for path in (resp, src):
        lines.append(f"== {path.name} ==")
        for i, rec in enumerate(_read_jsonl(path)):
            if i >= n:
                break
            lines.append(f"record {i}: keys = {sorted(rec.keys())}")
            lines.append(f"  types: { {k: type(v).__name__ for k, v in rec.items()} }")
    return "\n".join(lines)


def _source_to_text(source_info) -> tuple[str, dict]:
    """Turn the source_info field into a context string. Returns (context, extra_meta)."""
    extra: dict = {}
    if isinstance(source_info, str):
        return source_info, extra
    if isinstance(source_info, dict):
        if "passages" in source_info:
            extra["question"] = source_info.get("question")
            passages = source_info["passages"]
            if isinstance(passages, list):
                passages = "\n".join(str(p) for p in passages)
            return str(passages), extra
        return json.dumps(source_info, ensure_ascii=False), extra
    return json.dumps(source_info, ensure_ascii=False), extra


def load_ragtruth(
    data_dir: str | Path,
    split: str | None = "train",
    max_examples: int | None = None,
    seed: int = 42,
) -> list[Example]:
    data_dir = Path(data_dir)
    resp_path, src_path = _check_files(data_dir)

    sources = {rec["source_id"]: rec for rec in _read_jsonl(src_path)}
    examples: list[Example] = []
    for rec in _read_jsonl(resp_path):
        rec_split = rec.get("split")
        if split is not None and rec_split is not None and rec_split != split:
            continue
        src = sources.get(rec["source_id"])
        if src is None:
            continue
        context, extra = _source_to_text(src.get("source_info"))
        labels = rec.get("labels") or []
        meta = {
            "split": rec_split,
            "model": rec.get("model"),
            "source_id": rec["source_id"],
            "n_label_spans": len(labels),
            "quality": rec.get("quality"),
            "label_types": [lab.get("label_type") for lab in labels if isinstance(lab, dict) and lab.get("label_type")],
            **extra,
        }
        examples.append(
            Example(
                id=str(rec.get("id", rec["source_id"])),
                context=context,
                answer=rec["response"],
                label_hallucinated=len(rec.get("labels") or []) > 0,
                task_type=src.get("task_type", "QA"),
                meta=meta,
            )
        )

    if max_examples is not None and len(examples) > max_examples:
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:max_examples]
    return examples
