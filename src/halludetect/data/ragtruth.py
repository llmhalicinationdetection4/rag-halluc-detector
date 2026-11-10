from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from halludetect.data.schema import Example

logger = logging.getLogger(__name__)


def inspect_ragtruth_first_records(file_path: str | Path, num_records: int = 3) -> List[Dict[str, Any]]:
    path = Path(file_path)
    records: List[Dict[str, Any]] = []
    if not path.exists():
        logger.warning("RAGTruth file not found at: %s", path)
        return records

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= num_records:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))

    for idx, rec in enumerate(records):
        print(f"Record {idx + 1} Keys in {path.name}: {list(rec.keys())}")
    return records


def parse_ragtruth_record(rec: Dict[str, Any], split: str = "train") -> Example:
    rec_id = str(rec.get("id", rec.get("sample_id", rec.get("prompt_id", "ragtruth-unknown"))))
    
    # Handle context across RAGTruth file schemas (source_info, context, or passages)
    context_val = rec.get("source_info", rec.get("context", rec.get("passages", "")))
    if isinstance(context_val, list):
        context = "\n".join(str(p) for p in context_val)
    else:
        context = str(context_val)

    # Handle generated response / answer
    answer = str(rec.get("response", rec.get("answer", rec.get("generation", ""))))

    # A response is hallucinated if its labels list is non-empty
    labels = rec.get("labels", [])
    label_hallucinated = bool(labels and len(labels) > 0)

    # Task categorization: QA, Data2txt, Summary
    task_type = str(rec.get("task_type", rec.get("task", "QA")))
    task_lower = task_type.lower()
    if "qa" in task_lower or "question" in task_lower:
        canonical_task = "QA"
    elif "data2txt" in task_lower or "data_to_text" in task_lower or "table" in task_lower:
        canonical_task = "Data2txt"
    elif "sum" in task_lower:
        canonical_task = "Summary"
    else:
        canonical_task = task_type

    meta = {
        "split": split,
        "raw_labels": labels,
        "temperature": rec.get("temperature", None),
        "model": rec.get("model", None),
    }

    return Example(
        id=rec_id,
        context=context,
        answer=answer,
        label_hallucinated=label_hallucinated,
        task_type=canonical_task,
        meta=meta,
    )


def load_ragtruth_split(ragtruth_dir: str | Path, split: str = "test") -> List[Example]:
    directory = Path(ragtruth_dir)
    examples: List[Example] = []
    
    # RAGTruth files are typically structured as response_level/*.jsonl or split-based .jsonl
    candidate_files = list(directory.glob(f"*{split}*.jsonl")) + list(directory.glob(f"**/*{split}*.jsonl"))
    candidate_files = sorted(list({f.resolve() for f in candidate_files}))

    if not candidate_files:
        logger.warning("No RAGTruth split files found for split '%s' in %s", split, directory)
        return []

    for file_path in candidate_files:
        inspect_ragtruth_first_records(file_path, num_records=3)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                examples.append(parse_ragtruth_record(record, split=split))

    return examples