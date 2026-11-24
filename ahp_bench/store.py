"""JSONL record storage."""
from __future__ import annotations

import json
from pathlib import Path


def append_jsonl(path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def total_cost(path, run_tag: str | None = None) -> float:
    return sum(float(r.get("cost") or 0) for r in read_jsonl(path)
               if run_tag is None or r.get("run_tag") == run_tag)
