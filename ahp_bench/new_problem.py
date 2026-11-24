"""Write a commented template problem file.

python -m ahp_bench.new_problem [--out problems/my_problem.json] [--name my_problem] [--force]

JSON has no comments, so explanations are stored under keys starting with "_" (ignored by the loader).
The template is itself a valid problem; replace the names and numbers, then check the file with
python -m ahp_bench.problem PATH.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import problem as pb


def template(name: str = "my_problem") -> dict:
    return {
        "_help": "Template problem file for ahp_bench. Keys starting with '_' are comments. Each matrix needs 3-9 "
                 "criteria. Validate with: python -m ahp_bench.problem <this file>.",
        "schema_version": 1,
        "name": name,
        "_name_help": "Identifier: lower-case letters, digits and '_' (used for the default results folder "
                      "results/<name>).",
        "title": "Short human-readable title",
        "goal": "evaluating the options of my decision",
        "_goal_help": "Noun phrase inserted in the prompts: 'N criteria used in <goal> are compared in pairs' and "
                      "'what each criterion represents in <goal>'.",
        "role": "You are an experienced professional in this decision domain.",
        "_role_help": "First line of the system message. Describe a respondent, never a preferred answer.",
        "criteria": ["Criterion A", "Criterion B", "Criterion C", "Criterion D"],
        "short": ["A", "B", "C", "D"],
        "_short_help": "Optional short labels for tables (one per criterion).",
        "groups": [
            {
                "_help": "Optional sub-criteria of one main criterion; each group is a separate task. Delete the "
                         "'groups' list if there are no sub-criteria.",
                "task": "criterion_a",
                "parent": "Criterion A",
                "criteria": ["Sub-criterion A1", "Sub-criterion A2", "Sub-criterion A3"],
                "short": ["A1", "A2", "A3"],
            }
        ],
        "question_order": {
            "_help": "Optional. 1-based [left, right] pairs per task, each unordered pair exactly once. Default: "
                     "[1,2], [1,3], ..., [n-1,n]."
        },
        "ranking": {
            "_help": "Optional stage 2. Delete the whole 'ranking' block if there are no alternatives. Every leaf "
                     "criterion (a main criterion without sub-criteria, or a sub-criterion) maps to exactly one "
                     "indicator column. direction: 'benefit' (higher is better) or 'cost' (lower is better). All "
                     "matrix entries must be > 0. The numbers below are placeholders.",
            "alternatives": ["Option_1", "Option_2", "Option_3"],
            "indicators": [
                {"code": "A1", "name": "Indicator for A1", "unit": "unit", "direction": "benefit"},
                {"code": "A2", "name": "Indicator for A2", "unit": "unit", "direction": "benefit"},
                {"code": "A3", "name": "Indicator for A3", "unit": "unit", "direction": "cost"},
                {"code": "B", "name": "Indicator for B", "unit": "unit", "direction": "cost"},
                {"code": "C", "name": "Indicator for C", "unit": "unit", "direction": "benefit"},
                {"code": "D", "name": "Indicator for D", "unit": "unit", "direction": "benefit"},
            ],
            "criterion_indicator": {"Sub-criterion A1": "A1", "Sub-criterion A2": "A2", "Sub-criterion A3": "A3",
                                    "Criterion B": "B", "Criterion C": "C", "Criterion D": "D"},
            "matrix": [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]],
            "_transform_help": "Optional: {'type': 'row_share_percent', 'indicators': [codes], 'note': '...'} replaces "
                               "the listed columns by their percentage of the row total (e.g. counts -> rates).",
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ahp_bench.new_problem")
    ap.add_argument("--out", default="problems/my_problem.json")
    ap.add_argument("--name", default=None, help="problem identifier (default: the file name)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.force:
        print(f"{out} exists; use --force to overwrite", file=sys.stderr)
        return 2
    name = a.name or out.stem.lower().replace("-", "_")
    data = template(name)
    pb.problem_from_dict(data)  # the template must stay valid
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}; edit it, then run: python -m ahp_bench.problem {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
