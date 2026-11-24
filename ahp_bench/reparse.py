"""Re-parse stored raw replies with the current parser (no API calls).

python -m ahp_bench.reparse [--problem PATH] [--results DIR]
    rewrites <results>/raw/calls.jsonl, keeping a .bak copy. Raw replies, requests, usage and costs are
    never changed; only derived fields are recomputed. Not needed for verification: the audit (section g)
    performs the same comparison without writing anything.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys

from . import cli, run
from .parse import ParseError

DERIVED = ("labels", "label_order", "labels_verified", "reordered", "raw_matrix", "rationales",
           "reciprocal_returned", "upper_on_scale", "matrix", "weights", "lambda_max", "CI", "CR",
           "answers", "reasoning", "ranking", "bare_fractions_quoted", "guided_labels_expected")
PARSER_VERSION = 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ahp_bench.reparse")
    cli.add_common(ap, models=False)
    a = ap.parse_args(argv)
    ctx = cli.context(a, models=False)
    problem = None if ctx.problem.is_study else ctx.problem
    path = ctx.raw / "calls.jsonl"
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    shutil.copyfile(path, path.with_suffix(".jsonl.bak"))
    for r in recs:
        if r.get("ok") is False or r.get("raw_reply") is None:
            continue
        for k in DERIVED:
            r.pop(k, None)
        try:
            r.update(run.analyse(r["condition"], r["task"], r["raw_reply"], problem), parse_ok=True, parse_error=None)
        except ParseError as e:
            r.update(parse_ok=False, parse_error=str(e))
        r["parser_version"] = PARSER_VERSION
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    bad = [r for r in recs if not r.get("parse_ok")]
    print(f"{len(recs)} records, {len(bad)} not parsed")
    for r in bad:
        print("  ", r.get("run_tag"), r["model_key"], r["condition"], r["task"], r["rep"], str(r.get("parse_error"))[:100])
    return 0


if __name__ == "__main__":
    sys.exit(main())
