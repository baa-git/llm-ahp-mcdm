"""Stage 5 with weights the user already has: TOPSIS, ARAS and EDAS rankings of a problem's alternatives.

ahp-bench rank --problem P --weights W.json [--out DIR]

W.json holds either global leaf weights {"<leaf criterion>": w, ...} or local weights per task
{"main": {"<main criterion>": w, ...}, "<task>": {"<sub-criterion>": w, ...}}. Weights are normalised to sum 1
(per task for local weights). Writes DIR/rank.md and DIR/tables/R01-R03 (CSV): rankings with the given
weights, the equal-weight baseline and the one-at-a-time sensitivity. No model is asked; the weights are the
user's, and the output says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import analysis as an
from . import problem as pb
from .report import Table, write_text


def leaf_weights(problem, data: dict) -> np.ndarray:
    data = {k: v for k, v in data.items() if not str(k).startswith("_")}
    leaves = problem.leaf_names
    if set(data) == set(leaves):
        w = np.array([float(data[x]) for x in leaves])
    elif set(data) <= set(problem.task_names) and pb.MAIN_TASK in data:
        local = {}
        for t, d in data.items():
            names = problem.criteria(t)
            if not isinstance(d, dict) or set(d) != set(names):
                raise pb.ProblemError(f"weights for task {t!r}: one value per criterion: {', '.join(names)}")
            v = np.array([float(d[n]) for n in names])
            local[t] = v / v.sum()
        w = problem.global_weights(local)
        if w is None:
            raise pb.ProblemError("local weights: every sub-criteria group needs its weights as well")
    else:
        raise pb.ProblemError("weights: give one value per leaf criterion (" + ", ".join(leaves) + ") or local "
                              "weights per task (" + ", ".join(problem.task_names) + ")")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise pb.ProblemError("weights must be finite, >= 0 and not all 0")
    return w / w.sum()


def rank_tables(problem, w) -> list:
    perf = problem.performance()
    X, benefit, alts = perf["X"], perf["benefit"], perf["alternatives"]
    cols, kinds = [], []
    for m in an.METHODS:
        cols += [f"{m} score", f"{m} rank"]
        kinds += ["score", "rank"]

    def table(tid, name, title, desc, label, weights):
        t = Table(tid, name, title, desc, ["Weights", "Alternative"] + cols, ["s", "s"] + kinds)
        res = an.rank_all(X, weights, benefit)
        for i, a in enumerate(alts):
            vals = []
            for m in an.METHODS:
                vals += [res[m]["scores"][i], res[m]["ranks"][i]]
            t.add(label, a, *vals)
        return t

    n = X.shape[1]
    out = [table("R01", "rankings_given_weights", "Rankings with the given weights",
                 "TOPSIS, ARAS and EDAS scores and ranks with the user's weights (indicator weights = leaf weights).",
                 "Given", problem.indicator_weights(w)),
           table("R02", "equal_weight_baseline", "Equal-weight baseline",
                 f"TOPSIS, ARAS and EDAS with all {n} indicator weights equal to 1/{n}.", f"Equal (1/{n})",
                 np.full(n, 1.0 / n))]
    steps = list(an.SENSITIVITY_STEPS)
    t = Table("R03", "weight_sensitivity", "One-at-a-time weight sensitivity",
              "For each indicator, its weight is set to each tested level while the others share the rest equally; "
              "the lowest level at which the top alternative differs from the equal-weight top (empty if never).",
              ["Indicator", "Method", "Equal-weight ranking", f"Ranking at {steps[0]:g}", f"Ranking at {steps[-1]:g}",
               "Top changes at"], ["s"] * 6)
    for row in an.one_at_a_time_sensitivity(X, benefit, alts):
        ch = row["top_change_at"]
        t.add(perf["codes"][row["indicator"]], row["method"], row["base_ranking"], row["rankings"][0],
              row["rankings"][-1], None if ch is None else f"{ch:g}")
    out.append(t)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ahp-bench rank", description="Rank alternatives with weights you already have.")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--weights", required=True, help="JSON: leaf criterion -> weight, or task -> {criterion: weight}")
    ap.add_argument("--out", default="rank", help="output folder (default: rank)")
    a = ap.parse_args(argv)
    try:
        p = pb.load_problem(a.problem)
        if not p.has_ranking:
            raise pb.ProblemError("the problem file has no performance matrix (ranking block)")
        w = leaf_weights(p, json.loads(Path(a.weights).read_text(encoding="utf-8-sig")))
    except (pb.ProblemError, OSError, ValueError) as e:
        print(f"cannot rank: {e}", file=sys.stderr)
        return 2
    out = Path(a.out)
    tables = rank_tables(p, w)
    md = [f"# Rankings with given weights: {p.title}", "",
          "The weights below were supplied by the user (not elicited by ahp-bench).", "",
          "| Leaf criterion | Weight |", "| --- | ---: |"] + [f"| {n} | {x:.4f} |" for n, x in zip(p.leaf_names, w)] + [""]
    for t in tables:
        write_text(out / "tables" / f"{t.stem}.csv", t.csv_text())
        md += [f"## {t.tid} {t.title}", "", t.description, "", t.markdown(), ""]
    write_text(out / "rank.md", "\n".join(md))
    print(tables[0].markdown())
    print(f"wrote {out / 'rank.md'} and {len(tables)} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
