"""Appendix tables A04-A08 (CSV; TeX with --latex), written by report.run_report.

A04 group (geometric-mean) matrices: experts and each condition's LLM panel (first answers).
A05 per-model geometric-mean matrices (model x condition x task).
A06 panel group matrices after revision (final answers of the revision tag; unchanged when no
    revision was needed). Only written when revision records exist.
A07 run-level table, one row per selected (model, condition, task, rep) record of the run tag.
A08 judge feedback and verdict per revision call of the revision tag. Only written when
    revision records exist.
All values are computed from the records; nothing is typed in.
"""
from __future__ import annotations

import numpy as np

from . import ahp
from . import analysis as an
from .prompts import condition_full as FULL
from .prompts import condition_label as L

NAMES = {"main": an.MAIN_SHORT, "delivery": an.DELIVERY_SHORT}
MAX_N = 5


def _f(v, d=4) -> str:
    if v is None:
        return ""
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, bool):
        return "y" if v else "n"
    if isinstance(v, float):
        if not np.isfinite(v):
            return ""
        s = f"{v:.{d}f}"
        return "0." + "0" * d if s == "-0." + "0" * d else s
    return str(v)


def _matrix_rows(prefix: list, task: str, g) -> list[list[str]]:
    """Rows of one aggregated matrix: prefix + criterion + a_i1..a_i5 (blank beyond n) + weight +
    lambda_max, CI, CR (repeated on each row)."""
    g = np.asarray(g, dtype=float)
    c = ahp.consistency(g)
    n = g.shape[0]
    rows = []
    for i in range(n):
        cells = [_f(float(g[i, j])) for j in range(n)] + [""] * (MAX_N - n)
        rows.append(prefix + [task, NAMES[task][i]] + cells
                    + [_f(float(c["weights"][i])), _f(c["lambda_max"]), _f(c["CI"]), _f(c["CR"])])
    return rows


MATRIX_COLS = ["task", "row criterion"] + [f"a_i{j}" for j in range(1, MAX_N + 1)] + [
    "weight", "lambda_max", "CI", "CR"]


def _group_rows(selected, conditions, models) -> list[list[str]]:
    rows = []
    for c in conditions:
        for task in an.TASKS:
            mats = [m for _, _, m in an.matrices(selected, c, task, models)]
            if mats:
                rows += _matrix_rows([f"LLM panel {FULL(c)}", str(len(mats))], task, ahp.geometric_mean_matrix(mats))
    return rows


def _verdict(rec) -> str:
    v = (rec or {}).get("judge_verdict")
    return v.get("logic_preserved", "") if isinstance(v, dict) else ""


def build(records, revision_records, expert, tag="main", revision_tag="revision") -> dict:
    """Return {filename_stem: (header, rows, appendix id, description)}."""
    selected, _, tagged = an.select_records(records, tag)
    models = sorted({r.get("model_key") for r in tagged if r.get("model_key")})
    conds = [c for c in an.CONDITIONS if any(k[1] == c for k in selected)]
    out = {}

    # A04 -------------------------------------------------------------------
    rows = []
    for task in an.TASKS:
        rows += _matrix_rows(["Experts", str(len(expert[task]))], task, ahp.geometric_mean_matrix(expert[task]))
    rows += _group_rows(selected, conds, set(models))
    out["A04_group_matrices"] = (["source", "matrices"] + MATRIX_COLS, rows, "A04",
                                 f"Geometric-mean group matrices (experts; LLM panel per condition, first answers of "
                                 f"tag '{tag}') with eigenvector weights, lambda_max, CI and CR.")

    # A05 -------------------------------------------------------------------
    rows = []
    for m in models:
        for c in conds:
            for task in an.TASKS:
                mats = [x for _, _, x in an.matrices(selected, c, task, {m})]
                if mats:
                    rows += _matrix_rows([m, L(c), str(len(mats))], task, ahp.geometric_mean_matrix(mats))
    out["A05_model_matrices"] = (["model", "condition", "reps"] + MATRIX_COLS, rows, "A05",
                                 "Per-model geometric-mean matrices of the repetitions (model x condition x task) "
                                 "with weights, lambda_max, CI and CR.")

    revs = [r for r in (revision_records or []) if r.get("run_tag") == revision_tag and r.get("source_tag") == tag]
    summ, by_key = [], {}
    if revs:
        from . import revise
        rconds = [c for c in revise.REVISION_CONDITIONS if any(r.get("condition") == c for r in revs)]
        summ = revise.chain_summary(records, revs, tag, revision_tag, rconds)
        by_key = {(s["model_key"], s["condition"], s["task"], s["rep"]): s for s in summ}
        final_sel = dict(selected)
        for k, s in by_key.items():
            final_sel[k] = {**selected[k], "matrix": s["final_matrix"]}
        rows = _group_rows(final_sel, rconds, set(models))
        out["A06_group_matrices_after_revision"] = (
            ["source", "matrices"] + MATRIX_COLS, rows, "A06",
            f"Geometric-mean LLM panel matrices after revision (tag '{revision_tag}': best valid answers of each "
            f"chain; matrices that needed no revision unchanged), conditions {'; '.join(map(FULL, rconds))}.")

    # A07 -------------------------------------------------------------------
    revs_by_chain = {}
    for r in revs:
        revs_by_chain.setdefault(an.record_key(r), []).append(r)
    header = (["model", "condition", "task", "rep", "CR"] + [f"w{i}" for i in range(1, MAX_N + 1)]
              + ["provider", "prompt tokens", "completion tokens", "reasoning tokens", "cost (USD)",
                 "parse_ok", "bare fractions quoted", "labels verified", "reordered", "reciprocal as returned",
                 "needs revision (CR > 0.10)", "revision calls", "revisions used", "final from revision",
                 "final CR", "verdict last call", "verdict final answers"])
    rows = []
    for key in sorted(selected):
        m, c, task, rep = key
        r = selected[key]
        w, cr = an.priority(r["matrix"])
        u = r.get("usage") or {}
        guided = c == "guided"
        s = by_key.get(key)
        chain = revs_by_chain.get(key, [])
        rows.append([m, L(c), task, str(rep), _f(cr)] + [_f(float(x)) for x in w] + [""] * (MAX_N - len(w))
                    + [str(r.get("provider") or ""), _f(u.get("prompt_tokens")), _f(u.get("completion_tokens")),
                       _f(u.get("reasoning_tokens")), _f(r.get("cost"), 6), _f(bool(r.get("parse_ok"))),
                       _f(r.get("bare_fractions_quoted")),
                       _f(r.get("labels_verified")) if guided else "", _f(r.get("reordered")) if guided else "",
                       _f(r.get("reciprocal_returned")) if guided else "",
                       _f(not an.is_consistent(cr)),
                       str(len(chain)) if s else "", str(s["revisions_used"]) if s and s["needs_revision"] else "",
                       str(s["final_revision"]) if s and s["needs_revision"] else "",
                       _f(s["final_CR"]) if s and s["needs_revision"] else "",
                       _verdict(chain[-1]) if chain else "",
                       _verdict(s["final_record"]) if s and s["final_record"] else ""])
    out["A07_run_level"] = (header, rows, "A07",
                            f"One row per selected record of tag '{tag}': CR, eigenvector weights (w1..w5 in the "
                            f"criterion order of the task), provider, tokens, cost, parse flags, revision need, "
                            f"revision calls, revisions used, final CR and judge verdicts (tag '{revision_tag}'; "
                            f"empty for conditions outside the revision design).")

    # A08 -------------------------------------------------------------------
    if revs:
        rows = []
        for r in revs:
            v = r.get("judge_verdict") if isinstance(r.get("judge_verdict"), dict) else {}
            rows.append([r.get("model_key"), L(r.get("condition")), r.get("task"), str(r.get("rep")),
                         str(r.get("revision_number")), _f(r.get("based_on_CR")),
                         _f(r.get("CR")) if r.get("parse_ok") else "",
                         _f(r.get("judge_feedback_ok")), _f(r.get("judge_feedback_reused")),
                         (r.get("judge_feedback") or "").replace("\n", " "),
                         v.get("logic_preserved", ""), "; ".join(v.get("changed_priorities") or []),
                         (v.get("comment") or "").replace("\n", " ")])
        out["A08_judge_feedback"] = (
            ["model", "condition", "task", "rep", "revision", "CR before", "CR after", "feedback ok",
             "feedback reused", "feedback", "verdict", "changed priorities", "comment"], rows, "A08",
            f"Judge feedback text and verdict for every revision call of tag '{revision_tag}' (CR before = CR of "
            f"the best answers the revision was built from; CR after empty when the reply was unparsable).")
    return out


def tex(stem: str, header, rows, description: str, text_cols=()) -> str:
    """longtable; columns listed in text_cols get a p{} paragraph column."""
    from .report import tex_escape
    align = "".join("p{4.5cm}" if h in text_cols else "l" for h in header)
    out = [f"% {stem}: {description}", f"\\begin{{longtable}}{{{align}}}", "\\toprule",
           " & ".join(tex_escape(h) for h in header) + " \\\\", "\\midrule", "\\endhead"]
    out += [" & ".join(tex_escape(c) for c in row) + " \\\\" for row in rows]
    out += ["\\bottomrule", "\\end{longtable}", ""]
    return "\n".join(out)


TEXT_COLS = {"A08_judge_feedback": ("feedback", "comment", "changed priorities")}
