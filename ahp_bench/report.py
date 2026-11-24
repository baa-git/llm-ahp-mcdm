"""CLI: python -m ahp_bench.report --results STUDY/results --data STUDY/data [--tag main] [--out DIR] [--latex]

Reads <results>/raw/calls.jsonl (read-only), <data>/expert_matrices.json and
<data>/performance_matrix.json, and writes tables (CSV), figures (PDF + PNG), appendix CSVs and a
combined report.md; with --latex also booktabs/longtable TeX versions of the tables and appendix.
No network access. The condition stored under the key "lean" is labelled "direct"
(zero-shot, unguided) and "cot" as "CoT" (chain-of-thought, unguided) in every output.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import analysis as an
from .config import MODELS
from .prompts import condition_column as COL
from .prompts import condition_full as FULL
from .prompts import condition_label as L

PANEL = "Panel"
ONE_PER_DEV_DROP = ("claude-haiku-4.5", "gpt-5.2")
NO_GPT52_DROP = ("gpt-5.2",)

# ---------------------------------------------------------------- formatting

DECIMALS = {"w": 4, "cr": 4, "r": 3, "pp": 2, "usd": 4, "tok": 1, "score": 4, "cv": 3, "share": 4}
REVISION_STEPS = 5


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and not math.isfinite(v))


def fmt(v, kind: str) -> str:
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if _missing(v):
        return ""
    if isinstance(v, str) or kind == "s":
        return str(v)
    if kind == "i":
        return str(int(v))
    if kind == "rank":
        return str(int(v)) if float(v).is_integer() else f"{v:.1f}"
    d = DECIMALS[kind]
    s = f"{float(v):.{d}f}"
    return "0." + "0" * d if s == "-0." + "0" * d else s


def with_axes(header, rows):
    """Insert guidance and technique columns before a condition column (rows per condition)."""
    header = list(header)
    low = [str(h).lower() for h in header]
    if "condition" not in low or "guidance" in low:
        return header, [list(r) for r in rows]
    from .prompts import condition_guidance, condition_technique
    i = low.index("condition")
    names = ["Guidance", "Technique"] if header[i][0].isupper() else ["guidance", "technique"]
    out = [list(r[:i]) + [condition_guidance(r[i]), condition_technique(r[i])] + list(r[i:]) for r in rows]
    return header[:i] + names + header[i:], out


class Table:
    """A report table. A "Condition" column (short labels guided / direct / CoT) is preceded automatically by
    "Guidance" (guided / unguided) and "Technique" (zero-shot / CoT) columns."""

    def __init__(self, tid, name, title, description, columns, kinds):
        self.tid, self.name, self.title, self.description = tid, name, title, description
        self.columns, self.kinds, self.rows = list(columns), list(kinds), []
        self._cond = self.columns.index("Condition") if "Condition" in self.columns else None
        if self._cond is not None:
            self.columns[self._cond:self._cond] = ["Guidance", "Technique"]
            self.kinds[self._cond:self._cond] = ["s", "s"]

    def add(self, *values):
        if self._cond is not None:
            values = with_axes(["Condition"] if self._cond == 0 else ["x"] * self._cond + ["Condition"],
                               [list(values)])[1][0]
        if len(values) != len(self.columns):
            raise ValueError(f"{self.tid}: row has {len(values)} values, expected {len(self.columns)}")
        self.rows.append([fmt(v, k) for v, k in zip(values, self.kinds)])

    @property
    def stem(self):
        return f"{self.tid}_{self.name}"

    def csv_text(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(self.columns)
        w.writerows(self.rows)
        return buf.getvalue()

    def markdown(self) -> str:
        esc = lambda s: s.replace("|", "\\|")
        lines = ["| " + " | ".join(esc(c) for c in self.columns) + " |",
                 "| " + " | ".join("---" if k == "s" else "---:" for k in self.kinds) + " |"]
        lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in self.rows]
        return "\n".join(lines)

    def tex(self) -> str:
        align = "".join("l" if k == "s" else "r" for k in self.kinds)
        out = [f"% {self.tid} {self.title}", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
               " & ".join(tex_escape(c) for c in self.columns) + " \\\\", "\\midrule"]
        out += [" & ".join(tex_escape(c) for c in row) + " \\\\" for row in self.rows]
        out += ["\\bottomrule", "\\end{tabular}", ""]
        return "\n".join(out)


_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "<": r"\textless{}", ">": r"\textgreater{}", "≤": r"$\leq$", "−": "-", "×": r"$\times$"}


def tex_escape(s: str) -> str:
    return "".join(_TEX.get(ch, ch) for ch in str(s))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_csv(path: Path, header, rows) -> None:
    header, rows = with_axes(header, rows)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    write_text(path, buf.getvalue())


# The study's conditions on two axes (report.md and the READMEs open the condition description with these).
CONDITIONS_MD = [
    "## Conditions", "",
    "Conditions are described on two axes: prompting technique (zero-shot = no worked examples and no reasoning "
    "instruction; chain-of-thought = an instruction to reason step by step) and weight guidance (guided = the prompt "
    "states a priority order and importance hints; unguided = neutral questions). Short labels: guided, direct "
    "(stored key `lean`) and CoT (stored key `cot`).", "",
    "| Technique / guidance | Guided | Unguided |",
    "| --- | --- | --- |",
    "| Zero-shot | Guided (zero-shot): `guided` | Unguided, zero-shot (direct): `lean` |",
    "| Chain-of-thought | not run | Unguided, CoT: `cot` |", "",
    "| Prompt content | Guided (zero-shot) | Unguided, zero-shot (direct) | Unguided, CoT |",
    "| --- | --- | --- | --- |",
    "| Role line | AHP expert specialised in supplier evaluation, with AHP rules | procurement professional with "
    "experience in supplier evaluation | same as direct |",
    "| Priority order | yes: typical priority order (Quality highest ... Volume lower); \"never use all 1s\" | no | no |",
    "| Criterion hints | yes: importance-laden descriptions of the criteria (in Turkish) | no (names only) | no (names only) |",
    "| KPI report | yes: data report on the suppliers (in Turkish) | no | no |",
    "| Reasoning instruction | rationale per comparison requested | none | reason step by step and state a ranking "
    "before answering |",
    "| Reply format | full pairwise matrix with labels and rationales (JSON) | one value per question (JSON) | "
    "reasoning, ranking and one value per question (JSON) |", "",
]


# ---------------------------------------------------------------- helpers

def _pp(a, b):
    return None if a is None or b is None else (a - b) * 100.0


def _mean(xs):
    xs = [x for x in xs if not _missing(x)]
    return float(np.mean(xs)) if xs else None


def _count_ok(crs):
    return f"{sum(an.is_consistent(c) for c in crs)}/{len(crs)}" if crs else None


def _reasoning_label(v) -> str:
    if v is None:
        return "omitted"
    if isinstance(v, dict):
        return ",".join(f"{k}={v[k]}" for k in sorted(v))
    return str(v)


def _sim_values(w, ref):
    if w is None:
        return [None] * 4
    s = an.similarity(w, ref)
    return [s["spearman"], s["kendall"], s["rmse_pp"], s["mad_pp"]]


SIM_COLS = ["Spearman rho", "Kendall tau", "RMSE (pp)", "Mean abs diff (pp)"]
SIM_KINDS = ["r", "r", "pp", "pp"]


# ---------------------------------------------------------------- build

def build(records, tag, expert, perf, revision_records=None, revision_tag="revision"):
    """Compute every table and figure input. Returns (tables, figure_data, appendix).
    T16/T17/T20 need revision records for `revision_tag`; T18 needs temp_<value> run tags."""
    selected, failures, tagged = an.select_records(records, tag)
    models = sorted({r.get("model_key") for r in tagged if r.get("model_key")})
    conds = list(an.CONDITIONS)
    ex = an.expert_weights(expert)
    tables = []

    model_panel = {(m, c): an.panel_weights(selected, c, {m}) for m in models for c in conds}
    cond_panel = {c: an.panel_weights(selected, c, set(models)) for c in conds}

    # T01 ------------------------------------------------------------------
    t = Table("T01", "models_settings", "Models and settings",
              "Lists each model with its developer, OpenRouter identifier, requested temperature, "
              "applied reasoning setting, mean reasoning tokens, providers seen, number of calls, "
              "number of parse failures and total cost (USD) for the selected run tag.",
              ["Model", "Developer", "OpenRouter id", "Temperature (requested)", "Reasoning (applied)",
               "Mean reasoning tokens", "Providers seen", "Calls", "Parse failures", "Total cost (USD)"],
              ["s", "s", "s", "s", "s", "tok", "s", "i", "i", "usd"])
    by_model = defaultdict(list)
    for r in tagged:
        by_model[r.get("model_key")].append(r)
    for m in models:
        rs = by_model[m]
        cfg = MODELS.get(m, {})
        dev = next((r["developer"] for r in rs if r.get("developer")), cfg.get("developer"))
        mid = next((r["model_id"] for r in rs if r.get("model_id")), cfg.get("id"))
        temps = sorted({str((r.get("requested_params") or {}).get("temperature")) for r in rs
                        if (r.get("requested_params") or {}).get("temperature") is not None})
        reas = sorted({_reasoning_label((r.get("applied_params") or {}).get("reasoning")) for r in rs
                       if r.get("applied_params") is not None})
        rtok = [(r.get("usage") or {}).get("reasoning_tokens") for r in rs]
        provs = sorted({str(r["provider"]) for r in rs if r.get("provider")})
        nfail = sum(1 for r in rs if not r.get("parse_ok"))
        cost = sum(float(r.get("cost") or 0) for r in rs)
        t.add(m, dev, mid, "; ".join(temps) or None, "; ".join(reas) or None, _mean(rtok),
              "; ".join(provs) or None, len(rs), nfail, cost)
    tables.append(t)

    # T02 ------------------------------------------------------------------
    t = Table("T02", "expert_consistency", "Expert consistency",
              "Gives the consistency ratio (CR) of each expert's main-criteria and delivery "
              "sub-criteria matrix, the CR of the geometric-mean group matrices, and the number of "
              "matrices with CR ≤ 0.10.",
              ["Expert", "CR main", "CR delivery"], ["s", "cr", "cr"])
    for name, a, b in zip(expert["experts"], ex["cr_main"], ex["cr_delivery"]):
        t.add(name, a, b)
    t.add("Group (geometric mean)", ex["group_cr_main"], ex["group_cr_delivery"])
    t.add("n with CR ≤ 0.10", _count_ok(ex["cr_main"]), _count_ok(ex["cr_delivery"]))
    tables.append(t)

    # T03 ------------------------------------------------------------------
    t = Table("T03", "llm_consistency", "LLM consistency",
              "Gives, per model and condition, the mean and maximum CR over repetitions and the "
              "number of matrices with CR ≤ 0.10 for the main and delivery tasks; panel rows pool all "
              "models and add the CR of the geometric-mean group matrix.",
              ["Model", "Condition", "Mean CR main", "Max CR main", "CR ≤ 0.10 main", "Group CR main",
               "Mean CR delivery", "Max CR delivery", "CR ≤ 0.10 delivery", "Group CR delivery"],
              ["s", "s", "cr", "cr", "s", "cr", "cr", "cr", "s", "cr"])
    crs = {}
    for (m, c, task, rep), r in selected.items():
        crs.setdefault((m, c, task), []).append(an.priority(r["matrix"])[1])
    for m in models:
        for c in conds:
            vals = []
            for task in an.TASKS:
                x = crs.get((m, c, task), [])
                vals += [_mean(x), max(x) if x else None, _count_ok(x), None]
            t.add(m, L(c), *vals)
    for c in conds:
        vals = []
        for task in an.TASKS:
            x = [v for (m, cc, tt), xs in sorted(crs.items()) if cc == c and tt == task for v in xs]
            vals += [_mean(x), max(x) if x else None, _count_ok(x), cond_panel[c][f"cr_{task}"]]
        t.add(PANEL, L(c), *vals)
    tables.append(t)

    # T04-T06 --------------------------------------------------------------
    for tid, name, title, key, labels, what in [
        ("T04", "main_weights", "Main-criteria weights", "main", an.MAIN_CRITERIA, "five main criteria"),
        ("T05", "delivery_local_weights", "Delivery local weights", "delivery", an.DELIVERY_SHORT,
         "four delivery sub-criteria (local weights)"),
        ("T06", "global_weights", "Global weights", "global", an.GLOBAL_LABELS, "eight global indicators"),
    ]:
        cols, kinds = ["Criterion", "Experts"], ["s", "w"]
        for c in conds:
            cols += [f"LLM {COL(c)}", f"Diff {COL(c)} (pp)"]
            kinds += ["w", "pp"]
        t = Table(tid, name, title,
                  f"Lists the expert group weights and each condition's LLM panel group weights for the "
                  f"{what}, with the LLM minus expert difference in percentage points.", cols, kinds)
        for i, lab in enumerate(labels):
            row = [lab, ex[key][i]]
            for c in conds:
                w = cond_panel[c][key]
                v = None if w is None else w[i]
                row += [v, _pp(v, ex[key][i])]
            t.add(*row)
        tables.append(t)

    # T07 ------------------------------------------------------------------
    t = Table("T07", "model_weights_main", "Model-level main weights",
              "Gives, per model and condition, the main-criteria weights from the geometric mean of "
              "the model's repetition matrices, the number of repetitions used and the top-ranked "
              "criterion.",
              ["Model", "Condition", "Reps"] + an.MAIN_SHORT + ["Top criterion"],
              ["s", "s", "i"] + ["w"] * 5 + ["s"])
    for m in models:
        for c in conds:
            p = model_panel[(m, c)]
            w = p["main"]
            t.add(m, L(c), p["n_main"] or None, *([None] * 5 if w is None else list(w)),
                  None if w is None else an.MAIN_SHORT[int(np.argmax(w))])
    tables.append(t)

    # T08 ------------------------------------------------------------------
    t = Table("T08", "similarity_to_experts", "Similarity to expert weights",
              "Gives Spearman rho, Kendall tau, RMSE and mean absolute difference (percentage points) "
              "between each model's (and each condition panel's) group weights and the expert group "
              "weights, for the five main and the eight global weights.",
              ["Model", "Condition"] + [f"{s} main" for s in SIM_COLS] + [f"{s} global" for s in SIM_COLS],
              ["s", "s"] + SIM_KINDS * 2)
    for m in models:
        for c in conds:
            p = model_panel[(m, c)]
            t.add(m, L(c), *_sim_values(p["main"], ex["main"]), *_sim_values(p["global"], ex["global"]))
    for c in conds:
        p = cond_panel[c]
        t.add(PANEL, L(c), *_sim_values(p["main"], ex["main"]), *_sim_values(p["global"], ex["global"]))
    tables.append(t)

    # T09 ------------------------------------------------------------------
    t = Table("T09", "run_to_run_stability", "Run-to-run stability",
              "Gives, per model and condition, the sample standard deviation and coefficient of "
              "variation of each main-criteria weight across repetitions, their mean CV, and the "
              "number of distinct top-ranked criteria across repetitions.",
              ["Model", "Condition", "Reps"] + [f"SD {s}" for s in an.MAIN_SHORT]
              + [f"CV {s}" for s in an.MAIN_SHORT] + ["Mean CV", "Distinct top criteria"],
              ["s", "s", "i"] + ["w"] * 5 + ["cv"] * 5 + ["cv", "i"])
    for m in models:
        for c in conds:
            ws = [an.priority(mat)[0] for _, _, mat in an.matrices(selected, c, "main", {m})]
            if len(ws) >= 2:
                arr = np.array(ws)
                sd = arr.std(axis=0, ddof=1)
                cv = sd / arr.mean(axis=0)
                tops = len({int(np.argmax(w)) for w in ws})
                t.add(m, L(c), len(ws), *sd, *cv, float(cv.mean()), tops)
            else:
                tops = 1 if ws else None
                t.add(m, L(c), len(ws) or None, *([None] * 11), tops)
    tables.append(t)

    # T10 ------------------------------------------------------------------
    variants = [("All models", set(models)),
                ("One per developer", set(models) - set(ONE_PER_DEV_DROP)),
                ("Without gpt-5.2", set(models) - set(NO_GPT52_DROP))]
    t = Table("T10", "one_per_developer", "Panel composition variants",
              "Gives, per condition, the panel main-criteria group weights and their similarity to the "
              "expert weights (main and global) for all models, one model per developer "
              "(claude-haiku-4.5 and gpt-5.2 removed) and the panel without gpt-5.2.",
              ["Condition", "Panel", "Models"] + an.MAIN_SHORT
              + [f"{s} main" for s in SIM_COLS] + [f"{s} global" for s in SIM_COLS],
              ["s", "s", "i"] + ["w"] * 5 + SIM_KINDS * 2)
    for c in conds:
        for label, ms in variants:
            p = an.panel_weights(selected, c, ms)
            w = p["main"]
            n_used = len({m for m, _, _ in an.matrices(selected, c, "main", ms)})
            t.add(L(c), label, n_used or None, *([None] * 5 if w is None else list(w)),
                  *_sim_values(w, ex["main"]), *_sim_values(p["global"], ex["global"]))
    tables.append(t)

    # T11-T13 --------------------------------------------------------------
    X_counts = perf["matrix"]
    X_rates = an.delivery_rates(X_counts)
    benefit = perf["benefit_mask"]
    alts = perf["alternatives"]
    rank_cols, rank_kinds = [], []
    for mth in an.METHODS:
        rank_cols += [f"{mth} score", f"{mth} rank"]
        rank_kinds += ["score", "rank"]

    def rank_rows(t, source, w, X):
        res = an.rank_all(X, w, benefit) if w is not None else None
        for i, a in enumerate(alts):
            vals = []
            for mth in an.METHODS:
                vals += [None, None] if res is None else [res[mth]["scores"][i], res[mth]["ranks"][i]]
            t.add(source, a, *vals)

    sources = [("Experts", ex["global"])] + [(f"LLM panel {FULL(c)}", cond_panel[c]["global"]) for c in conds]
    t = Table("T11", "supplier_rankings", "Supplier rankings (rates)",
              "Gives TOPSIS, ARAS and EDAS scores and ranks of the suppliers under the expert global "
              "weights and each condition's LLM panel global weights, with delivery indicators as rates.",
              ["Weights", "Supplier"] + rank_cols, ["s", "s"] + rank_kinds)
    for src, w in sources:
        rank_rows(t, src, w, X_rates)
    tables.append(t)

    cols, kinds = ["Weights", "Supplier"], ["s", "s"]
    for mth in an.METHODS:
        cols += [f"{mth} rank counts", f"{mth} rank rates"]
        kinds += ["rank", "rank"]
    t = Table("T12", "rankings_counts_vs_rates", "Rankings: counts vs rates",
              "Gives supplier ranks from TOPSIS, ARAS and EDAS with delivery indicators as counts and as "
              f"rates, under the expert global weights and the LLM panel global weights of the {FULL('lean')} "
              "condition.", cols, kinds)
    for src, w in [("Experts", ex["global"]), (f"LLM panel {FULL('lean')}", cond_panel["lean"]["global"])]:
        rc = an.rank_all(X_counts, w, benefit) if w is not None else None
        rr = an.rank_all(X_rates, w, benefit) if w is not None else None
        for i, a in enumerate(alts):
            vals = []
            for mth in an.METHODS:
                vals += [None, None] if w is None else [rc[mth]["ranks"][i], rr[mth]["ranks"][i]]
            t.add(src, a, *vals)
    tables.append(t)

    t = Table("T13", "equal_weight_baseline", "Equal-weight baseline",
              "Gives TOPSIS, ARAS and EDAS scores and ranks of the suppliers with all eight global "
              "weights equal to 1/8 and delivery indicators as rates.",
              ["Weights", "Supplier"] + rank_cols, ["s", "s"] + rank_kinds)
    rank_rows(t, "Equal (1/8)", np.full(8, 1 / 8), X_rates)
    tables.append(t)
    agreement_sources = [(f"LLM panel {FULL(c)}", cond_panel[c]["global"]) for c in conds] + [
        ("Equal (1/8)", np.full(8, 1 / 8))]

    # T14-T15 (main-criterion weight sensitivity) -------------------------
    initials = an.supplier_initials(alts)
    sens = an.main_weight_sensitivity(X_rates, benefit, alts)
    levels = list(an.MAIN_LEVELS)
    meth_cols, meth_kinds = [], []
    for mth in an.METHODS:
        meth_cols += [f"{mth} ranking", f"{mth} top supplier"]
        meth_kinds += ["s", "s"]
    t = Table("T14", "weight_sensitivity", "Main-criterion weight sensitivity",
              "For each of the five main criteria in turn, sets its main weight to each level from "
              f"{levels[0]:.2f} to {levels[-1]:.2f} in steps of 0.05 while the other four main criteria share "
              "the remainder equally; the Delivery main weight is split equally over D1-D4, giving eight global "
              "weights. Gives the global weights and, for TOPSIS, ARAS (eta = 1) and EDAS with delivery "
              "indicators as rates, the supplier ranking (initials, best first: "
              + ", ".join(f"{i} = {a}" for i, a in zip(initials, alts))
              + ") and the top supplier. The first row is the equal-weight reference (all eight global weights 1/8).",
              ["Varied criterion", "Level"] + an.GLOBAL_CODES + meth_cols,
              ["s", "s"] + ["w"] * 8 + meth_kinds)
    for row in sens:
        vals = []
        for mth in an.METHODS:
            vals += [row["methods"][mth]["ranking"], row["methods"][mth]["top"]]
        t.add("Equal" if row["criterion"] is None else row["criterion"],
              None if row["level"] is None else f"{row['level']:.2f}", *row["weights"], *vals)
    tables.append(t)

    t = Table("T15", "sensitivity_summary", "Weight sensitivity summary",
              "Summarises T14 per varied main criterion and method: the top supplier at the lowest "
              f"({levels[0]:.2f}) and highest ({levels[-1]:.2f}) level, the number of distinct supplier rankings "
              f"over the {len(levels)} levels, the number of levels whose top supplier equals the equal-weight top "
              "supplier, and the lowest level at which the top supplier differs from the equal-weight top "
              "supplier (empty if never).",
              ["Varied criterion", "Method", "Equal-weight top supplier", f"Top supplier at {levels[0]:.2f}",
               f"Top supplier at {levels[-1]:.2f}", "Distinct rankings", "Levels with equal-weight top",
               "Top supplier first differs at"],
              ["s", "s", "s", "s", "s", "i", "i", "s"])
    for r in an.sensitivity_summary(sens):
        ch = r["first_change_at"]
        t.add(r["criterion"], r["method"], r["equal_top"], r["top_low"], r["top_high"], r["distinct_rankings"],
              r["levels_same_top"], None if ch is None else f"{ch:.2f}")
    tables.append(t)

    # T16-T17 (revision) ---------------------------------------------------
    revs = [r for r in (revision_records or []) if r.get("run_tag") == revision_tag and r.get("source_tag") == tag]
    if revs:
        from . import revise
        rconds = [c for c in revise.REVISION_CONDITIONS if any(r.get("condition") == c for r in revs)]
        summ = revise.chain_summary(records, revs, tag, revision_tag, rconds)
        ks = list(range(1, REVISION_STEPS + 1))
        t = Table("T16", "revision_summary", "Revision summary",
                  f"Gives, per model, condition and task (revision run tag '{revision_tag}', source tag '{tag}'), "
                  "the number of source matrices, the number with CR > 0.10 (needing revision), the cumulative "
                  "number that reached CR ≤ 0.10 after at most 1 to 5 revisions, the number not reached, the "
                  "number of chains with no revision record yet, and the mean initial and mean final CR of the "
                  "matrices needing revision (final = the best valid answers of the chain, i.e. the lowest CR "
                  "among the first answers and all parsable revisions); CR ≤ 0.10 means CR rounded to 4 "
                  "decimals is at most 0.10; panel rows pool all models.",
                  ["Model", "Condition", "Task", "Matrices", "Needing revision"]
                  + [f"Reached after ≤{k}" for k in ks] + ["Not reached", "No revision record",
                                                           "Mean initial CR", "Mean final CR"],
                  ["s", "s", "s", "i", "i"] + ["i"] * len(ks) + ["i", "i", "cr", "cr"])

        def rev_row(label, c, task, ss):
            need = [s for s in ss if s["needs_revision"]]
            t.add(label, L(c), task, len(ss), len(need),
                  *[sum(1 for s in need if s["reached_at"] is not None and s["reached_at"] <= k) for k in ks],
                  sum(1 for s in need if not s["reached"]), sum(1 for s in need if s["revisions_used"] == 0),
                  _mean([s["initial_CR"] for s in need]), _mean([s["final_CR"] for s in need]))
        for m in models:
            for c in rconds:
                for task in an.TASKS:
                    ss = [s for s in summ if s["model_key"] == m and s["condition"] == c and s["task"] == task]
                    if ss:
                        rev_row(m, c, task, ss)
        for c in rconds:
            for task in an.TASKS:
                rev_row(PANEL, c, task, [s for s in summ if s["condition"] == c and s["task"] == task])
        tables.append(t)

        final_sel = dict(selected)
        for s in summ:
            key = (s["model_key"], s["condition"], s["task"], s["rep"])
            final_sel[key] = {**selected[key], "matrix": s["final_matrix"]}
        t = Table("T17", "weights_before_after_revision", "Weights before and after revision",
                  "Gives, per condition, the LLM panel group weights (main criteria, delivery local and global) "
                  "computed from the first answers and from the final answers after revision (the best valid "
                  "answers of each chain; matrices that needed no revision are used unchanged), next to the expert group weights, with Spearman rho "
                  "and RMSE (percentage points) against the expert main and global weights.",
                  ["Condition", "Answers", "Main matrices"] + an.MAIN_SHORT + an.DELIVERY_SHORT + an.GLOBAL_CODES
                  + ["Spearman rho main", "RMSE main (pp)", "Spearman rho global", "RMSE global (pp)"],
                  ["s", "s", "i"] + ["w"] * 17 + ["r", "pp", "r", "pp"])
        t.add("Experts", "", len(expert["main"]), *ex["main"], *ex["delivery"], *ex["global"],
              None, None, None, None)
        for c in rconds:
            for label, sel in (("first", selected), ("final", final_sel)):
                p = an.panel_weights(sel, c, set(models))
                sm, sg = _sim_values(p["main"], ex["main"]), _sim_values(p["global"], ex["global"])
                t.add(L(c), label, p["n_main"] or None,
                      *([None] * 5 if p["main"] is None else p["main"]),
                      *([None] * 4 if p["delivery"] is None else p["delivery"]),
                      *([None] * 8 if p["global"] is None else p["global"]), sm[0], sm[2], sg[0], sg[2])
        tables.append(t)

    # T18 (temperature) ----------------------------------------------------
    temp_tags = sorted({r.get("run_tag") for r in records
                        if isinstance(r.get("run_tag"), str) and r["run_tag"].startswith("temp_")})
    if temp_tags:
        t = Table("T18", "temperature_comparison", f"Temperature comparison ({FULL('lean')})",
                  f"Gives, for the {FULL('lean')} condition, per run tag and requested temperature (the main tag at its "
                  "recorded temperature, temp_<value> tags at <value>), the panel main-criteria group weights, "
                  "the number of main and delivery matrices with CR ≤ 0.10, and the mean over models of each "
                  "model's mean coefficient of variation of the main weights across repetitions.",
                  ["Run tag", "Temperature", "Models"] + an.MAIN_SHORT
                  + ["CR ≤ 0.10 main", "CR ≤ 0.10 delivery", "Mean CV main weights"],
                  ["s", "s", "i"] + ["w"] * 5 + ["s", "s", "cv"])
        rows18 = []
        for rt in [tag] + temp_tags:
            sel_t, _, tagged_t = an.select_records(records, rt)
            if rt == tag:
                temps = set()
                for r in tagged_t:
                    if r.get("condition") != "lean":
                        continue
                    v = r.get("temperature_requested", (r.get("requested_params") or {}).get("temperature"))
                    if v is not None:
                        temps.add(str(v))
                temp = "; ".join(sorted(temps)) or None
            else:
                temp = rt[len("temp_"):]
            ms = sorted({k[0] for k in sel_t if k[1] == "lean"})
            p = an.panel_weights(sel_t, "lean", set(ms))
            cr_counts = [_count_ok([an.priority(r["matrix"])[1] for k, r in sel_t.items()
                                    if k[1] == "lean" and k[2] == task]) for task in an.TASKS]
            cvs = []
            for m in ms:
                ws = [an.priority(mat)[0] for _, _, mat in an.matrices(sel_t, "lean", "main", {m})]
                if len(ws) >= 2:
                    arr = np.array(ws)
                    cvs.append(float((arr.std(axis=0, ddof=1) / arr.mean(axis=0)).mean()))
            try:
                order = float(temp.split(";")[0]) if temp else math.inf
            except ValueError:
                order = math.inf
            rows18.append((order, rt, temp, len(ms) or None,
                           *([None] * 5 if p["main"] is None else p["main"]), *cr_counts, _mean(cvs)))
        for row in sorted(rows18, key=lambda x: (x[0], x[1])):
            t.add(*row[1:])
        tables.append(t)

    # T20 (logic preservation after revision) --------------------------------
    if revs:
        t = Table("T20", "logic_preservation", "Logic preservation after revision",
                  f"Gives, per model, condition and task (revision run tag '{revision_tag}', source tag '{tag}') and "
                  "pooled panel rows, for the chains with at least one revision record: the number of chains, the "
                  "number whose final answers reached CR ≤ 0.10, the mean number of revision calls used (until the "
                  "first consistent result, else all calls), and, comparing the final answers (the best valid "
                  "answers: lowest CR among the first answers and all parsable revisions) with the chain's "
                  "first answers, the mean number of changed answers, the mean Spearman rho of the eigenvector "
                  "weights, the share with the same top criterion, the share with the same full ranking and the "
                  "mean of the largest absolute weight change (percentage points). The last three columns count "
                  "the judge verdicts (logic preserved yes/partly/no) on the revision that gave the final answers "
                  "(none when the first answers stayed best); they are empty when no judge data exist.",
                  ["Model", "Condition", "Task", "Chains revised", "Reached CR ≤ 0.10", "Mean revisions used",
                   "Mean answers changed", "Mean Spearman rho", "Top criterion unchanged", "Ranking unchanged",
                   "Mean max abs weight change (pp)", "Judge yes", "Judge partly", "Judge no"],
                  ["s", "s", "s", "i", "i", "r", "r", "r", "share", "share", "pp", "i", "i", "i"])

        def logic_row(label, c, task, ss):
            ss = [s for s in ss if s["needs_revision"] and s["revisions_used"] > 0]
            if not ss:
                return
            lm = [revise.logic_metrics(task, s["initial_answers"], s["final_answers"]) for s in ss]
            verdicts = [(s["final_record"] or {}).get("judge_verdict") for s in ss]
            has_judge = any(s.get("has_judge") or "judge_model_id" in (s["final_record"] or {}) for s in ss)
            counts = ([sum(1 for v in verdicts if isinstance(v, dict) and v.get("logic_preserved") == x)
                       for x in revise.VERDICTS] if has_judge else [None] * 3)
            t.add(label, L(c), task, len(ss), sum(1 for s in ss if s["reached"]), _mean([s["revisions_used"] for s in ss]),
                  _mean([x["n_answers_changed"] for x in lm]), _mean([x["spearman"] for x in lm]),
                  _mean([float(x["top_criterion_unchanged"]) for x in lm]),
                  _mean([float(x["ranking_unchanged"]) for x in lm]),
                  _mean([x["max_abs_weight_change_pp"] for x in lm]), *counts)
        for m in models:
            for c in rconds:
                for task in an.TASKS:
                    logic_row(m, c, task, [s for s in summ if s["model_key"] == m and s["condition"] == c
                                           and s["task"] == task])
        for c in rconds:
            for task in an.TASKS:
                logic_row(PANEL, c, task, [s for s in summ if s["condition"] == c and s["task"] == task])
        tables.append(t)

    # T21 (ranking agreement with the expert-weight ranking) ---------------
    t = Table("T21", "ranking_agreement", "Ranking agreement with the expert weights",
              "Gives, for TOPSIS, ARAS and EDAS with delivery indicators as rates, Spearman rho and Kendall tau "
              "between the supplier ranks under each condition's LLM panel global weights (and under equal "
              "weights) and the supplier ranks under the expert global weights, and the number of suppliers "
              "whose rank differs; ranks as in T11 and T13.",
              ["Method", "Weights", "Spearman rho", "Kendall tau", "Suppliers with different rank"],
              ["s", "s", "r", "r", "i"])
    ref = an.rank_all(X_rates, ex["global"], benefit)
    for mth in an.METHODS:
        for src, w in agreement_sources:
            if w is None:
                t.add(mth, src, None, None, None)
                continue
            rk, rr = an.rank_all(X_rates, w, benefit)[mth]["ranks"], ref[mth]["ranks"]
            t.add(mth, src, an.spearman(rk, rr), an.kendall_tau(rk, rr), int(np.sum(rk != rr)))
    tables.append(t)

    # T22 (did the replies follow the priority order written in the guided prompt?) --------
    from . import prompts as pr
    t22 = guided_compliance_table(selected, models, expert, pr)
    if t22 is not None:
        tables.append(t22)

    # figures --------------------------------------------------------------
    fig_heat = {}
    for c in conds:
        rows = [(m, model_panel[(m, c)]["global"]) for m in models] + [(PANEL, cond_panel[c]["global"])]
        fig_heat[c] = [(lab, None if w is None else (w - ex["global"]) * 100.0) for lab, w in rows]
    fig_cr = {(m, c): crs.get((m, c, "main"), []) for m in models for c in conds}

    # appendix -------------------------------------------------------------
    llm_rows = []
    for key in sorted(selected):
        m, c, task, rep = key
        mat = np.asarray(selected[key]["matrix"], dtype=float)
        cr = an.priority(mat)[1]
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                llm_rows.append([m, L(c), task, rep, i + 1, j + 1, repr(float(mat[i, j])), fmt(cr, "cr")])
    exp_rows = []
    for task in an.TASKS:
        for name, mat in zip(expert["experts"], expert[task]):
            cr = an.priority(mat)[1]
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    exp_rows.append([name, task, i + 1, j + 1, repr(float(mat[i, j])), fmt(cr, "cr")])
    cot_rows = []
    for key in sorted(selected):
        m, c, task, rep = key
        if c != "cot":
            continue
        r = selected[key]
        ranking = r.get("ranking")
        ranking = " > ".join(map(str, ranking)) if isinstance(ranking, list) else (ranking or "")
        cot_rows.append([m, task, rep, ranking, r.get("reasoning") or ""])
    appendix = {
        "matrices_llm.csv": (["model", "condition", "task", "rep", "i", "j", "value", "CR"], llm_rows,
                             "A01", "Long-format list of every selected LLM judgement matrix entry with the "
                             "matrix CR."),
        "matrices_expert.csv": (["expert", "task", "i", "j", "value", "CR"], exp_rows, "A02",
                                "Long-format list of every expert judgement matrix entry with the matrix CR."),
        "cot_reasoning.csv": (["model", "task", "rep", "ranking", "reasoning"], cot_rows, "A03",
                              "Ranking and reasoning text returned by each model in the CoT condition."),
    }
    meta = {"tag": tag, "n_tagged": len(tagged), "n_selected": len(selected),
            "n_failed": sum(failures.values()), "models": models}
    return tables, {"heat": fig_heat, "cr": fig_cr, "models": models, "conds": conds}, appendix, meta


def guided_compliance_table(selected, models, expert, pr):
    """T22: per model and pooled, the guided replies against the order the guided prompt states (read from
    the prompt file), with direct and CoT replies and the experts' matrices as a reference (they never saw it)."""
    try:
        prompted = {task: pr.guided_prompted_order(task) for task in an.TASKS}
    except (OSError, KeyError, ValueError, StopIteration):
        return None
    shorts = {"main": an.MAIN_SHORT, "delivery": an.DELIVERY_SHORT}
    what = []
    for task in an.TASKS:
        p = prompted[task]
        if p["order"]:
            what.append(f"{task}: the numbered priority order " + " > ".join(shorts[task][i] for i in p["order"])
                        + f" ({p['source']})")
        else:
            what.append(f"{task}: only the first criterion, {shorts[task][p['top']]}, named the primary positive "
                        f"indicator ({p['source']}; no full order, so the order columns are empty)")
    t = Table("T22", "guided_compliance", "Replies and the order stated in the guided prompt",
              "Shows whether the replies followed the order of the criteria stated in the guided prompt, read from the "
              "prompt file - " + "; ".join(what) + ". For each set of replies (selected first answers of the run tag): "
              "the number of replies, the number whose eigenvector-weight ranking equals the prompted order, the number "
              "with the prompted first criterion ranked first, and the mean Spearman rho between the weights and the "
              "prompted order. Rows marked 'reference' are replies that never saw this order (direct and CoT "
              "conditions) and the experts' individual matrices.",
              ["Respondents", "Condition", "Comparison", "Task", "Replies", "Ranking equals prompted order",
               "Prompted first criterion ranked first", "Mean Spearman rho with prompted order"],
              ["s", "s", "s", "s", "i", "i", "i", "r"])

    def stats(task, mats):
        p, n = prompted[task], len(shorts[task])
        ws = [an.priority(m)[0] for m in mats]
        if not ws:
            return [None] * 4
        top = sum(1 for w in ws if int(np.argmax(w)) == p["top"])
        if not p["order"]:
            return [len(ws), None, top, None]
        score = [n - p["order"].index(i) for i in range(n)]
        exact = sum(1 for w in ws if [int(i) for i in np.argsort(-w, kind="mergesort")] == p["order"])
        return [len(ws), exact, top, _mean([an.spearman(w, score) for w in ws])]

    for task in an.TASKS:
        for m in models:
            mats = [x for _, _, x in an.matrices(selected, "guided", task, {m})]
            if mats:
                t.add(m, L("guided"), "prompted", task, *stats(task, mats))
        t.add(PANEL, L("guided"), "prompted", task, *stats(task, [x for _, _, x in an.matrices(selected, "guided", task)]))
        for c in ("lean", "cot"):
            t.add(PANEL, L(c), "reference (order not shown)", task,
                  *stats(task, [x for _, _, x in an.matrices(selected, c, task)]))
        t.add("Experts (individual matrices)", "", "reference (order not shown)", task, *stats(task, expert[task]))
    return t


# ---------------------------------------------------------------- figures

def _save(fig, out: Path, stem: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None, "Creator": None,
                                               "Producer": None})
    fig.savefig(out / f"{stem}.png", dpi=200, metadata={"Software": None})
    return [f"{stem}.pdf", f"{stem}.png"]


def make_figures(fd, out: Path) -> list[tuple[str, str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"font.size": 8, "svg.hashsalt": "ahp", "pdf.fonttype": 42})

    figs = []
    conds, models = fd["conds"], fd["models"]

    # F1 heatmap
    nrows = len(models) + 1
    fig, axes = plt.subplots(1, len(conds), figsize=(3.4 * len(conds), 0.32 * nrows + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    allv = [abs(v) for c in conds for _, d in fd["heat"][c] if d is not None for v in d]
    lim = max(allv) if allv else 1.0
    lim = lim if lim > 0 else 1.0
    cmap = plt.get_cmap("PuOr_r").with_extremes(bad="white")
    im = None
    for ax, c in zip(axes, conds):
        rows = fd["heat"][c]
        data = np.array([d if d is not None else np.full(8, np.nan) for _, d in rows], dtype=float)
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isfinite(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(data[i, j]) > 0.6 * lim else "black")
        ax.set_xticks(range(8), an.GLOBAL_CODES)
        ax.set_yticks(range(len(rows)), [lab for lab, _ in rows])
        ax.axhline(len(rows) - 1.5, color="black", lw=0.8)
        ax.set_title(FULL(c))
        ax.tick_params(length=0)
        if not np.isfinite(data).any():
            ax.text(3.5, (len(rows) - 1) / 2, "no data", ha="center", va="center", color="0.4")
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="Model weight − expert weight (percentage points)")
    figs.append(("F1_global_weight_diff_heatmap", _save(fig, out, "F1_global_weight_diff_heatmap"),
                 "Heatmap of model minus expert global weights in percentage points for the eight global "
                 "indicators, one panel per condition, with the condition panel in the last row."))
    plt.close(fig)

    # F2 CR dot plot
    fig, ax = plt.subplots(figsize=(max(5.0, 0.7 * len(models) + 1.5), 3.2))
    markers, shades = ["o", "s", "^"], ["0.0", "0.4", "0.7"]
    offs = np.linspace(-0.22, 0.22, len(conds))
    for k, c in enumerate(conds):
        xs, ys = [], []
        for i, m in enumerate(models):
            for v in fd["cr"][(m, c)]:
                xs.append(i + offs[k])
                ys.append(v)
        ax.scatter(xs, ys, marker=markers[k % 3], s=22, facecolor=shades[k % 3], edgecolor="black",
                   linewidth=0.5, label=FULL(c), zorder=3)
    ax.axhline(an.CR_THRESHOLD, color="black", ls="--", lw=0.8, zorder=2)
    ax.set_xticks(range(len(models)), models, rotation=35, ha="right")
    ax.set_ylabel("CR (main criteria)")
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
    ax.legend(title="Condition", frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    figs.append(("F2_cr_main_dotplot", _save(fig, out, "F2_cr_main_dotplot"),
                 "Dot plot of the consistency ratio of every main-criteria matrix per model and condition, "
                 "with a dashed line at CR = 0.10."))
    plt.close(fig)
    return figs


# ---------------------------------------------------------------- main

def run_report(input_path, out_dir, tag="main", data_dir=None, figures=True, revision_tag="revision",
               revisions_path=None, latex=False) -> dict:
    if data_dir is None:
        raise ValueError("data_dir is required: the folder with expert_matrices.json and performance_matrix.json")
    data_dir = Path(data_dir)
    records, bad = an.load_records(input_path)
    expert = an.load_expert_data(data_dir / "expert_matrices.json")
    perf = an.load_performance(data_dir / "performance_matrix.json")
    revisions_path = Path(revisions_path) if revisions_path else Path(input_path).parent / "revisions.jsonl"
    rev_records, _ = an.load_records(revisions_path)
    tables, fd, appendix, meta = build(records, tag, expert, perf, rev_records, revision_tag)
    out = Path(out_dir)

    for t in tables:
        write_text(out / "tables" / f"{t.stem}.csv", t.csv_text())
        if latex:
            write_text(out / "tables" / f"{t.stem}.tex", t.tex())
    for fname, (header, rows, _, _) in appendix.items():
        write_csv(out / "appendix" / fname, header, rows)
    from . import appendix_extra as ax  # A04-A08 (CSV; TeX with latex=True)
    for stem, (header, rows, aid, desc) in ax.build(records, rev_records, expert, tag, revision_tag).items():
        write_csv(out / "appendix" / f"{stem}.csv", header, rows)
        if latex:
            write_text(out / "appendix" / f"{stem}.tex", ax.tex(stem, header, rows, desc, ax.TEXT_COLS.get(stem, ())))
        appendix[f"{stem}.csv"] = (header, rows, aid, desc)
    figs = make_figures(fd, out / "figures") if figures else []

    md = [f"# LLM-AHP report (run tag: {tag})", "",
          f"Input: `{Path(input_path).name}`; records with this tag: {meta['n_tagged']}; selected "
          f"matrices: {meta['n_selected']}; failed or unusable records: {meta['n_failed']}; "
          f"unreadable lines skipped: {bad}.", "",
          "Rounding: weights and CR 4 dp, correlations 3 dp, percentage points 2 dp. Empty cells mean "
          "no data. Agreement with the expert weights is a similarity measure, not a quality measure.", ""]
    md += CONDITIONS_MD
    for t in tables:
        md += [f"## {t.tid} {t.title}", "", t.description, "", t.markdown(), ""]
    if figs:
        md += ["## Figures", ""]
        for stem, _, desc in figs:
            md += [f"### {stem}", "", desc, "", f"![{stem}](figures/{stem}.png)", ""]
    md += ["## Appendix files", ""]
    for fname, (_, rows, aid, desc) in appendix.items():
        md += [f"- {aid} `appendix/{fname}` ({len(rows)} rows): {desc}"]
    md.append("")
    write_text(out / "report.md", "\n".join(md))
    return {"tables": [t.stem for t in tables], "figures": [s for s, _, _ in figs],
            "appendix": sorted(appendix), "skipped_lines": bad, **meta}


def main(argv=None) -> int:
    from . import cli
    ap = argparse.ArgumentParser(prog="python -m ahp_bench.report")
    cli.add_common(ap)
    ap.add_argument("--tag", default="main", help="run_tag to analyse (default: main)")
    ap.add_argument("--out", default=None, help="default: <results>/report")
    ap.add_argument("--input", default=None, help="default: <results>/raw/calls.jsonl")
    cli.add_data(ap)
    ap.add_argument("--latex", action="store_true", help="also write TeX versions of the tables and appendix")
    ap.add_argument("--experts", default=None,
                    help="other problems: expert matrices JSON ({'experts': [...], '<task>': [matrices]}); "
                         "without it no expert-comparison tables are written")
    ap.add_argument("--revision-tag", default="revision", help="run_tag of revision records (T16, T17, T20)")
    ap.add_argument("--revisions", default=None,
                    help="revision JSONL, read-only (default: revisions.jsonl next to --input)")
    a = ap.parse_args(argv)
    ctx = cli.context(a)
    if ctx.problem.is_study and a.experts is None:
        if a.input and a.data:  # explicit files; results folder = the parent of the input's folder
            results, data = cli.resolve(a.input).parent.parent, cli.resolve(a.data)
        else:
            results, data = cli.study_paths(a)
        inp = a.input or str(results / "raw" / "calls.jsonl")
        out = a.out or str(results / "report")
        info = run_report(inp, out, a.tag, data, revision_tag=a.revision_tag, revisions_path=a.revisions,
                          latex=a.latex)
    else:
        from . import generic_report
        inp = a.input or str(ctx.raw / "calls.jsonl")
        out = a.out or str(ctx.results / "report")
        info = generic_report.run_report(ctx.problem, inp, out, a.tag, cli.resolve(a.experts) if a.experts else None,
                                         a.revisions, a.revision_tag, ctx.panel["models"], latex=a.latex)
    a.out = out
    print(f"tag={info['tag']}: {info['n_tagged']} records, {info['n_selected']} matrices used, "
          f"{info['n_failed']} failed, {info['skipped_lines']} unreadable lines skipped")
    print(f"wrote {len(info['tables'])} tables, {len(info['figures'])} figures, "
          f"{len(info['appendix'])} appendix files to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
