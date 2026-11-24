"""Report for any problem definition (used by python -m ahp_bench.report --problem PATH).

Tables (CSV; booktabs TeX with --latex) and report.md, computed from the stored records only:
  G01 models and settings              G07 rankings (TOPSIS, ARAS, EDAS) with the group weights
  G02 consistency per matrix set       G08 equal-weight baseline ranking
  G03 local weights per task           G09 one-at-a-time weight sensitivity
  G04 global weights                   G10 revision summary            (revision records only)
  G05 model-level weights per task     G11 weights before / after revision
  G06 expert consistency and similarity (only with --experts)
                                       G12 logic preservation after revision
  G13 guidance compliance            (only when a guided condition was run)
Before the tables, report.md shows the decision hierarchy (Mermaid diagram and text tree), the indicator
table (H01), the performance matrix (H02) and, for problems built from a data set, the data summary of the
provenance block; hierarchy.md and hierarchy.png are written next to report.md.
Ranking tables need a ranking block with a performance matrix in the problem file; expert tables need
an expert matrices file (--experts). Without experts there is no human reference and no similarity
table is written.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from . import analysis as an
from .prompts import condition_column as COL
from .prompts import condition_full as FULL
from .prompts import condition_label as L
from .report import (PANEL, SIM_COLS, SIM_KINDS, Table, _count_ok, _mean, _pp, _reasoning_label, _sim_values,
                     write_text)

REVISION_STEPS = 5


def build(problem, records, tag="main", expert=None, revision_records=None, revision_tag="revision",
          model_cfg=None) -> tuple[list, dict]:
    selected, failures, tagged = an.select_records(records, tag, problem)
    models = sorted({r.get("model_key") for r in tagged if r.get("model_key")}, key=str)
    conds = [c for c in problem.conditions if any(r.get("condition") == c for r in tagged)]
    tasks = list(problem.task_names)
    model_cfg = model_cfg or {}
    ex = an.problem_expert_weights(expert, problem) if expert else None
    cond_panel = {c: an.problem_panel_weights(selected, c, problem, set(models)) for c in conds}
    tables = []

    # G01 -----------------------------------------------------------------
    t = Table("G01", "models_settings", "Models and settings",
              "Lists each model with its developer, OpenRouter identifier, requested temperature, applied reasoning "
              "setting, mean reasoning tokens, providers seen, number of calls, parse failures and total cost (USD) "
              "for the selected run tag.",
              ["Model", "Developer", "OpenRouter id", "Temperature (requested)", "Reasoning (applied)",
               "Mean reasoning tokens", "Providers seen", "Calls", "Parse failures", "Total cost (USD)"],
              ["s", "s", "s", "s", "s", "tok", "s", "i", "i", "usd"])
    by_model = defaultdict(list)
    for r in tagged:
        by_model[r.get("model_key")].append(r)
    for m in models:
        rs, cfg = by_model[m], model_cfg.get(m, {})
        temps = sorted({str((r.get("requested_params") or {}).get("temperature")) for r in rs
                        if (r.get("requested_params") or {}).get("temperature") is not None})
        reas = sorted({_reasoning_label((r.get("applied_params") or {}).get("reasoning")) for r in rs
                       if r.get("applied_params") is not None})
        t.add(m, next((r["developer"] for r in rs if r.get("developer")), cfg.get("developer")),
              next((r["model_id"] for r in rs if r.get("model_id")), cfg.get("id")),
              "; ".join(temps) or None, "; ".join(reas) or None,
              _mean([(r.get("usage") or {}).get("reasoning_tokens") for r in rs]),
              "; ".join(sorted({str(r["provider"]) for r in rs if r.get("provider")})) or None,
              len(rs), sum(1 for r in rs if not r.get("parse_ok")), sum(float(r.get("cost") or 0) for r in rs))
    tables.append(t)

    # G02 -----------------------------------------------------------------
    crs = defaultdict(list)
    for (m, c, task, rep), r in selected.items():
        crs[(m, c, task)].append(an.priority(r["matrix"])[1])
    t = Table("G02", "consistency", "Consistency of the first answers",
              "Gives, per model, condition and task, the number of matrices, the mean and maximum consistency ratio "
              "(CR) and the number of matrices with CR ≤ 0.10 (CR rounded to 4 decimals); panel rows pool all models "
              "and add the CR of the geometric-mean group matrix. Inconsistent matrices are kept in the aggregation.",
              ["Model", "Condition", "Task", "Matrices", "Mean CR", "Max CR", "CR ≤ 0.10", "Group CR"],
              ["s", "s", "s", "i", "cr", "cr", "s", "cr"])
    for m in models:
        for c in conds:
            for task in tasks:
                x = crs.get((m, c, task), [])
                if x:
                    t.add(m, L(c), task, len(x), _mean(x), max(x), _count_ok(x), None)
    for c in conds:
        for task in tasks:
            x = [v for (m, cc, tt), xs in sorted(crs.items(), key=str) if cc == c and tt == task for v in xs]
            if x:
                t.add(PANEL, L(c), task, len(x), _mean(x), max(x), _count_ok(x), cond_panel[c][f"cr_{task}"])
    tables.append(t)

    # G03 / G04 -----------------------------------------------------------
    def weight_table(tid, name, title, what, labels, key):
        cols, kinds = ["Criterion"], ["s"]
        if ex:
            cols, kinds = cols + ["Experts"], kinds + ["w"]
        for c in conds:
            cols += [f"LLM {COL(c)}"] + ([f"Diff {COL(c)} (pp)"] if ex else [])
            kinds += ["w"] + (["pp"] if ex else [])
        tb = Table(tid, name, title, f"Lists each condition's LLM panel group weights (geometric mean of all models' "
                   f"matrices, then eigenvector) for {what}" + (", the expert group weights and the LLM minus expert "
                                                                 "difference in percentage points." if ex else "."),
                   cols, kinds)
        for i, lab in enumerate(labels):
            row = [lab]
            ref = None if not ex or ex[key] is None else ex[key][i]
            if ex:
                row.append(ref)
            for c in conds:
                w = cond_panel[c][key]
                v = None if w is None else w[i]
                row += [v] + ([_pp(v, ref)] if ex else [])
            tb.add(*row)
        return tb

    for k, task in enumerate(tasks):
        tk = problem.task(task)
        what = "the main criteria" if tk.parent is None else f"the sub-criteria of {tk.parent} (local weights)"
        tables.append(weight_table(f"G03{chr(97 + k)}", f"local_weights_{task}", f"Local weights: {task}", what,
                                   tk.short, task))
    tables.append(weight_table("G04", "global_weights", "Global weights",
                               "the global (leaf) criteria: local weight x parent main weight", problem.leaf_short,
                               "global"))

    # G05 -----------------------------------------------------------------
    t = Table("G05", "model_weights", "Model-level weights",
              "Gives, per model, condition and task, the weights from the geometric mean of the model's repetition "
              "matrices, the number of repetitions and the top-ranked criterion. Columns beyond the task's number "
              "of criteria are empty.",
              ["Model", "Condition", "Task", "Reps"] + [f"w{j}" for j in range(1, 10)] + ["Top criterion"],
              ["s", "s", "s", "i"] + ["w"] * 9 + ["s"])
    for m in models:
        for c in conds:
            pw = an.problem_panel_weights(selected, c, problem, {m})
            for task in tasks:
                w = pw[task]
                if w is None:
                    continue
                vals = list(w) + [None] * (9 - len(w))
                t.add(m, L(c), task, pw[f"n_{task}"], *vals, problem.task(task).short[int(np.argmax(w))])
    tables.append(t)

    # G13 -----------------------------------------------------------------
    gt = guidance_compliance_table(problem, selected, models, conds)

    # G06 -----------------------------------------------------------------
    if ex:
        t = Table("G06a", "expert_consistency", "Expert consistency",
                  "Gives the CR of each expert's matrix per task and the CR of the geometric-mean group matrix.",
                  ["Expert"] + [f"CR {task}" for task in tasks], ["s"] + ["cr"] * len(tasks))
        for i, name in enumerate(expert["experts"]):
            t.add(name, *[ex[f"cr_{task}"][i] for task in tasks])
        t.add("Group (geometric mean)", *[ex[f"group_cr_{task}"] for task in tasks])
        t.add("n with CR ≤ 0.10", *[_count_ok(ex[f"cr_{task}"]) for task in tasks])
        tables.append(t)
        t = Table("G06b", "similarity_to_experts", "Similarity to expert weights",
                  "Gives Spearman rho, Kendall tau, RMSE and mean absolute difference (percentage points) between "
                  "each model's (and each condition panel's) group weights and the expert group weights, for the main "
                  "criteria and the global weights.",
                  ["Model", "Condition"] + [f"{s} main" for s in SIM_COLS] + [f"{s} global" for s in SIM_COLS],
                  ["s", "s"] + SIM_KINDS * 2)
        for m in models:
            for c in conds:
                pw = an.problem_panel_weights(selected, c, problem, {m})
                t.add(m, L(c), *_sim_values(pw["main"], ex["main"]), *_sim_values(pw["global"], ex["global"]))
        for c in conds:
            pw = cond_panel[c]
            t.add(PANEL, L(c), *_sim_values(pw["main"], ex["main"]), *_sim_values(pw["global"], ex["global"]))
        tables.append(t)

    if gt is not None:
        tables.append(gt)

    # G07-G09 ranking -----------------------------------------------------
    if problem.has_ranking:
        perf = problem.performance()
        X, benefit, alts = perf["X"], perf["benefit"], perf["alternatives"]
        note = (problem.ranking.get("transform") or {}).get("note")
        rank_cols, rank_kinds = [], []
        for mth in an.METHODS:
            rank_cols += [f"{mth} score", f"{mth} rank"]
            rank_kinds += ["score", "rank"]

        def rank_rows(tb, label, w):
            res = an.rank_all(X, w, benefit) if w is not None else None
            for i, alt in enumerate(alts):
                vals = []
                for mth in an.METHODS:
                    vals += [None, None] if res is None else [res[mth]["scores"][i], res[mth]["ranks"][i]]
                tb.add(label, alt, *vals)

        t = Table("G07", "rankings", "Rankings with the group weights",
                  "Gives TOPSIS, ARAS and EDAS scores and ranks of the alternatives under each condition's LLM panel "
                  "global weights" + (" and the expert global weights" if ex else "") + " (indicator directions from "
                  "the problem file" + (f"; transform: {note}" if note else "") + ").",
                  ["Weights", "Alternative"] + rank_cols, ["s", "s"] + rank_kinds)
        if ex and ex["global"] is not None:
            rank_rows(t, "Experts", problem.indicator_weights(ex["global"]))
        for c in conds:
            g = cond_panel[c]["global"]
            rank_rows(t, f"LLM panel {FULL(c)}", None if g is None else problem.indicator_weights(g))
        tables.append(t)
        n_ind = X.shape[1]
        t = Table("G08", "equal_weight_baseline", "Equal-weight baseline",
                  f"Gives TOPSIS, ARAS and EDAS scores and ranks with all {n_ind} indicator weights equal to 1/{n_ind}.",
                  ["Weights", "Alternative"] + rank_cols, ["s", "s"] + rank_kinds)
        rank_rows(t, f"Equal (1/{n_ind})", np.full(n_ind, 1.0 / n_ind))
        tables.append(t)
        initials = an.supplier_initials(alts)
        sens = an.one_at_a_time_sensitivity(X, benefit, alts)
        steps = list(an.SENSITIVITY_STEPS)
        t = Table("G09", "weight_sensitivity", "One-at-a-time weight sensitivity",
                  "For each indicator in turn, sets its weight to each tested level (" + ", ".join(f"{s:g}" for s in steps)
                  + ") while the other indicators share the remainder equally; gives the equal-weight ranking, the "
                  "ranking at the lowest and highest level (initials, best first: "
                  + ", ".join(f"{i} = {a}" for i, a in zip(initials, alts))
                  + ") and the lowest tested weight at which the top alternative differs from the equal-weight top "
                  "alternative (empty if never).",
                  ["Indicator", "Method", "Equal-weight ranking", f"Ranking at {steps[0]:g}", f"Ranking at {steps[-1]:g}",
                   "Top changes at"], ["s", "s", "s", "s", "s", "s"])
        for row in sens:
            ch = row["top_change_at"]
            t.add(perf["codes"][row["indicator"]], row["method"], row["base_ranking"], row["rankings"][0],
                  row["rankings"][-1], None if ch is None else f"{ch:g}")
        tables.append(t)

    # G10-G12 revision ----------------------------------------------------
    revs = [r for r in (revision_records or []) if r.get("run_tag") == revision_tag and r.get("source_tag") == tag]
    if revs:
        from . import revise
        rconds = [c for c in revise.REVISION_CONDITIONS if any(r.get("condition") == c for r in revs)]
        rmodels = {r.get("model_key") for r in revs}
        summ = [s for s in revise.chain_summary(records, revs, tag, revision_tag, rconds, problem=problem)
                if s["model_key"] in rmodels]
        ks = list(range(1, REVISION_STEPS + 1))
        t = Table("G10", "revision_summary", "Revision summary",
                  f"Per model, condition and task (revision tag '{revision_tag}'): source matrices, matrices with "
                  "CR > 0.10, the cumulative number that reached CR ≤ 0.10 after at most 1-5 revisions, the number not "
                  "reached and the mean initial and final CR of the matrices needing revision (final = best valid "
                  "answers of the chain).",
                  ["Model", "Condition", "Task", "Matrices", "Needing revision"] + [f"Reached after ≤{k}" for k in ks]
                  + ["Not reached", "Mean initial CR", "Mean final CR"],
                  ["s", "s", "s", "i", "i"] + ["i"] * len(ks) + ["i", "cr", "cr"])

        def rev_row(label, c, task, ss):
            need = [s for s in ss if s["needs_revision"]]
            t.add(label, L(c), task, len(ss), len(need),
                  *[sum(1 for s in need if s["reached_at"] is not None and s["reached_at"] <= k) for k in ks],
                  sum(1 for s in need if not s["reached"]), _mean([s["initial_CR"] for s in need]),
                  _mean([s["final_CR"] for s in need]))
        for m in models:
            for c in rconds:
                for task in tasks:
                    ss = [s for s in summ if s["model_key"] == m and s["condition"] == c and s["task"] == task]
                    if ss:
                        rev_row(m, c, task, ss)
        for c in rconds:
            for task in tasks:
                ss = [s for s in summ if s["condition"] == c and s["task"] == task]
                if ss:
                    rev_row(PANEL, c, task, ss)
        tables.append(t)

        final_sel = dict(selected)
        for s in summ:
            key = (s["model_key"], s["condition"], s["task"], s["rep"])
            final_sel[key] = {**selected[key], "matrix": s["final_matrix"]}
        t = Table("G11", "weights_before_after_revision", "Global weights before and after revision",
                  "Gives, per condition, the LLM panel global weights from the first answers and from the final "
                  "answers after revision (matrices that needed no revision unchanged)"
                  + (", next to the expert global weights." if ex else "."),
                  ["Condition", "Answers"] + problem.leaf_short, ["s", "s"] + ["w"] * len(problem.leaves))
        if ex and ex["global"] is not None:
            t.add("Experts", "", *ex["global"])
        for c in rconds:
            for label, sel in (("first", selected), ("final", final_sel)):
                g = an.problem_panel_weights(sel, c, problem, set(models))["global"]
                t.add(L(c), label, *([None] * len(problem.leaves) if g is None else g))
        tables.append(t)

        t = Table("G12", "logic_preservation", "Logic preservation after revision",
                  "For chains with at least one revision: chains, chains reaching CR ≤ 0.10, mean revisions used, mean "
                  "answers changed, mean Spearman rho of the weights (final vs first answers), share with the same top "
                  "criterion, share with the same full ranking, mean largest absolute weight change (pp) and the judge "
                  "verdicts on the revision that gave the final answers (empty without judge data).",
                  ["Model", "Condition", "Task", "Chains revised", "Reached CR ≤ 0.10", "Mean revisions used",
                   "Mean answers changed", "Mean Spearman rho", "Top criterion unchanged", "Ranking unchanged",
                   "Mean max abs weight change (pp)", "Judge yes", "Judge partly", "Judge no"],
                  ["s", "s", "s", "i", "i", "r", "r", "r", "share", "share", "pp", "i", "i", "i"])

        def logic_row(label, c, task, ss):
            ss = [s for s in ss if s["needs_revision"] and s["revisions_used"] > 0]
            if not ss:
                return
            lm = [revise.logic_metrics(task, s["initial_answers"], s["final_answers"], problem) for s in ss]
            verdicts = [(s["final_record"] or {}).get("judge_verdict") for s in ss]
            has_judge = any(s.get("has_judge") for s in ss)
            counts = ([sum(1 for v in verdicts if isinstance(v, dict) and v.get("logic_preserved") == x)
                       for x in revise.VERDICTS] if has_judge else [None] * 3)
            t.add(label, L(c), task, len(ss), sum(1 for s in ss if s["reached"]), _mean([s["revisions_used"] for s in ss]),
                  _mean([x["n_answers_changed"] for x in lm]), _mean([x["spearman"] for x in lm]),
                  _mean([float(x["top_criterion_unchanged"]) for x in lm]),
                  _mean([float(x["ranking_unchanged"]) for x in lm]),
                  _mean([x["max_abs_weight_change_pp"] for x in lm]), *counts)
        for m in models:
            for c in rconds:
                for task in tasks:
                    logic_row(m, c, task, [s for s in summ if s["model_key"] == m and s["condition"] == c
                                           and s["task"] == task])
        for c in rconds:
            for task in tasks:
                logic_row(PANEL, c, task, [s for s in summ if s["condition"] == c and s["task"] == task])
        tables.append(t)

    meta = {"tag": tag, "n_tagged": len(tagged), "n_selected": len(selected), "n_failed": sum(failures.values()),
            "models": models, "conditions": conds, "experts": bool(ex), "ranking": problem.has_ranking,
            "revision_records": len(revs)}
    return tables, meta


GUIDED_KEYS = ("user_guided", "user_guided_cot")


def guidance_compliance_table(problem, selected, models, conds):
    """G13 (generalised T22 of the study): replies under the guided condition against the priority order the
    guidance block states, with the other conditions' replies as a reference (they never saw the order)."""
    gconds = [c for c in conds if c in GUIDED_KEYS]
    order = (problem.guidance or {}).get("order") or {}
    tasks = [t for t in problem.task_names if order.get(t)]
    if not gconds or not tasks:
        return None
    t = Table("G13", "guidance_compliance", "Replies and the priority order stated in the guided prompt",
              "Shows whether the replies followed the priority order of the guidance block (" + "; ".join(
                  f"{task}: " + " > ".join(problem.task(task).short[problem.criteria(task).index(c)] for c in order[task])
                  for task in tasks) + "). For each set of replies (selected first answers of the run tag): the number "
              "of replies, the share whose eigenvector-weight ranking equals the prompted order, the share with the "
              "prompted first criterion ranked first, and the mean Spearman rho between the weights and the prompted "
              "order. Rows marked 'reference' are replies that never saw this order.",
              ["Respondents", "Condition", "Comparison", "Task", "Replies", "Share ranking equals prompted order",
               "Share prompted first criterion ranked first", "Mean Spearman rho with prompted order"],
              ["s", "s", "s", "s", "i", "share", "share", "r"])

    def stats(task, mats):
        names = problem.criteria(task)
        idx = [names.index(c) for c in order[task]]
        n = len(names)
        ws = [an.priority(m)[0] for m in mats]
        if not ws:
            return [0, None, None, None]
        score = [n - idx.index(i) for i in range(n)]
        exact = sum(1 for w in ws if [int(i) for i in np.argsort(-w, kind="mergesort")] == idx)
        top = sum(1 for w in ws if int(np.argmax(w)) == idx[0])
        return [len(ws), exact / len(ws), top / len(ws), _mean([an.spearman(w, score) for w in ws])]

    for task in tasks:
        for c in gconds:
            for m in models:
                mats = [x for _, _, x in an.matrices(selected, c, task, {m})]
                if mats:
                    t.add(m, L(c), "prompted", task, *stats(task, mats))
            t.add(PANEL, L(c), "prompted", task, *stats(task, [x for _, _, x in an.matrices(selected, c, task)]))
        for c in conds:
            if c not in GUIDED_KEYS:
                mats = [x for _, _, x in an.matrices(selected, c, task)]
                if mats:
                    t.add(PANEL, L(c), "reference (order not shown)", task, *stats(task, mats))
    return t


def problem_tables(problem) -> list:
    """H01 indicator table and H02 performance matrix (problems with a ranking block)."""
    from .hierarchy import indicator_rows
    if not problem.ranking:
        return []
    t = Table("H01", "indicators", "Criteria and indicators",
              "Lists each leaf criterion with its main criterion, indicator, unit, direction (benefit = higher is "
              "better, cost = lower is better) and, for problems built from a data set, the source column(s), the "
              "aggregation per alternative and the row filter.",
              ["Main criterion", "Criterion", "Indicator", "Name", "Unit", "Direction", "Source", "Aggregation",
               "Filter"], ["s"] * 9)
    for r in indicator_rows(problem):
        t.add(r["main"] if r["level"] == "sub" else "", r["criterion"], r["code"], r["indicator"], r["unit"],
              r["direction"], r["source"], r["aggregation"], r["filter"])
    out = [t]
    if problem.has_ranking:
        perf = problem.performance(transformed=False)
        note = (problem.ranking.get("transform") or {}).get("note")
        t = Table("H02", "performance_matrix", "Performance matrix",
                  "Gives the indicator values of each alternative as stored in the problem file (before any transform"
                  + (f"; transform: {note}" if note else "") + ").",
                  ["Alternative"] + perf["codes"], ["s"] + ["g"] * len(perf["codes"]))
        for i, alt in enumerate(perf["alternatives"]):
            t.add(alt, *[f"{float(v):g}" for v in perf["X"][i]])
        out.append(t)
    return out


def _data_md(problem) -> list:
    prov = problem.raw.get("provenance") if isinstance(problem.raw, dict) else None
    if not isinstance(prov, dict):
        return []
    ds = prov.get("data_summary") or {}
    md = ["## Data set", "",
          f"Built with {prov.get('built_with')} from `{prov.get('data_file')}` (sha256 {prov.get('data_sha256')}): "
          f"{prov.get('data_rows')} rows, {ds.get('columns')} columns ({ds.get('encoding')}, delimiter "
          f"`{ds.get('delimiter')}`); {prov.get('rows_used')} rows used; alternative column `{ds.get('alternative_column')}`. "
          f"Criteria spec `{prov.get('spec_file')}` (sha256 {prov.get('spec_sha256')}); criteria proposed by "
          f"{prov.get('criteria_proposed_by')}; confirmed by the user: "
          f"{'yes' if prov.get('criteria_confirmed_by_user') else 'no'}.", ""]
    if not prov.get("criteria_confirmed_by_user"):
        md += ["**The criteria were not confirmed by the user before weighting.**", ""]
    rpa = ds.get("rows_per_alternative") or {}
    if rpa:
        md += ["| Alternative | Rows |", "| --- | ---: |"] + [f"| {k} | {v} |" for k, v in rpa.items()] + [""]
    miss = ds.get("missing_cells_in_used_columns") or {}
    if miss:
        md += ["Missing cells in the columns used: " + ", ".join(f"{k} {v}" for k, v in miss.items()) + ".", ""]
    return md


def _guidance_md(problem, conds) -> list:
    from . import guidance as gd
    md = []
    if any(c in gd.DESCRIBED + GUIDED_KEYS for c in conds) and problem.descriptions:
        d = problem.descriptions
        md += ["## Criterion descriptions shown in the described and guided prompts", "",
               f"Drafted by {d.get('drafted_by')}; confirmed by the user: {'yes' if d.get('confirmed_by_user') else 'no'}.",
               ""] + [f"- {c}: {d['criteria'][c]}" for c in problem.all_criteria] + [""]
    if any(c in GUIDED_KEYS for c in conds) and problem.guidance:
        g = problem.guidance
        md += ["## Guidance", "", gd.GUIDED_NOTE, "",
               f"Source of the priority order and hints: {g.get('source')}; confirmed by the user: "
               f"{'yes' if g.get('confirmed_by_user') else 'no'}."]
        md += [f"Order ({t}): " + " > ".join(o) + "." for t, o in (g.get("order") or {}).items()]
        md += [f"Hint ({t}): {h}" for t, hs in (g.get("hints") or {}).items() for h in hs]
        w = gd.guidance_warning(problem)
        if w:
            md += ["", f"**WARNING: {w}**"]
        md.append("")
    return md


def run_report(problem, input_path, out_dir, tag="main", experts_path=None, revisions_path=None,
               revision_tag="revision", model_cfg=None, latex=False) -> dict:
    records, bad = an.load_records(input_path)
    revisions_path = Path(revisions_path) if revisions_path else Path(input_path).parent / "revisions.jsonl"
    rev_records, _ = an.load_records(revisions_path)
    expert = an.load_expert_matrices(experts_path, problem) if experts_path else None
    tables, meta = build(problem, records, tag, expert, rev_records, revision_tag, model_cfg)
    out = Path(out_dir)
    from . import hierarchy
    htabs = problem_tables(problem)
    hier = hierarchy.write(problem, out)
    tables = htabs + tables
    for t in tables:
        write_text(out / "tables" / f"{t.stem}.csv", t.csv_text())
        if latex:
            write_text(out / "tables" / f"{t.stem}.tex", t.tex())
    md = [f"# LLM-AHP report: {problem.title} (run tag: {tag})", "",
          f"Problem file: `{problem.path.name if problem.path else problem.name}` (sha256 {problem.sha256}). "
          f"Input: `{Path(input_path).name}`; records with this tag: {meta['n_tagged']}; selected matrices: "
          f"{meta['n_selected']}; failed or unusable records: {meta['n_failed']}; unreadable lines skipped: {bad}.", "",
          "Rounding: weights and CR 4 dp, correlations 3 dp, percentage points 2 dp. Empty cells mean no data.", ""]
    if not meta["experts"]:
        md += ["No expert matrices were given (--experts): there is no human reference, so no similarity to "
               "experts is reported.", ""]
    if not meta["ranking"]:
        md += ["The problem file has no performance matrix: ranking tables are skipped.", ""]
    md += ["## Decision hierarchy", "", hierarchy.mermaid(problem), "", "```", hierarchy.text_tree(problem), "```", ""]
    if hier["png"]:
        md += ["![hierarchy](hierarchy.png)", ""]
    md += _data_md(problem)
    md += _guidance_md(problem, meta["conditions"])
    for t in tables:
        md += [f"## {t.tid} {t.title}", "", t.description, "", t.markdown(), ""]
    write_text(out / "report.md", "\n".join(md))
    return {"tables": [t.stem for t in tables], "figures": [], "appendix": [], "skipped_lines": bad, **meta}
