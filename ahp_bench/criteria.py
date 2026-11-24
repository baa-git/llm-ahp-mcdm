"""Stages 1-2: criteria specification, performance matrix and problem file.

A criteria spec (JSON; keys starting with "_" are comments) says how indicators are computed from a data set
and how they sit in the AHP hierarchy:

    {"spec_version": 1,
     "proposed_by": "human" | "host agent" | "<model id>",
     "problem": {"name": "...", "title": "...", "goal": "<noun phrase>", "role": "<respondent line>"},
     "alternative_column": "<column>",
     "alternatives": ["...", ...],              # optional: subset / order (default: all values, sorted)
     "filter": {...} | [{...}, ...],            # optional row filter for every indicator
     "indicators": [{"id": "OTD", "label": "...", "aggregation": "share", "column": "...", "value": "...",
                     "filter": {...}, "direction": "benefit" | "cost", "unit": "...", "rationale": "...",
                     "scale": 1, "offset": 0}, ...],
     "hierarchy": {"criteria": [{"name": "...", "short": "...", "indicator": "<id>"},
                                {"name": "...", "sub_criteria": [{"name": "...", "indicator": "<id>"}, ...]}]}}

Aggregations per alternative: mean, sum, median, min, max (numeric "column"); count (rows, or non-empty
cells of "column"); share (rows whose "column" equals "value" - or one of a list of values - divided by
rows with a non-empty "column"); ratio (sum of "numerator" / sum of "denominator" over rows where both are
present); mean_days_between (mean days from date "start" to date "end"). A filter condition is
{"column": C, OP: V} with OP one of equals, not_equals, in, not_in, gt, ge, lt, le; a list means AND.
The result is multiplied by "scale" (e.g. 100 for percent) and "offset" is added (only to keep values
> 0, which ARAS and EDAS need; recorded in the problem file).

Hierarchy: goal -> 3-9 main criteria -> optional 3-9 sub-criteria per main criterion. Every leaf criterion
maps to exactly one indicator and every indicator is used by exactly one leaf (as in the published study);
deeper hierarchies and several indicators per leaf are not supported.

ahp-bench criteria template --eda DIR/eda.json --out spec.json      draft from the EDA suggestions (review it)
ahp-bench criteria validate --spec spec.json --data FILE.csv         check the spec against the data
ahp-bench criteria build --spec spec.json --data FILE.csv --out problems/NAME.json [--confirmed]
ahp-bench criteria propose --eda DIR/eda.json --data FILE.csv --out spec.json --model ID --budget USD [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from . import problem as pb
from .tabular import TableError, date_ordinal, is_missing, number, read_table

SPEC_VERSION = 1
AGGREGATIONS = {
    "mean": ("column",), "sum": ("column",), "median": ("column",), "min": ("column",), "max": ("column",),
    "count": (), "share": ("column", "value"), "ratio": ("numerator", "denominator"),
    "mean_days_between": ("start", "end"),
}
FILTER_OPS = ("equals", "not_equals", "in", "not_in", "gt", "ge", "lt", "le")
WEIGHT_KEYS = {"weight", "weights", "importance", "priority", "rank", "ranking", "order", "importance_order"}
PROMPT_ID = "criteria_proposal_v1"
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,39}$")


class SpecError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def strip_comments(obj):
    if isinstance(obj, dict):
        return {k: strip_comments(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_comments(v) for v in obj]
    return obj


def load_spec(path) -> tuple[dict, str]:
    data = Path(path).read_bytes()
    try:
        obj = json.loads(data.decode("utf-8-sig"))
    except json.JSONDecodeError as e:
        raise SpecError([f"{Path(path).name}: invalid JSON ({e.msg}, line {e.lineno})"]) from None
    return obj, hashlib.sha256(data).hexdigest()


def _review(value) -> bool:
    return isinstance(value, str) and value.strip().upper().startswith("REVIEW")


def _conditions(f, where) -> tuple[list, list]:
    if f is None:
        return [], []
    conds = f if isinstance(f, list) else [f]
    errs = []
    for k, c in enumerate(conds):
        w = f"{where}[{k}]" if isinstance(f, list) else where
        if not isinstance(c, dict) or not isinstance(c.get("column"), str):
            errs.append(f"{w}: a filter condition needs a 'column'")
            continue
        ops = [o for o in FILTER_OPS if o in c]
        if len(ops) != 1 or set(c) - {"column", *FILTER_OPS}:
            errs.append(f"{w}: exactly one of {', '.join(FILTER_OPS)} besides 'column'")
            continue
        v = c[ops[0]]
        if ops[0] in ("in", "not_in") and not isinstance(v, list):
            errs.append(f"{w}: '{ops[0]}' needs a list")
        if ops[0] in ("gt", "ge", "lt", "le") and (not isinstance(v, (int, float)) or isinstance(v, bool)):
            errs.append(f"{w}: '{ops[0]}' needs a number")
    return conds, errs


def check_structure(spec: dict) -> list:
    """Structural errors of a spec (no data needed). Empty list = valid."""
    s = strip_comments(spec)
    errs = []
    if not isinstance(s, dict):
        return ["the spec must be a JSON object"]
    if s.get("spec_version", SPEC_VERSION) != SPEC_VERSION:
        errs.append(f"spec_version must be {SPEC_VERSION}")
    pr = s.get("proposed_by")
    if not isinstance(pr, str) or not pr.strip() or _review(pr):
        errs.append("proposed_by: 'human', 'host agent' or a model id")
    p = s.get("problem")
    if not isinstance(p, dict):
        errs.append("problem: object with name, title, goal and role")
        p = {}
    for k in ("name", "goal", "role"):
        if not isinstance(p.get(k), str) or not p[k].strip():
            errs.append(f"problem.{k}: required (non-empty string)")
        elif _review(p[k]):
            errs.append(f"problem.{k}: still a REVIEW placeholder")
    if isinstance(p.get("name"), str) and not pb._TASK_NAME.match(p["name"]):
        errs.append("problem.name: lower-case letters, digits or '_' (max 40), starting with a letter")
    if not isinstance(s.get("alternative_column"), str) or not s["alternative_column"].strip():
        errs.append("alternative_column: required")
    alts = s.get("alternatives")
    if alts is not None and (not isinstance(alts, list) or len(alts) < 2 or len(set(map(str, alts))) != len(alts)):
        errs.append("alternatives: optional list of at least 2 unique values")
    errs += _conditions(s.get("filter"), "filter")[1]
    inds = s.get("indicators")
    ids = []
    if not isinstance(inds, list) or not inds:
        errs.append("indicators: non-empty list")
        inds = []
    for k, ind in enumerate(inds):
        w = f"indicators[{k}]"
        if not isinstance(ind, dict):
            errs.append(f"{w}: expected an object")
            continue
        iid = ind.get("id")
        if not isinstance(iid, str) or not _ID.match(iid):
            errs.append(f"{w}: 'id' must be a short identifier (letters, digits, '_', starting with a letter)")
        else:
            w = f"indicator {iid}"
            ids.append(iid)
        agg = ind.get("aggregation")
        if agg not in AGGREGATIONS:
            errs.append(f"{w}: aggregation must be one of {', '.join(AGGREGATIONS)}")
        else:
            for key in AGGREGATIONS[agg]:
                if key not in ind or (key != "value" and not isinstance(ind[key], str)):
                    errs.append(f"{w}: aggregation '{agg}' needs '{key}'")
        if ind.get("direction") not in pb.DIRECTIONS:
            errs.append(f"{w}: direction must be 'benefit' or 'cost'"
                        + (" (still a REVIEW placeholder)" if _review(ind.get("direction")) else ""))
        for key in ("label", "unit"):
            if key in ind and not isinstance(ind[key], str):
                errs.append(f"{w}: '{key}' must be a string")
        for key in ("scale", "offset"):
            if key in ind and (not isinstance(ind[key], (int, float)) or isinstance(ind[key], bool)):
                errs.append(f"{w}: '{key}' must be a number")
        if ind.get("scale") == 0:
            errs.append(f"{w}: 'scale' must not be 0")
        bad = sorted(WEIGHT_KEYS & set(ind))
        if bad:
            errs.append(f"{w}: weights or importance are not part of a criteria spec ({', '.join(bad)})")
        errs += _conditions(ind.get("filter"), f"{w}.filter")[1]
    if len(set(ids)) != len(ids):
        errs.append("indicators: duplicate ids")
    h = s.get("hierarchy")
    if not isinstance(h, dict) or not isinstance(h.get("criteria"), list):
        errs.append("hierarchy.criteria: list of main criteria")
        return errs
    leaves, names = [], []
    mains = h["criteria"]
    if not pb.MIN_N <= len(mains) <= pb.MAX_N:
        errs.append(f"hierarchy: {len(mains)} main criteria; an AHP matrix needs {pb.MIN_N}-{pb.MAX_N}")
    for k, c in enumerate(mains):
        w = f"hierarchy.criteria[{k}]"
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) or not c["name"].strip():
            errs.append(f"{w}: needs a 'name'")
            continue
        names.append(c["name"].strip())
        bad = sorted(WEIGHT_KEYS & set(c))
        if bad:
            errs.append(f"{w}: weights or importance are not part of a criteria spec ({', '.join(bad)})")
        subs = c.get("sub_criteria")
        if subs:
            if "indicator" in c:
                errs.append(f"{w}: a main criterion with sub_criteria maps no indicator itself")
            if not isinstance(subs, list) or not pb.MIN_N <= len(subs) <= pb.MAX_N:
                errs.append(f"{w}: {len(subs) if isinstance(subs, list) else 0} sub-criteria; a group needs "
                            f"{pb.MIN_N}-{pb.MAX_N} (fewer: make them main criteria)")
                subs = subs if isinstance(subs, list) else []
            for j, sc in enumerate(subs):
                if not isinstance(sc, dict) or not isinstance(sc.get("name"), str) or not isinstance(sc.get("indicator"), str):
                    errs.append(f"{w}.sub_criteria[{j}]: needs 'name' and 'indicator'")
                    continue
                if WEIGHT_KEYS & set(sc):
                    errs.append(f"{w}.sub_criteria[{j}]: weights or importance are not part of a criteria spec")
                names.append(sc["name"].strip())
                leaves.append((sc["name"].strip(), sc["indicator"]))
        elif isinstance(c.get("indicator"), str):
            leaves.append((c["name"].strip(), c["indicator"]))
        else:
            errs.append(f"{w}: a leaf criterion needs an 'indicator' (or 3-9 'sub_criteria')")
    if len({n.lower() for n in names}) != len(names):
        errs.append("hierarchy: criterion names must be unique")
    used = [i for _, i in leaves]
    for leaf, i in leaves:
        if i not in ids:
            errs.append(f"hierarchy: leaf {leaf!r} maps to unknown indicator {i!r}")
    dup = sorted({i for i in used if used.count(i) > 1})
    if dup:
        errs.append(f"hierarchy: indicators used by more than one leaf: {', '.join(dup)}")
    unused = [i for i in ids if i not in used]
    if unused:
        errs.append(f"indicators not used by any leaf criterion: {', '.join(unused)} (remove them or add leaves)")
    return errs


# ---------------------------------------------------------------- computing indicators

def _match(cell, cond, table, col):
    op = next(o for o in FILTER_OPS if o in cond)
    v = cond[op]
    if op in ("gt", "ge", "lt", "le"):
        x = number(cell, table.decimal(col))
        if x is None:
            return False
        return {"gt": x > v, "ge": x >= v, "lt": x < v, "le": x <= v}[op]
    vals = v if isinstance(v, list) else [v]
    hit = any(_eq(cell, x, table, col) for x in vals)
    return hit if op in ("equals", "in") else not hit


def _eq(cell, value, table, col) -> bool:
    if is_missing(cell):
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        x = number(cell, table.decimal(col))
        return x is not None and abs(x - value) < 1e-12
    return str(cell).strip() == str(value).strip()


def _row_filter(conds, table):
    idx = [(c, table.index(c["column"]), c["column"]) for c in conds]
    return lambda row: all(_match(row[i], c, table, col) for c, i, col in idx)


def aggregate(ind: dict, rows: list, table):
    agg = ind["aggregation"]
    if agg in ("mean", "sum", "median", "min", "max"):
        i, dec = table.index(ind["column"]), table.decimal(ind["column"])
        xs = [x for x in (number(r[i], dec) for r in rows) if x is not None]
        if not xs:
            return None
        return {"mean": statistics.fmean, "sum": sum, "median": statistics.median, "min": min, "max": max}[agg](xs)
    if agg == "count":
        if ind.get("column"):
            i = table.index(ind["column"])
            return float(sum(1 for r in rows if not is_missing(r[i])))
        return float(len(rows))
    if agg == "share":
        i, col = table.index(ind["column"]), ind["column"]
        present = [r for r in rows if not is_missing(r[i])]
        if not present:
            return None
        vals = ind["value"] if isinstance(ind["value"], list) else [ind["value"]]
        return sum(1 for r in present if any(_eq(r[i], v, table, col) for v in vals)) / len(present)
    if agg == "ratio":
        a, b = table.index(ind["numerator"]), table.index(ind["denominator"])
        da, db = table.decimal(ind["numerator"]), table.decimal(ind["denominator"])
        pairs = [(number(r[a], da), number(r[b], db)) for r in rows]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        den = sum(y for _, y in pairs)
        return None if not pairs or den == 0 else sum(x for x, _ in pairs) / den
    if agg == "mean_days_between":
        a, b = table.index(ind["start"]), table.index(ind["end"])
        d = [(date_ordinal(r[b]), date_ordinal(r[a])) for r in rows]
        d = [e - s for e, s in d if e is not None and s is not None]
        return statistics.fmean(d) if d else None
    raise SpecError([f"unknown aggregation {agg!r}"])


def _data_errors(s: dict, table) -> list:
    errs = []
    cols = set(table.header)

    def need(col, w, numeric=False, date=False):
        if col not in cols:
            errs.append(f"{w}: column {col!r} not in the data (columns: {', '.join(table.header)})")
            return
        vals = [v for v in table.column(col) if not is_missing(v)]
        if not vals:
            errs.append(f"{w}: column {col!r} has no values")
        elif numeric and sum(number(v, table.decimal(col)) is not None for v in vals) < 0.95 * len(vals):
            errs.append(f"{w}: column {col!r} is not numeric")
        elif date and sum(date_ordinal(v) is not None for v in vals) < 0.95 * len(vals):
            errs.append(f"{w}: column {col!r} does not hold dates (yyyy-mm-dd, dd.mm.yyyy or dd/mm/yyyy)")

    need(s["alternative_column"], "alternative_column")
    for c in _conditions(s.get("filter"), "filter")[0]:
        need(c["column"], "filter")
    for ind in s["indicators"]:
        w, agg = f"indicator {ind['id']}", ind["aggregation"]
        for c in _conditions(ind.get("filter"), "")[0]:
            need(c["column"], f"{w}.filter")
        if agg in ("mean", "sum", "median", "min", "max"):
            need(ind["column"], w, numeric=True)
        elif agg == "count" and ind.get("column"):
            need(ind["column"], w)
        elif agg == "share":
            need(ind["column"], w)
            if ind["column"] in cols:
                vals = ind["value"] if isinstance(ind["value"], list) else [ind["value"]]
                for v in vals:
                    if not any(_eq(c, v, table, ind["column"]) for c in table.column(ind["column"])):
                        errs.append(f"{w}: value {v!r} never occurs in column {ind['column']!r}")
        elif agg == "ratio":
            need(ind["numerator"], w, numeric=True)
            need(ind["denominator"], w, numeric=True)
        elif agg == "mean_days_between":
            need(ind["start"], w, date=True)
            need(ind["end"], w, date=True)
    return errs


def compute(spec: dict, table) -> dict:
    """Performance matrix from a valid spec and a table. Raises SpecError with every problem found."""
    s = strip_comments(spec)
    errs = check_structure(s)
    if errs:
        raise SpecError(errs)
    errs = _data_errors(s, table)
    if errs:
        raise SpecError(errs)
    ai = table.index(s["alternative_column"])
    gfilter = _row_filter(_conditions(s.get("filter"), "")[0], table)
    rows = [r for r in table.rows if not is_missing(r[ai]) and gfilter(r)]
    present = sorted({r[ai] for r in rows})
    if s.get("alternatives"):
        alts = [str(a) for a in s["alternatives"]]
        missing = [a for a in alts if a not in present]
        if missing:
            raise SpecError([f"alternatives not found in column {s['alternative_column']!r}: {', '.join(missing)}"])
    else:
        alts = present
    if len(alts) < 2:
        raise SpecError([f"column {s['alternative_column']!r} has {len(alts)} alternatives after filtering; at least 2 "
                         "are needed"])
    by_alt = {a: [r for r in rows if r[ai] == a] for a in alts}
    matrix = [[None] * len(s["indicators"]) for _ in alts]
    rows_used = {}
    for j, ind in enumerate(s["indicators"]):
        f = _row_filter(_conditions(ind.get("filter"), "")[0], table)
        scale, offset = float(ind.get("scale", 1)), float(ind.get("offset", 0))
        for i, a in enumerate(alts):
            sel = [r for r in by_alt[a] if f(r)]
            rows_used[(a, ind["id"])] = len(sel)
            v = aggregate(ind, sel, table)
            if v is None:
                errs.append(f"indicator {ind['id']}: no usable rows for alternative {a!r}")
                continue
            v = v * scale + offset
            if not v > 0:
                errs.append(f"indicator {ind['id']}: value {v:g} for alternative {a!r} is not > 0 (ARAS and EDAS divide "
                            "by the values; choose another indicator or set an 'offset' and say why)")
            matrix[i][j] = float(f"{v:.6g}")
    if errs:
        raise SpecError(errs)
    return {"alternatives": alts, "matrix": matrix, "rows": len(rows), "rows_per_alternative":
            {a: len(by_alt[a]) for a in alts}, "rows_used": rows_used}


def _source(ind: dict) -> str:
    agg = ind["aggregation"]
    if agg == "share":
        v = ind["value"]
        return f"{ind['column']} = {' or '.join(map(str, v)) if isinstance(v, list) else v}"
    if agg == "ratio":
        return f"{ind['numerator']} / {ind['denominator']}"
    if agg == "mean_days_between":
        return f"{ind['start']} -> {ind['end']}"
    return ind.get("column") or "(rows)"


def _filter_text(conds) -> str:
    sym = {"equals": "=", "not_equals": "!=", "in": "in", "not_in": "not in", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}
    out = []
    for c in conds:
        op = next(o for o in FILTER_OPS if o in c)
        out.append(f"{c['column']} {sym[op]} {c[op]}")
    return " and ".join(out)


def problem_dict(spec: dict, table, spec_name: str = "", spec_sha256: str = "", confirmed: bool = False) -> dict:
    """Complete problem file (existing schema plus a provenance block) from a spec and a table."""
    s = strip_comments(spec)
    res = compute(s, table)
    p = s["problem"]
    criteria, short, groups, mapping = [], [], [], {}
    used_tasks = {pb.MAIN_TASK}
    for c in s["hierarchy"]["criteria"]:
        name = c["name"].strip()
        criteria.append(name)
        short.append(str(c.get("short") or name))
        if c.get("sub_criteria"):
            task = c.get("task") or re.sub(r"[^a-z0-9_]", "", re.sub(r"[\s\-]+", "_", name.lower()))[:40]
            if not task or not task[0].isalpha():
                task = f"group_{task}"[:40]
            base, n = task, 1
            while task in used_tasks:
                n += 1
                task = f"{base[:36]}_{n}"
            used_tasks.add(task)
            subs = c["sub_criteria"]
            groups.append({"task": task, "parent": name, "criteria": [x["name"].strip() for x in subs],
                           "short": [str(x.get("short") or x["name"].strip()) for x in subs]})
            mapping.update((x["name"].strip(), x["indicator"]) for x in subs)
        else:
            mapping[name] = c["indicator"]
    gconds = _conditions(s.get("filter"), "")[0]
    indicators = []
    for ind in s["indicators"]:
        e = {"code": ind["id"], "name": ind.get("label") or ind["id"], "unit": ind.get("unit", ""),
             "direction": ind["direction"], "aggregation": ind["aggregation"], "source": _source(ind)}
        conds = gconds + _conditions(ind.get("filter"), "")[0]
        if conds:
            e["filter"] = _filter_text(conds)
        for k in ("scale", "offset", "rationale"):
            if k in ind:
                e[k] = ind[k]
        indicators.append(e)
    used_cols = sorted({c for ind in s["indicators"] for c in (ind.get(k) for k in ("column", "numerator",
                        "denominator", "start", "end")) if c} | {s["alternative_column"]}
                       | {c["column"] for c in gconds} | {c["column"] for ind in s["indicators"]
                                                          for c in _conditions(ind.get("filter"), "")[0]})
    out = {
        "_comment": f"Built by ahp-bench criteria build from the data file {table.name} and the criteria spec "
                    f"{spec_name or '(in memory)'}. Rebuild instead of editing the ranking block by hand.",
        "schema_version": 1, "name": p["name"], "title": p.get("title") or p["name"], "goal": p["goal"].strip(),
        "role": p["role"].strip(), "criteria": criteria, "short": short}
    if groups:
        out["groups"] = groups
    out["ranking"] = {"alternatives": res["alternatives"], "indicators": indicators, "criterion_indicator": mapping,
                      "matrix": res["matrix"]}
    out["provenance"] = {
        "built_with": f"ahp-bench {__version__}", "data_file": table.name, "data_sha256": table.sha256,
        "data_rows": len(table.rows), "rows_used": res["rows"], "spec_file": spec_name, "spec_sha256": spec_sha256,
        "criteria_proposed_by": s["proposed_by"], "criteria_confirmed_by_user": bool(confirmed),
        "data_summary": {"encoding": table.encoding, "delimiter": {"\t": "tab"}.get(table.delimiter, table.delimiter),
                         "columns": len(table.header), "alternative_column": s["alternative_column"],
                         "rows_per_alternative": res["rows_per_alternative"],
                         "missing_cells_in_used_columns": {c: sum(1 for v in table.column(c) if is_missing(v))
                                                           for c in used_cols}}}
    pb.problem_from_dict(json.loads(json.dumps(out)))  # must be a valid problem
    return out


# ---------------------------------------------------------------- template from the EDA

def template(eda: dict) -> dict:
    inds = eda.get("candidate_indicators") or []
    chosen, rest = inds[:pb.MAX_N], inds[pb.MAX_N:]
    out_inds = []
    for c in chosen:
        d = {"id": c["id"], "label": c["label"], "aggregation": c["aggregation"]}
        for k in ("column", "value", "numerator", "denominator", "start", "end"):
            if k in c:
                d[k] = c[k]
        d["direction"] = c.get("suggested_direction") or "REVIEW: benefit or cost"
        d["_direction_note"] = c.get("note", "")
        d["unit"] = {"share": "share of rows (0-1)", "count": "rows", "mean_days_between": "days"}.get(c["aggregation"], "")
        d["rationale"] = "REVIEW: why this indicator measures the criterion"
        out_inds.append(d)
    name = re.sub(r"[^a-z0-9_]", "_", Path(str(eda.get("data_file") or "my_problem")).stem.lower()).strip("_")
    name = (name if name[:1].isalpha() else f"p_{name}")[:40] or "my_problem"
    spec = {
        "_DRAFT": "DRAFT built by `ahp-bench criteria template` from the EDA suggestions. A person must review "
                  "every entry: keep only criteria that matter for the decision, fix every direction marked REVIEW, "
                  "arrange the criteria in a hierarchy (3-9 main criteria, optional 3-9 sub-criteria each) and "
                  "fill in goal and role. Then run `ahp-bench criteria validate`. Keys starting with '_' are comments.",
        "spec_version": SPEC_VERSION,
        "proposed_by": "human",
        "_proposed_by_help": "'human' when a person wrote or edited the criteria, 'host agent' or a model id when "
                             "they come from an LLM proposal.",
        "problem": {"name": name, "title": f"Decision based on {eda.get('data_file', 'the data')}",
                    "goal": "REVIEW: neutral noun phrase naming the decision, e.g. supplier evaluation",
                    "role": "REVIEW: one line describing the respondent, e.g. You are a procurement professional."},
        "alternative_column": eda.get("alternative_column") or "REVIEW: column naming the alternatives",
        "indicators": out_inds,
        "hierarchy": {
            "_help": "Flat starting point: one main criterion per indicator. Group related leaves as sub_criteria "
                     "of a main criterion: {\"name\": \"Delivery\", \"sub_criteria\": [{\"name\": ..., \"indicator\": ...}]}.",
            "criteria": [{"name": c["label"], "indicator": c["id"]} for c in chosen]},
    }
    if rest:
        spec["_unused_suggestions"] = [{"id": c["id"], "aggregation": c["aggregation"], "label": c["label"]} for c in rest]
    if len(chosen) < pb.MIN_N:
        spec["_WARNING"] = (f"only {len(chosen)} candidate indicators: an AHP matrix needs at least {pb.MIN_N} criteria. "
                            "Add indicators (derived columns) or reconsider whether AHP fits this data set.")
    return spec


# ---------------------------------------------------------------- LLM proposal (host and API mode)

CRITERIA_SYSTEM = ("You help structure a multi-criteria decision problem from a tabular data set. You propose decision "
                   "criteria and the indicators that measure them, using only the columns described. Respond only in "
                   "the requested JSON format.")

CRITERIA_RULES = """Rules:
1. Propose only criteria that the columns above can measure. Each indicator is computed per alternative with one of these aggregations: mean, sum, median, min, max (of a numeric column); count (number of rows, or of non-empty cells of a column); share (share of rows where a column equals a value); ratio (sum of one numeric column divided by the sum of another); mean_days_between (mean number of days between two date columns). An optional row filter restricts the rows used: {"column": "...", "equals": "..."} (also not_equals, in, not_in, gt, ge, lt, le).
2. For every indicator state its direction ("benefit" = higher values are better, "cost" = lower values are better), its unit and a one-sentence rationale that names the columns used.
3. Do not assign weights, importance levels or a priority order, and do not say which criterion matters more. The order in which you list the criteria carries no meaning. Weights are elicited later in a separate pairwise-comparison step.
4. Arrange the criteria as an AHP hierarchy: 3 to 9 main criteria; a main criterion may have 3 to 9 sub-criteria. Every leaf criterion (a main criterion without sub-criteria, or a sub-criterion) maps to exactly one indicator, and every indicator to exactly one leaf criterion.
5. Use neutral criterion names without evaluative words. Do not use identifier columns or the alternative column as indicators.
6. If the columns cannot support at least 3 criteria, return the criteria they can support and explain the gap in "notes"."""

CRITERIA_FORMAT = """Response format (only this JSON):
{"alternative_column": "column naming the alternatives",
 "goal": "short neutral noun phrase naming the decision",
 "indicators": [{"id": "SHORT_ID", "label": "...", "aggregation": "mean", "column": "...", "direction": "benefit", "unit": "...", "rationale": "..."},
                {"id": "...", "label": "...", "aggregation": "share", "column": "...", "value": "...", "filter": {"column": "...", "not_equals": "..."}, "direction": "cost", "unit": "...", "rationale": "..."},
                {"id": "...", "label": "...", "aggregation": "ratio", "numerator": "...", "denominator": "...", "direction": "cost", "unit": "...", "rationale": "..."}],
 "hierarchy": {"criteria": [{"name": "...", "indicator": "SHORT_ID"},
                            {"name": "...", "sub_criteria": [{"name": "...", "indicator": "..."}, ...]}]},
 "notes": "..."}"""


def _column_line(c: dict) -> str:
    bits = [c["type"] + (" 0/1" if c.get("binary") else ""), f"{c['missing']} missing", f"{c['distinct']} distinct"]
    n = c.get("numeric")
    if n:
        bits.append("min {min}, max {max}, mean {mean}, median {median}".format(**{k: f"{n[k]:.4g}" for k in
                                                                                  ("min", "max", "mean", "median")}))
    if c.get("date_range"):
        bits.append(f"{c['date_range'][0]} to {c['date_range'][1]}")
    if c.get("values") and c["type"] != "identifier":
        bits.append("values: " + ", ".join(f"{v} ({k})" for v, k in c["values"])
                    + ("" if c.get("values_complete") else ", ..."))
    return f"- {c['name']} ({'; '.join(bits)})"


def criteria_messages(eda: dict, goal: str | None = None) -> list:
    alts = eda.get("candidate_alternative_columns") or []
    lines = [f"A data set will be used to compare alternatives in {goal or 'a decision'} with the Analytic Hierarchy "
             "Process (AHP). Propose the criteria hierarchy and one indicator per leaf criterion.", "",
             f"Data set: {eda.get('data_file')}, {eda.get('rows')} rows. Columns:"]
    lines += [_column_line(c) for c in eda.get("columns", [])]
    lines += ["", "Candidate alternative columns: " + (", ".join(f"{a['column']} ({a['distinct']} distinct values)"
                                                               for a in alts) or "none found")]
    if eda.get("alternative_column"):
        lines.append(f"Alternative column chosen for this analysis: {eda['alternative_column']}")
    user = "\n".join(lines) + "\n\n" + CRITERIA_RULES + "\n\n" + CRITERIA_FORMAT
    return [{"role": "system", "content": CRITERIA_SYSTEM}, {"role": "user", "content": user}]


def prompt_sha256(messages) -> str:
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()


def parse_reply(text: str) -> dict:
    """The JSON object of a reply (code fences and text around the object are ignored)."""
    if text is None:
        raise SpecError(["empty reply"])
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < a:
        raise SpecError(["no JSON object in the reply"])
    try:
        obj = json.loads(t[a:b + 1])
    except json.JSONDecodeError as e:
        raise SpecError([f"reply is not valid JSON ({e.msg})"]) from None
    if not isinstance(obj, dict):
        raise SpecError(["reply is not a JSON object"])
    return obj


def _drop_weight_keys(obj, path="") -> list:
    removed = []
    if isinstance(obj, dict):
        for k in list(obj):
            if k in WEIGHT_KEYS:
                removed.append(f"{path}{k}")
                del obj[k]
            else:
                removed += _drop_weight_keys(obj[k], f"{path}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            removed += _drop_weight_keys(v, f"{path}{i}.")
    return removed


def spec_from_reply(reply: dict, proposed_by: str, name: str, goal: str | None, role: str | None,
                    title: str | None = None) -> tuple[dict, list]:
    """Criteria spec from a parsed proposal. Weight or importance fields are removed (returned as a list)."""
    reply = json.loads(json.dumps(reply))
    removed = _drop_weight_keys({"indicators": reply.get("indicators"), "hierarchy": reply.get("hierarchy")})
    g = (goal or reply.get("goal") or "").strip()
    spec = {"_DRAFT": f"Criteria proposed by {proposed_by} (LLM). A person must review the criteria, directions, "
                      "hierarchy, goal and role before weighting; set proposed_by to 'human' only if a person "
                      "rewrote the criteria.",
            "spec_version": SPEC_VERSION, "proposed_by": proposed_by,
            "problem": {"name": name, "title": title or f"{g[:1].upper()}{g[1:]}" if g else name, "goal": g,
                        "role": role or (f"You are an experienced professional in {g}." if g else "")},
            "alternative_column": reply.get("alternative_column"),
            "indicators": reply.get("indicators"), "hierarchy": reply.get("hierarchy")}
    if reply.get("notes"):
        spec["_model_notes"] = reply["notes"]
    if removed:
        spec["_removed_weight_fields"] = removed
    return spec, removed


def validate_against(spec: dict, data_path) -> list:
    errs = check_structure(spec)
    if errs:
        return errs
    try:
        compute(spec, read_table(data_path))
    except SpecError as e:
        return e.errors
    except (TableError, OSError) as e:
        return [str(e)]
    return []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def log_record(**fields) -> dict:
    return {"kind": "criteria_proposal", "prompt_id": PROMPT_ID, "timestamp_utc": _now(), **fields}


def finish_proposal(raw_reply, messages, eda, data_path, out, proposed_by, name, goal, role, log_path, extra) -> int:
    """Parse, clean, validate and write a proposal; append the log record. Shared by host and API mode."""
    from . import store
    rec = log_record(**extra, prompt_sha256=prompt_sha256(messages), messages=messages, raw_reply=raw_reply,
                     data_file=Path(data_path).name, data_sha256=eda.get("data_sha256"))
    try:
        reply = parse_reply(raw_reply)
    except SpecError as e:
        rec.update(parse_ok=False, parse_error=str(e), spec=None, validation_errors=None)
        store.append_jsonl(log_path, rec)
        print(f"reply not parsed: {e}", file=sys.stderr)
        return 2
    spec, removed = spec_from_reply(reply, proposed_by, name, goal, role)
    errs = validate_against(spec, data_path)
    if errs:
        spec["_validation_errors"] = errs
    rec.update(parse_ok=True, parse_error=None, spec=spec, removed_weight_fields=removed, validation_errors=errs,
               spec_file=Path(out).name)
    store.append_jsonl(log_path, rec)
    write_json(out, spec)
    if removed:
        print(f"removed weight / importance fields from the proposal: {', '.join(removed)}")
    print(f"wrote {out} (proposed by {proposed_by}; a draft for the user to review); log: {log_path}")
    if errs:
        print("the proposal does not validate against the data:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------- CLI

def _load_eda(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _print_errors(errs) -> int:
    print("criteria spec is not valid:", file=sys.stderr)
    for e in errs:
        print(f"  - {e}", file=sys.stderr)
    return 2


def _matrix_text(d: dict) -> str:
    r = d["ranking"]
    codes = [i["code"] for i in r["indicators"]]
    w = max(len(a) for a in r["alternatives"] + ["Alternative"])
    cw = [max(len(c), max(len(f"{row[j]:g}") for row in r["matrix"])) for j, c in enumerate(codes)]
    lines = ["  ".join(["Alternative".ljust(w)] + [c.rjust(cw[j]) for j, c in enumerate(codes)])]
    for a, row in zip(r["alternatives"], r["matrix"]):
        lines.append("  ".join([a.ljust(w)] + [f"{v:g}".rjust(cw[j]) for j, v in enumerate(row)]))
    lines.append("directions: " + ", ".join(f"{i['code']}={i['direction']}" for i in r["indicators"]))
    return "\n".join(lines)


def main(argv=None, transport=None) -> int:
    ap = argparse.ArgumentParser(prog="ahp-bench criteria", description="Stages 1-2: criteria spec, performance "
                                 "matrix and problem file from a data set.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("template", help="draft spec from the EDA suggestions (a person must review it)")
    p.add_argument("--eda", required=True, help="eda.json written by ahp-bench eda")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("validate", help="check a spec against the data and show hierarchy and matrix")
    p.add_argument("--spec", required=True)
    p.add_argument("--data", required=True)
    p = sub.add_parser("build", help="compute the performance matrix and write a problem file")
    p.add_argument("--spec", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True, help="problem file to write, e.g. problems/NAME.json")
    p.add_argument("--confirmed", action="store_true",
                   help="the user has confirmed criteria, directions, hierarchy and matrix (recorded in the file)")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("propose", help="API mode: ask one OpenRouter model for a criteria spec (needs --budget)")
    p.add_argument("--eda", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True, help="spec file to write")
    p.add_argument("--model", required=True, help="model key of the models file or an OpenRouter model id")
    p.add_argument("--budget", type=float, required=True, help="USD cap for criteria proposals (per --tag)")
    p.add_argument("--dry-run", action="store_true", help="print the prompt; no call")
    p.add_argument("--models-file", default=None)
    p.add_argument("--goal", default=None, help="noun phrase naming the decision (else the model proposes one)")
    p.add_argument("--name", default=None, help="problem identifier (default: from the data file name)")
    p.add_argument("--role", default=None, help="respondent line for the weighting prompts")
    p.add_argument("--log", default=None, help="JSONL log (default: criteria_calls.jsonl next to eda.json)")
    p.add_argument("--tag", default="criteria")
    p.add_argument("--temperature", type=float, default=None)
    a = ap.parse_args(argv)

    if a.cmd == "template":
        out = Path(a.out)
        if out.exists() and not a.force:
            print(f"{out} exists; use --force to overwrite", file=sys.stderr)
            return 2
        spec = template(_load_eda(a.eda))
        write_json(out, spec)
        open_items = sum(1 for i in spec["indicators"] if _review(i["direction"]))
        print(f"wrote DRAFT {out}: {len(spec['indicators'])} indicators, {open_items} directions to decide; review "
              "criteria, directions, hierarchy, goal and role, then run ahp-bench criteria validate")
        return 0

    if a.cmd in ("validate", "build"):
        try:
            spec, sha = load_spec(a.spec)
            table = read_table(a.data)
            errs = check_structure(spec)
            if errs:
                return _print_errors(errs)
            d = problem_dict(spec, table, Path(a.spec).name, sha, getattr(a, "confirmed", False))
        except SpecError as e:
            return _print_errors(e.errors)
        except (TableError, OSError, pb.ProblemError) as e:
            return _print_errors([str(e)])
        from . import hierarchy
        prob = pb.problem_from_dict(json.loads(json.dumps(d)))
        print(hierarchy.text_tree(prob))
        print()
        print(_matrix_text(d))
        prov = d["provenance"]
        print(f"\n{len(d['ranking']['alternatives'])} alternatives x {len(d['ranking']['indicators'])} indicators from "
              f"{prov['rows_used']} of {prov['data_rows']} rows; criteria proposed by {prov['criteria_proposed_by']}")
        if a.cmd == "validate":
            print("spec is valid")
            return 0
        out = Path(a.out)
        if out.exists() and not a.force:
            try:
                same = json.loads(out.read_text(encoding="utf-8")) == d
            except (OSError, ValueError):
                same = False
            if not same:
                print(f"{out} exists with other content; use --force to overwrite", file=sys.stderr)
                return 2
        write_json(out, d)
        print(f"wrote {out}" + ("" if a.confirmed else "; criteria_confirmed_by_user is false: show the hierarchy and "
                                "the matrix to the user and rebuild with --confirmed after they agree"))
        return 0

    # propose (API mode)
    from . import store
    from .client import api_key_for, call_model
    from .config import SHARED_KEY_ENV, load_env, load_panel, reasoning_for
    eda = _load_eda(a.eda)
    msgs = criteria_messages(eda, a.goal)
    panel = load_panel(Path(a.models_file) if a.models_file else None)
    pm = panel["models"]
    if a.model in pm:
        key, model = a.model, pm[a.model]
    else:
        key, model = a.model, {"id": a.model, "env": SHARED_KEY_ENV, "developer": a.model.split("/")[0]}
    log = Path(a.log) if a.log else Path(a.eda).parent / "criteria_calls.jsonl"
    spent = sum(float(r.get("cost") or 0) for r in store.read_jsonl(log) if r.get("run_tag") == a.tag)
    temperature = panel["temperature"] if a.temperature is None else a.temperature
    reasoning = reasoning_for(key, pm, panel["default_reasoning"]) if key in pm else panel["default_reasoning"]
    if a.dry_run:
        for m in msgs:
            print(f"--- {m['role']} ---\n{m['content']}")
        print(f"dry run: 1 planned call to {model['id']} (T={temperature}, reasoning={reasoning}); prompt "
              f"{sum(len(m['content']) for m in msgs)} characters; cost unknown (no reference for criteria prompts); "
              f"budget ${a.budget:.4f}, spent ${spent:.4f} under tag {a.tag!r}")
        return 0
    if spent >= a.budget:
        print(f"budget reached (${spent:.4f} >= ${a.budget:.4f}) for tag {a.tag!r}; no call", file=sys.stderr)
        return 2
    load_env(Path.cwd() / ".env")
    if not api_key_for(model, pm):
        print(f"missing API key variable {model.get('env') or SHARED_KEY_ENV} (or {SHARED_KEY_ENV})", file=sys.stderr)
        return 2
    call = call_model(key, msgs, temperature=temperature, reasoning=reasoning, max_tokens=panel["max_tokens"],
                      transport=transport, model=model, panel=pm)
    extra = {"run_tag": a.tag, "source": "api", "eda_file": Path(a.eda).name,
             **{k: v for k, v in call.items() if k != "content"}}
    name = a.name or _name_from(a.data)
    print(f"call {'ok' if call['ok'] else 'failed'}: cost ${float(call.get('cost') or 0):.4f}, "
          f"total ${spent + float(call.get('cost') or 0):.4f}")
    if not call["ok"]:
        from . import store as st
        st.append_jsonl(log, log_record(**extra, prompt_sha256=prompt_sha256(msgs), messages=msgs, raw_reply=None,
                                        parse_ok=False, parse_error=call["error"], spec=None))
        print(f"call failed: {call['error']}", file=sys.stderr)
        return 2
    return finish_proposal(call["content"], msgs, eda, a.data, Path(a.out), model["id"], name, a.goal, a.role, log,
                           extra)


def _name_from(data_path) -> str:
    n = re.sub(r"[^a-z0-9_]", "_", Path(data_path).stem.lower()).strip("_")
    return (n if n[:1].isalpha() else f"p_{n}")[:40] or "my_problem"


if __name__ == "__main__":
    raise SystemExit(main())
