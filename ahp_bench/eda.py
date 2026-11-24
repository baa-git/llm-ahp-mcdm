"""Stage 0: exploratory profile of a tabular data set (standard library only).

ahp-bench eda --data FILE.csv [--alternative-column COL] [--out DIR]

Writes DIR/eda.md (readable) and DIR/eda.json (machine-readable): rows, columns, inferred types, missing
values, distinct counts, numeric summaries (min, max, mean, median, sd), category frequencies of
low-cardinality columns, candidate alternative (entity) columns and candidate indicator columns with a
suggested aggregation. Directions (benefit / cost) are never inferred from the data: a suggestion is made
only from words in the column name and is marked as such; otherwise the direction is left open.
The candidates are suggestions for a person (or an agent) to review, not decisions.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .tabular import TableError, date_ordinal, is_missing, number, read_table

EDA_VERSION = 1
LOW_CARDINALITY = 12          # all category frequencies are listed up to this many distinct values
SHARE_MAX_VALUES = 6          # share-of-value candidates are proposed for categorical columns up to this
ALT_MAX = 100                 # an alternative column has 2..ALT_MAX distinct values, each on several rows

COST_WORDS = ("cost", "price", "defect", "defective", "delay", "delayed", "late", "lead", "leadtime", "days",
              "duration", "return", "returned", "reject", "rejected", "complaint", "complaints", "cancel",
              "cancelled", "canceled", "pending", "error", "errors", "fail", "failure", "failed", "damage",
              "damaged", "scrap", "risk", "emission", "emissions", "waste", "penalty", "noncompliance",
              "incident", "incidents", "backorder", "shortage")
BENEFIT_WORDS = ("quality", "rating", "score", "satisfaction", "ontime", "delivered", "discount", "revenue",
                 "profit", "margin", "compliance", "compliant", "capacity", "reliability", "availability",
                 "fill", "yield", "accuracy", "approved", "passed", "success", "complete", "completed")
POSITIVE_VALUES = ("yes", "true", "y", "1", "evet", "pass", "passed", "ok", "compliant", "approved")
NEGATIVE_VALUES = ("no", "false", "n", "0", "hayir", "hayır", "fail", "failed", "noncompliant", "rejected")
ALT_WORDS = ("supplier", "vendor", "company", "firm", "product", "brand", "model", "store", "branch", "region",
             "site", "plant", "alternative", "option", "candidate", "provider", "carrier", "tedarikci",
             "tedarikçi", "name")
RATIO_NUM = re.compile(r"defect|reject|return|scrap|fail|error|damag|late", re.I)
RATIO_DEN = re.compile(r"quantity|qty|units|ordered|volume|amount|total|shipped", re.I)
ID_NAME = re.compile(r"(^|_|\b)(id|no|nr|number|code|key)($|_|\b)", re.I)


def words(text: str) -> list:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    s = s.lower().replace("on-time", "ontime").replace("on_time", "ontime").replace("on time", "ontime")
    s = s.replace("non-compliance", "noncompliance").replace("non_compliance", "noncompliance")
    return [w for w in re.split(r"[^0-9a-zçğıöşü]+", s) if w]


def slug(text: str, upper: bool = True) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")
    return (s.upper() if upper else s.lower()) or "X"


def direction_hint(column: str, value=None):
    """(direction or None, basis). Only words of the column name (and of the counted value) are used."""
    ws = words(column)
    vws = words(value) if value is not None else []
    col_cost = any(w in COST_WORDS for w in ws)
    col_benefit = any(w in BENEFIT_WORDS for w in ws)
    if vws:
        if any(w in COST_WORDS for w in vws):
            return "cost", f"words in the value {value!r}"
        if any(w in BENEFIT_WORDS for w in vws):
            return "benefit", f"words in the value {value!r}"
        v = str(value).strip().lower()
        if v in POSITIVE_VALUES and (col_benefit or col_cost):
            return ("benefit" if col_benefit else "cost"), f"column name and the value {value!r}"
        if v in NEGATIVE_VALUES and (col_benefit or col_cost):
            return ("cost" if col_benefit else "benefit"), f"column name and the value {value!r}"
        return None, "no hint in the column name or value"
    if col_cost and not col_benefit:
        return "cost", "words in the column name"
    if col_benefit and not col_cost:
        return "benefit", "words in the column name"
    return None, "no hint in the column name"


# ---------------------------------------------------------------- column profile

def _r(x, nd=6):
    return None if x is None else float(f"{x:.{nd}g}")


def profile_column(table, name: str) -> dict:
    vals = table.column(name)
    present = [v for v in vals if not is_missing(v)]
    n, miss = len(vals), len(vals) - len(present)
    out = {"name": name, "type": "empty", "missing": miss, "missing_share": _r(miss / n if n else 0.0, 4),
           "distinct": len(set(present))}
    if not present:
        return out
    dec = table.decimal(name)
    nums = [number(v, dec) for v in present]
    ok = [x for x in nums if x is not None]
    dates = [date_ordinal(v) for v in present]
    n_dates = sum(d is not None for d in dates)
    distinct = out["distinct"]
    if len(ok) >= 0.95 * len(present):
        integral = all(float(x).is_integer() for x in ok)
        out["type"] = "integer" if integral else "numeric"
        if integral and distinct == len(present) and (ID_NAME.search(name) or _consecutive(ok)):
            out["type"] = "identifier"
        out["numeric"] = {"min": _r(min(ok)), "max": _r(max(ok)), "mean": _r(statistics.fmean(ok)),
                          "median": _r(statistics.median(ok)),
                          "sd": _r(statistics.stdev(ok)) if len(ok) > 1 else None,
                          "non_numeric": len(present) - len(ok), "decimal_mark": dec}
        if set(ok) <= {0.0, 1.0}:
            out["binary"] = True
    elif n_dates >= 0.95 * len(present):
        out["type"] = "date"
        ds = [d for d in dates if d is not None]
        import datetime as dt
        out["date_range"] = [dt.date.fromordinal(min(ds)).isoformat(), dt.date.fromordinal(max(ds)).isoformat()]
        out["unparsed_dates"] = len(present) - n_dates
    elif distinct == len(present) and len(present) >= 10:
        out["type"] = "identifier"
    elif distinct <= max(LOW_CARDINALITY * 2, ALT_MAX if distinct < len(present) / 2 else 0):
        out["type"] = "categorical"
    else:
        out["type"] = "text"
    if out["type"] in ("categorical", "text") or (out["type"] == "integer" and distinct <= LOW_CARDINALITY):
        freq = Counter(present).most_common()
        out["values"] = [[v, c] for v, c in (freq if distinct <= LOW_CARDINALITY else freq[:5])]
        out["values_complete"] = distinct <= LOW_CARDINALITY
    return out


def _consecutive(xs) -> bool:
    s = sorted(xs)
    return len(s) > 5 and all(b - a == 1 for a, b in zip(s, s[1:]))


# ---------------------------------------------------------------- candidates

def alternative_candidates(table, cols: list) -> list:
    out = []
    n = len(table.rows)
    for c in cols:
        if c["type"] not in ("categorical", "text"):
            continue
        d = c["distinct"]
        present = n - c["missing"]
        if not 2 <= d <= ALT_MAX or d >= present:
            continue
        counts = Counter(v for v in table.column(c["name"]) if not is_missing(v))
        hint = any(w in ALT_WORDS for w in words(c["name"]))
        reason = []
        if hint:
            reason.append("the column name suggests entities")
        if 3 <= d <= 30:
            reason.append(f"{d} distinct values, each on {min(counts.values())}-{max(counts.values())} rows")
        else:
            reason.append(f"{d} distinct values" + (" (fewer than 3 alternatives)" if d < 3 else " (many alternatives)"))
        if c["missing"]:
            reason.append(f"{c['missing']} rows without a value")
        out.append({"column": c["name"], "distinct": d, "rows_min": min(counts.values()),
                    "rows_max": max(counts.values()), "alternatives": sorted(counts) if d <= 30 else None,
                    "score": (hint, 3 <= d <= 30, c["type"] != "integer", -c["missing"]), "reason": "; ".join(reason)})
    out.sort(key=lambda x: x["score"], reverse=True)
    for x in out:
        x.pop("score")
    return out


def _cand(ident, label, aggregation, columns, direction, basis, note, **extra) -> dict:
    d = {"id": ident, "label": label, "aggregation": aggregation, **columns, **extra,
         "suggested_direction": direction, "direction_basis": basis}
    notes = [note] if note else []
    if direction is None:
        notes.append("direction cannot be inferred from the data: decide benefit (higher is better) or cost "
                     "(lower is better)")
    else:
        notes.append("direction suggested from words only; confirm it")
    d["note"] = "; ".join(notes)
    return d


def indicator_candidates(table, cols: list, alt_column: str | None) -> tuple[list, list]:
    out, notes = [], []
    by = {c["name"]: c for c in cols}
    usable = [c for c in cols if c["name"] != alt_column]
    for c in usable:
        name, t = c["name"], c["type"]
        if t in ("integer", "numeric"):
            num = c["numeric"]
            if num["min"] == num["max"]:
                notes.append(f"{name}: constant ({num['min']}), not useful as an indicator")
                continue
            d, basis = direction_hint(name)
            note = ("mean per alternative; 'sum' gives totals and favours alternatives with more rows"
                    if not c.get("binary") else "0/1 column: the mean is the share of rows with 1")
            if num["min"] <= 0:
                note += "; values <= 0 occur, the aggregated value must be > 0 for ARAS and EDAS (see 'offset')"
            out.append(_cand(f"{slug(name)}_MEAN", f"Mean {name}", "mean", {"column": name}, d, basis, note))
        elif t == "categorical" and 2 <= c["distinct"] <= SHARE_MAX_VALUES and c.get("values_complete"):
            values = [v for v, _ in c["values"]]
            if c["distinct"] == 2:
                pos = [v for v in values if str(v).strip().lower() in POSITIVE_VALUES]
                values = pos[:1] or [min(c["values"], key=lambda x: x[1])[0]]
            for v in values:
                d, basis = direction_hint(name, v)
                out.append(_cand(f"{slug(name)}_{slug(v)}_SHARE", f"Share of rows with {name} = {v}", "share",
                                 {"column": name}, d, basis, "share of rows with a value in this column", value=v))
        elif t == "categorical" and c["distinct"] > SHARE_MAX_VALUES:
            notes.append(f"{name}: {c['distinct']} categories; group them before using a share")
    nums = [c for c in usable if c["type"] in ("integer", "numeric")]
    for a in nums:
        for b in nums:
            if a is not b and RATIO_NUM.search(a["name"]) and RATIO_DEN.search(b["name"]) and not RATIO_NUM.search(b["name"]):
                d, basis = direction_hint(a["name"])
                out.append(_cand(f"{slug(a['name'])}_RATE", f"{a['name']} per {b['name']}", "ratio",
                                 {"numerator": a["name"], "denominator": b["name"]}, d, basis,
                                 f"sum of {a['name']} / sum of {b['name']} per alternative (a rate)"))
    dates = [c for c in usable if c["type"] == "date"]
    pairs = 0
    for i, a in enumerate(dates):
        for b in dates[i + 1:]:
            starts, ends = table.column(a["name"]), table.column(b["name"])
            diffs = [date_ordinal(e) - date_ordinal(s) for s, e in zip(starts, ends)
                     if date_ordinal(s) is not None and date_ordinal(e) is not None]
            if diffs and sum(x >= 0 for x in diffs) >= 0.9 * len(diffs) and pairs < 3:
                pairs += 1
                out.append(_cand(f"DAYS_{slug(a['name'])}_TO_{slug(b['name'])}",
                                 f"Mean days from {a['name']} to {b['name']}", "mean_days_between",
                                 {"start": a["name"], "end": b["name"]}, "cost",
                                 "a duration (e.g. lead time) is usually a cost",
                                 "mean number of days between the two dates per alternative"))
    if alt_column:
        out.append(_cand("ROWS_COUNT", "Number of rows", "count", {}, None, "no hint",
                         "rows per alternative (e.g. orders); favours alternatives with more rows"))
    order = {"mean": 0, "ratio": 1, "share": 2, "mean_days_between": 3, "count": 4}
    out.sort(key=lambda x: order.get(x["aggregation"], 9))  # stable: column order within each kind
    seen = Counter()
    for x in out:  # unique ids
        seen[x["id"]] += 1
        if seen[x["id"]] > 1:
            x["id"] = f"{x['id']}_{seen[x['id']]}"
    for c in cols:
        if c["type"] == "identifier":
            notes.append(f"{c['name']}: unique per row (identifier), not an indicator")
        if c.get("numeric", {}).get("non_numeric"):
            notes.append(f"{c['name']}: {c['numeric']['non_numeric']} non-numeric cells are treated as missing")
        if c["missing_share"] and c["missing_share"] >= 0.2:
            notes.append(f"{c['name']}: {c['missing_share']:.0%} missing")
    del by
    return out, notes


# ---------------------------------------------------------------- profile

def profile(path, alternative_column: str | None = None) -> dict:
    table = read_table(path)
    cols = [profile_column(table, h) for h in table.header]
    alts = alternative_candidates(table, cols)
    if alternative_column is not None:
        table.index(alternative_column)
        alt = alternative_column
    else:
        alt = alts[0]["column"] if alts else None
    inds, notes = indicator_candidates(table, cols, alt)
    rows = len(table.rows)
    dup = rows - len({tuple(r) for r in table.rows})
    per_alt = None
    if alt:
        counts = Counter(v for v in table.column(alt) if not is_missing(v))
        per_alt = dict(sorted(counts.items()))
        if len(counts) < 3:
            notes.append(f"alternative column {alt!r} has {len(counts)} distinct values; ranking needs at least 2 "
                         "and is more informative with 3 or more")
    else:
        notes.append("no column looks like an alternative (entity) column: name one with --alternative-column, or "
                     "the data set does not describe alternatives")
    n_ind = len(inds)
    if n_ind < 3:
        notes.append(f"only {n_ind} candidate indicators: an AHP matrix needs at least 3 criteria")
    return {"eda_version": EDA_VERSION, "tool": f"ahp-bench {__version__}", "data_file": Path(path).name,
            "data_sha256": table.sha256, "encoding": table.encoding,
            "delimiter": {"\t": "tab"}.get(table.delimiter, table.delimiter), "comment_lines": table.comment_lines,
            "rows": rows, "n_columns": len(table.header), "malformed_rows": table.malformed_rows,
            "duplicate_rows": dup, "missing_cells": sum(c["missing"] for c in cols), "columns": cols,
            "alternative_column": alt, "alternative_column_given": alternative_column is not None,
            "rows_per_alternative": per_alt, "candidate_alternative_columns": alts,
            "candidate_indicators": inds, "notes": notes}


def _fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def markdown(p: dict) -> str:
    md = [f"# Data profile: {p['data_file']}", "",
          f"{p['rows']} rows, {p['n_columns']} columns; encoding {p['encoding']}, delimiter `{p['delimiter']}`"
          + (f", {p['comment_lines']} comment lines skipped" if p["comment_lines"] else "")
          + f"; missing cells {p['missing_cells']}; duplicate rows {p['duplicate_rows']}; malformed rows "
          f"{p['malformed_rows']}. sha256 {p['data_sha256']}.", "",
          "Candidates below are suggestions from the data and the column names. Criteria, directions and the "
          "hierarchy are decisions for the user; nothing here is a weight or an importance order.", "",
          "## Columns", "",
          "| Column | Type | Missing | Distinct | Min | Max | Mean | Median | SD |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in p["columns"]:
        n = c.get("numeric") or {}
        md.append(f"| {c['name']} | {c['type']}{' (0/1)' if c.get('binary') else ''} | {c['missing']} | {c['distinct']} | "
                  + " | ".join(_fmt(n.get(k)) for k in ("min", "max", "mean", "median", "sd")) + " |")
    cats = [c for c in p["columns"] if c.get("values")]
    if cats:
        md += ["", "## Category frequencies", ""]
        for c in cats:
            vals = ", ".join(f"{v} ({k})" for v, k in c["values"])
            md.append(f"- **{c['name']}**: {vals}" + ("" if c.get("values_complete") else ", ... (top 5)"))
    dates = [c for c in p["columns"] if c["type"] == "date"]
    if dates:
        md += ["", "## Dates", ""] + [f"- **{c['name']}**: {c['date_range'][0]} to {c['date_range'][1]}" for c in dates]
    md += ["", "## Candidate alternative columns", ""]
    if p["candidate_alternative_columns"]:
        for a in p["candidate_alternative_columns"]:
            md.append(f"- `{a['column']}`: {a['reason']}" + (f"; values: {', '.join(a['alternatives'])}"
                                                            if a.get("alternatives") else ""))
    else:
        md.append("- none found")
    md += ["", f"Alternative column used below: `{p['alternative_column']}`"
           + (" (given)" if p["alternative_column_given"] else " (first candidate)") if p["alternative_column"]
           else "No alternative column."]
    if p["rows_per_alternative"]:
        md += ["", "| Alternative | Rows |", "| --- | ---: |"]
        md += [f"| {k} | {v} |" for k, v in p["rows_per_alternative"].items()]
    md += ["", "## Candidate indicators", "",
           "| Id | Aggregation | Source | Suggested direction | Note |", "| --- | --- | --- | --- | --- |"]
    for c in p["candidate_indicators"]:
        src = ", ".join(f"{k}={c[k]}" for k in ("column", "value", "numerator", "denominator", "start", "end") if k in c)
        md.append(f"| {c['id']} | {c['aggregation']} | {src} | {c['suggested_direction'] or 'open'} | {c['note']} |")
    if p["notes"]:
        md += ["", "## Notes", ""] + [f"- {x}" for x in p["notes"]]
    md += ["", "Next: `ahp-bench criteria template --eda eda.json --out spec.json` (a draft to review), or let an "
           "agent propose criteria (`ahp-bench host criteria-prompt` / `ahp-bench criteria propose`).", ""]
    return "\n".join(md)


def write(p: dict, out: Path) -> tuple[Path, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    j, m = out / "eda.json", out / "eda.md"
    with j.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(p, ensure_ascii=False, indent=1) + "\n")
    with m.open("w", encoding="utf-8", newline="\n") as f:
        f.write(markdown(p))
    return j, m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ahp-bench eda", description="Stage 0: profile a tabular data set.")
    ap.add_argument("--data", required=True, help="CSV / delimited text file")
    ap.add_argument("--alternative-column", default=None, help="column naming the alternatives (default: suggested)")
    ap.add_argument("--out", default="eda", help="output folder for eda.md and eda.json (default: eda)")
    a = ap.parse_args(argv)
    try:
        p = profile(a.data, a.alternative_column)
    except (TableError, OSError) as e:
        print(f"cannot profile {a.data}: {e}", file=sys.stderr)
        return 2
    j, m = write(p, Path(a.out))
    print(f"{p['rows']} rows x {p['n_columns']} columns ({p['encoding']}, delimiter {p['delimiter']!r}); "
          f"alternative column: {p['alternative_column'] or 'none'}; {len(p['candidate_indicators'])} candidate "
          f"indicators")
    for x in p["notes"]:
        print(f"note: {x}")
    print(f"wrote {m} and {j}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
