"""Pure analysis functions: record selection, AHP aggregation, global weights,
supplier ranking (TOPSIS, ARAS, EDAS) and weight-vector similarity.

Conventions
- Priorities: principal eigenvector (ahp.consistency) for every matrix, experts included.
- Group aggregation: element-wise geometric mean of matrices, then eigenvector.
- Global weights (8): delivery local weights x Delivery main weight, then the
  Financial, Quality, Volume and Compliance main weights (order D1..D4, F, Q, V, C).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from . import ahp, prompts

CONDITIONS = tuple(prompts.CONDITIONS)
TASKS = tuple(prompts.TASKS)
MAIN_CRITERIA = list(prompts.MAIN_CRITERIA)
DELIVERY_SUBCRITERIA = list(prompts.DELIVERY_SUBCRITERIA)
MAIN_SHORT = ["Delivery", "Financial", "Quality", "Volume", "Compliance"]
DELIVERY_SHORT = ["D1 Delivered", "D2 Pending", "D3 Cancelled", "D4 Partially delivered"]
GLOBAL_CODES = ["D1", "D2", "D3", "D4", "F", "Q", "V", "C"]
GLOBAL_LABELS = DELIVERY_SHORT + ["F Financial", "Q Quality", "V Volume", "C Compliance"]
CR_THRESHOLD = 0.10
CR_DECIMALS = 4


def is_consistent(cr, threshold: float = CR_THRESHOLD) -> bool:
    """Shared consistency rule: CR rounded to 4 decimals is <= 0.10 (e.g. 0.10004 counts as consistent,
    0.1001 does not). Used for source selection, stopping, summaries and report tables."""
    return round(float(cr), CR_DECIMALS) <= threshold


# ---------------------------------------------------------------- records

def load_records(path) -> tuple[list[dict], int]:
    """Read a JSONL file read-only. Lines that are not valid JSON objects (e.g. a partially
    written last line) are skipped; their number is returned."""
    path = Path(path)
    if not path.exists():
        return [], 0
    records, bad = [], 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                bad += 1
    return records, bad


def valid_matrix(m, n: int | None = None) -> bool:
    try:
        a = np.asarray(m, dtype=float)
    except (TypeError, ValueError):
        return False
    if a.ndim != 2 or a.shape[0] != a.shape[1] or (n is not None and a.shape[0] != n):
        return False
    return bool(np.all(np.isfinite(a)) and np.all(a > 0))


def record_key(r: dict) -> tuple:
    return (r.get("model_key"), r.get("condition"), r.get("task"), int(r.get("rep") or 0))


def select_records(records, tag: str = "main", problem=None) -> tuple[dict, Counter, list[dict]]:
    """Return (selected, failures, tagged).

    selected: (model, condition, task, rep) -> first record with parse_ok true and a valid
    n x n matrix, in file order. failures: Counter of (model, condition, task) -> number of
    records with parse_ok false (or an unusable matrix). tagged: all records with run_tag == tag.
    `problem` (default: the study) gives the matrix size per task.
    """
    sizes = ({t: len(c) for t, c in prompts.CRITERIA.items()} if problem is None
             else {t: problem.task(t).n for t in problem.task_names})
    tagged = [r for r in records if r.get("run_tag") == tag]
    selected: dict = {}
    failures: Counter = Counter()
    for r in tagged:
        key = record_key(r)
        n = sizes.get(r.get("task")) or None
        ok = bool(r.get("parse_ok")) and valid_matrix(r.get("matrix"), n)
        if not ok:
            failures[key[:3]] += 1
            continue
        if key not in selected:
            selected[key] = r
    return selected, failures, tagged


def matrices(selected: dict, condition: str, task: str, models=None) -> list[tuple]:
    """Sorted list of (model, rep, matrix ndarray) for one condition and task."""
    out = []
    for (m, c, t, rep), r in selected.items():
        if c == condition and t == task and (models is None or m in models):
            out.append((m, rep, np.asarray(r["matrix"], dtype=float)))
    return sorted(out, key=lambda x: (x[0], x[1]))


# ---------------------------------------------------------------- AHP

def priority(matrix) -> tuple[np.ndarray, float]:
    """Principal eigenvector weights and CR of one matrix."""
    c = ahp.consistency(np.asarray(matrix, dtype=float))
    return np.asarray(c["weights"], dtype=float), float(c["CR"])


def group_priority(mats) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Geometric-mean aggregation of matrices, then eigenvector. None if no matrices."""
    mats = [np.asarray(m, dtype=float) for m in mats]
    if not mats:
        return None
    g = ahp.geometric_mean_matrix(mats)
    w, cr = priority(g)
    return w, cr, g


def global_weights(main_w, delivery_w) -> np.ndarray:
    main_w, delivery_w = np.asarray(main_w, dtype=float), np.asarray(delivery_w, dtype=float)
    if main_w.shape != (5,) or delivery_w.shape != (4,):
        raise ValueError("expected 5 main and 4 delivery weights")
    return np.r_[main_w[0] * delivery_w, main_w[1:]]


def panel_weights(selected: dict, condition: str, models=None) -> dict:
    """Group weights for a set of models in one condition (all their matrices pooled).
    Keys: main, delivery, global (arrays or None), cr_main, cr_delivery, n_main, n_delivery."""
    out = {}
    for task in TASKS:
        mats = [m for _, _, m in matrices(selected, condition, task, models)]
        g = group_priority(mats)
        out[task] = g[0] if g else None
        out[f"cr_{task}"] = g[1] if g else None
        out[f"n_{task}"] = len(mats)
    out["global"] = (global_weights(out["main"], out["delivery"])
                     if out["main"] is not None and out["delivery"] is not None else None)
    return out


def problem_panel_weights(selected: dict, condition: str, problem, models=None) -> dict:
    """panel_weights for any problem: {task: weights or None, cr_<task>, n_<task>, global}."""
    out = {}
    for task in problem.task_names:
        mats = [m for _, _, m in matrices(selected, condition, task, models)]
        g = group_priority(mats)
        out[task] = g[0] if g else None
        out[f"cr_{task}"] = g[1] if g else None
        out[f"n_{task}"] = len(mats)
    out["global"] = problem.global_weights({t: out[t] for t in problem.task_names})
    return out


def load_expert_matrices(path, problem) -> dict:
    """Expert file for any problem: {"experts": [labels], "<task>": [n x n matrices, one per expert], ...}
    (the study's data/expert_matrices.json has this layout). Returns {experts, <task>: [arrays]}."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    names = d.get("experts")
    if not isinstance(names, list) or not names:
        raise ValueError(f"{path}: 'experts' must list the expert labels")
    out = {"experts": [str(x) for x in names]}
    for task in problem.task_names:
        mats = d.get(task)
        if not isinstance(mats, list) or len(mats) != len(names):
            raise ValueError(f"{path}: '{task}' must hold one matrix per expert")
        n = problem.task(task).n
        arrs = [np.asarray(m, dtype=float) for m in mats]
        if any(a.shape != (n, n) or not valid_matrix(a) for a in arrs):
            raise ValueError(f"{path}: '{task}' matrices must be positive {n} x {n}")
        out[task] = arrs
    return out


def problem_expert_weights(expert: dict, problem) -> dict:
    """Expert group weights per task (geometric mean, eigenvector), per-expert CRs and global weights."""
    out = {}
    for task in problem.task_names:
        out[f"cr_{task}"] = [priority(m)[1] for m in expert[task]]
        g = group_priority(expert[task])
        out[task], out[f"group_cr_{task}"] = g[0], g[1]
    out["global"] = problem.global_weights({t: out[t] for t in problem.task_names})
    return out


def expert_weights(expert: dict) -> dict:
    """Expert group weights and per-expert CRs from the expert data dict."""
    out = {"cr_main": [priority(m)[1] for m in expert["main"]],
           "cr_delivery": [priority(m)[1] for m in expert["delivery"]]}
    gm, gd = group_priority(expert["main"]), group_priority(expert["delivery"])
    out.update(main=gm[0], delivery=gd[0], group_cr_main=gm[1], group_cr_delivery=gd[1])
    out["global"] = global_weights(gm[0], gd[0])
    return out


# ---------------------------------------------------------------- performance & ranking

def delivery_rates(X) -> np.ndarray:
    """Replace the four delivery counts (columns 0-3) with percentages of their row total."""
    X = np.array(X, dtype=float)
    total = X[:, :4].sum(axis=1, keepdims=True)
    X[:, :4] = X[:, :4] / total * 100.0
    return X


def topsis(X, w, benefit) -> np.ndarray:
    X, w, benefit = np.asarray(X, float), np.asarray(w, float), np.asarray(benefit, bool)
    v = X / np.sqrt((X * X).sum(axis=0)) * w
    best = np.where(benefit, v.max(axis=0), v.min(axis=0))
    worst = np.where(benefit, v.min(axis=0), v.max(axis=0))
    dp = np.sqrt(((v - best) ** 2).sum(axis=1))
    dm = np.sqrt(((v - worst) ** 2).sum(axis=1))
    return dm / (dp + dm)


def aras(X, w, benefit) -> np.ndarray:
    """ARAS utility degrees K_i = S_i / S_0; cost criteria inverted (1/x); the ideal row is
    included in the normalisation denominator."""
    X, w, benefit = np.asarray(X, float), np.asarray(w, float), np.asarray(benefit, bool)
    b = np.where(benefit, X, 1.0 / X)
    opt = b.max(axis=0)
    denom = b.sum(axis=0) + opt
    return ((b / denom) @ w) / ((opt / denom) @ w)


def edas(X, w, benefit) -> np.ndarray:
    X, w, benefit = np.asarray(X, float), np.asarray(w, float), np.asarray(benefit, bool)
    avg = X.mean(axis=0)
    signed = np.where(benefit, (X - avg) / avg, (avg - X) / avg)
    sp = np.maximum(signed, 0) @ w
    sn = np.maximum(-signed, 0) @ w
    nsp = sp / sp.max() if sp.max() > 0 else np.zeros_like(sp)
    nsn = 1 - sn / sn.max() if sn.max() > 0 else np.ones_like(sn)
    return 0.5 * (nsp + nsn)


METHODS = {"TOPSIS": topsis, "ARAS": aras, "EDAS": edas}


def average_ranks(values) -> np.ndarray:
    """Ascending ranks starting at 1; ties get the average rank."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def rank_desc(scores) -> np.ndarray:
    """Rank 1 = highest score (average ranks for ties)."""
    return average_ranks(-np.asarray(scores, dtype=float))


def rank_all(X, w, benefit) -> dict:
    out = {}
    for name, fn in METHODS.items():
        s = fn(X, w, benefit)
        out[name] = {"scores": s, "ranks": rank_desc(s)}
    return out


# ---------------------------------------------------------------- similarity

def spearman(a, b) -> float:
    ra, rb = average_ranks(a), average_ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def kendall_tau(a, b) -> float:
    """Kendall tau-b (equals tau-a when there are no ties)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    conc = disc = ta = tb = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif da == db:
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + ta) * (conc + disc + tb))
    return float((conc - disc) / den) if den > 0 else float("nan")


def similarity(w, ref) -> dict:
    """Spearman rho, Kendall tau, RMSE and mean absolute difference, the last two in
    percentage points (weights x 100)."""
    w, ref = np.asarray(w, float), np.asarray(ref, float)
    d = (w - ref) * 100.0
    return {"spearman": spearman(w, ref), "kendall": kendall_tau(w, ref),
            "rmse_pp": float(np.sqrt((d * d).mean())), "mad_pp": float(np.abs(d).mean())}


# ---------------------------------------------------------------- data files

def load_expert_data(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d["main"] = [np.asarray(m, dtype=float) for m in d["main"]]
    d["delivery"] = [np.asarray(m, dtype=float) for m in d["delivery"]]
    return d


def load_performance(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d["matrix"] = np.asarray(d["matrix"], dtype=float)
    d["benefit_mask"] = np.asarray(d["benefit_mask"], dtype=bool)
    return d


# ---------------------------------------------------------------- weight sensitivity

SENSITIVITY_STEPS = (0.125, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def supplier_initials(alternatives) -> list[str]:
    """Shortest common prefix length (from 1) that makes every supplier label unique."""
    names = [str(a) for a in alternatives]
    for n in range(1, max(map(len, names), default=1) + 1):
        short = [a[:n] for a in names]
        if len(set(short)) == len(short):
            return short
    return names


def ranking_order(scores) -> list[int]:
    """Alternative indices from best to worst (higher score first; ties keep input order)."""
    return [int(i) for i in np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")]


def ranking_string(scores, initials) -> str:
    return ">".join(initials[i] for i in ranking_order(scores))


def one_at_a_time_weights(n: int, index: int, weight: float) -> np.ndarray:
    """Weight `weight` on indicator `index`; the other n-1 share the remainder equally."""
    w = np.full(n, (1.0 - weight) / (n - 1))
    w[index] = weight
    return w


def one_at_a_time_sensitivity(X, benefit, alternatives, steps=SENSITIVITY_STEPS, methods=None) -> list[dict]:
    """For each indicator and method: ranking string at each tested weight and the smallest tested
    weight at which the top-ranked supplier differs from the equal-weight top supplier (None if never)."""
    X = np.asarray(X, dtype=float)
    methods = methods or METHODS
    n = X.shape[1]
    initials = supplier_initials(alternatives)
    out = []
    for i in range(n):
        for name, fn in methods.items():
            base = ranking_order(fn(X, np.full(n, 1.0 / n), benefit))
            rankings, change = [], None
            for s in steps:
                scores = fn(X, one_at_a_time_weights(n, i, s), benefit)
                rankings.append(ranking_string(scores, initials))
                if change is None and ranking_order(scores)[0] != base[0]:
                    change = float(s)
            out.append({"indicator": i, "method": name, "steps": list(steps), "rankings": rankings,
                        "base_ranking": ">".join(initials[k] for k in base), "top_change_at": change})
    return out


MAIN_LEVELS = tuple(round(0.05 * k, 2) for k in range(1, 15))  # 0.05, 0.10, ..., 0.70


def main_level_weights(index: int, level: float) -> np.ndarray:
    """Global weights (D1..D4, F, Q, V, C) when main criterion `index` gets `level` and the other four
    main criteria share 1 - level equally; the Delivery main weight is split equally over D1..D4."""
    main = np.full(5, (1.0 - level) / 4)
    main[index] = level
    return global_weights(main, np.full(4, 0.25))


def _method_result(fn, X, w, benefit, alternatives, initials) -> dict:
    order = ranking_order(fn(X, w, benefit))
    return {"ranking": ">".join(initials[i] for i in order), "top": str(alternatives[order[0]])}


def main_weight_sensitivity(X, benefit, alternatives, levels=MAIN_LEVELS, methods=None) -> list[dict]:
    """Long-format rows: first the "Equal" reference (all 8 global weights 1/8, criterion None, level None),
    then for each main criterion and level: global weights and, per method, ranking string (initials, best
    first) and top supplier."""
    X = np.asarray(X, dtype=float)
    methods = methods or METHODS
    initials = supplier_initials(alternatives)
    cases = [(None, None, np.full(8, 1.0 / 8))]
    cases += [(k, float(lv), main_level_weights(k, lv)) for k in range(5) for lv in levels]
    out = []
    for k, lv, w in cases:
        out.append({"criterion": None if k is None else MAIN_SHORT[k], "level": lv, "weights": w,
                    "methods": {name: _method_result(fn, X, w, benefit, alternatives, initials)
                                for name, fn in methods.items()}})
    return out


def sensitivity_summary(rows, methods=None) -> list[dict]:
    """Per varied main criterion x method (from main_weight_sensitivity rows): top supplier at the lowest
    and highest level, number of distinct rankings over the levels, number of levels whose top supplier
    equals the equal-weight top, and the lowest level whose top differs from it (None if never)."""
    methods = list(methods or METHODS)
    equal = next(r for r in rows if r["criterion"] is None)
    out = []
    for crit in MAIN_SHORT:
        rs = sorted((r for r in rows if r["criterion"] == crit), key=lambda r: r["level"])
        if not rs:
            continue
        for m in methods:
            base = equal["methods"][m]["top"]
            tops = [r["methods"][m]["top"] for r in rs]
            out.append({"criterion": crit, "method": m, "levels": [r["level"] for r in rs],
                        "top_low": tops[0], "top_high": tops[-1], "equal_top": base,
                        "distinct_rankings": len({r["methods"][m]["ranking"] for r in rs}),
                        "levels_same_top": sum(t == base for t in tops),
                        "first_change_at": next((r["level"] for r, t in zip(rs, tops) if t != base), None)})
    return out
