"""Decision-problem definitions (JSON): loading, validation and derived structures.

A problem file describes one AHP hierarchy with at most two levels:

- ``goal``: noun phrase used in the prompts ("criteria used in <goal>"), e.g. "supplier evaluation".
- ``role``: first line of the direct/CoT system message.
- ``criteria``: the main criteria (3-9 names; optional ``short`` labels for tables).
- ``groups`` (optional): sub-criteria groups, each with a ``task`` name, the ``parent`` main criterion
  and 3-9 ``criteria``. Every group becomes its own pairwise-comparison task.
- ``question_order`` (optional, per task): list of 1-based [i, j] pairs; default (1,2),(1,3),...,(n-1,n).
- ``ranking`` (optional): alternatives, indicators (code, name, unit, direction benefit|cost), the
  mapping leaf criterion -> indicator code, the performance matrix (rows = alternatives, columns =
  indicators) and an optional transform (``row_share_percent`` over listed indicators).

Keys starting with "_" are comments and ignored. ``python -m ahp_bench.problem PATH`` validates a file
and prints a summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from .config import resource

PROBLEMS_DIR = resource("problems")
STUDY_PROBLEM_PATH = PROBLEMS_DIR / "supplier_selection.json"
EXAMPLE_PROBLEM_PATH = PROBLEMS_DIR / "example_laptop_choice.json"
MIN_N, MAX_N = 3, 9
MAIN_TASK = "main"
NUMBER_WORDS = {3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
DIRECTIONS = ("benefit", "cost")
TRANSFORMS = ("row_share_percent",)
_TASK_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


class ProblemError(ValueError):
    pass


@dataclass
class Task:
    name: str
    criteria: list
    short: list
    pairs: list                 # 0-based (i, j) per question, in question order
    parent: str | None = None   # None for the main task

    @property
    def n(self) -> int:
        return len(self.criteria)

    def intro(self, goal: str) -> str:
        word = NUMBER_WORDS[self.n]
        if self.parent is None:
            return (f"{word} criteria used in {goal} are compared in pairs.\n\n"
                    'For each question: "Which of the following two criteria is more important to you?"')
        return (f'{word} sub-criteria of "{self.parent}" are compared in pairs.\n\n'
                'For each question: "Which of the following two sub-criteria is more important to you?"')


@dataclass
class Problem:
    name: str
    title: str
    goal: str
    role: str
    main: Task
    groups: list
    ranking: dict | None = None
    descriptions: dict | None = None   # neutral criterion descriptions (condition "described")
    guidance: dict | None = None       # user priorities (condition "user_guided"); needs descriptions
    guided: bool = False         # the study's guided condition (problems/guided) is available
    path: Path | None = None
    sha256: str = ""
    raw: dict = field(default_factory=dict)

    # ------------------------------------------------------------ tasks
    @property
    def tasks(self) -> dict:
        out = {MAIN_TASK: self.main}
        out.update((g.name, g) for g in self.groups)
        return out

    @property
    def task_names(self) -> tuple:
        return tuple(self.tasks)

    def task(self, name: str) -> Task:
        try:
            return self.tasks[name]
        except KeyError:
            raise ProblemError(f"unknown task {name!r}; tasks: {', '.join(self.task_names)}") from None

    def criteria(self, task: str) -> list:
        return list(self.task(task).criteria)

    def pairs(self, task: str) -> list:
        return list(self.task(task).pairs)

    @property
    def conditions(self) -> tuple:
        """Stored condition keys available for this problem. The study: lean, cot and its verbatim guided
        prompts. Other problems: lean and cot, plus described / described_cot when the file has neutral
        criterion descriptions, plus user_guided / user_guided_cot when it also has a guidance block."""
        if self.guided:
            return ("lean", "cot", "guided")
        out = ("lean", "cot")
        if self.descriptions:
            out += ("described", "described_cot")
        if self.guidance:
            out += ("user_guided", "user_guided_cot")
        return out

    @property
    def default_conditions(self) -> tuple:
        """Conditions run when none are named: all for the study, lean and cot for other problems."""
        return self.conditions if self.guided else ("lean", "cot")

    @property
    def all_criteria(self) -> list:
        """Every criterion name of the hierarchy (main criteria, then each group's sub-criteria)."""
        return list(self.main.criteria) + [c for g in self.groups for c in g.criteria]

    @property
    def is_study(self) -> bool:
        return self.guided

    # ------------------------------------------------------------ hierarchy
    def group_of(self, main_criterion: str):
        return next((g for g in self.groups if g.parent == main_criterion), None)

    @property
    def leaves(self) -> list:
        """(leaf name, short label, main index, task, index within task) in global order: main criteria
        in order, each replaced by its sub-criteria when it has a group."""
        out = []
        for i, c in enumerate(self.main.criteria):
            g = self.group_of(c)
            if g is None:
                out.append((c, self.main.short[i], i, MAIN_TASK, i))
            else:
                out += [(s, g.short[k], i, g.name, k) for k, s in enumerate(g.criteria)]
        return out

    @property
    def leaf_names(self) -> list:
        return [x[0] for x in self.leaves]

    @property
    def leaf_short(self) -> list:
        return [x[1] for x in self.leaves]

    def global_weights(self, task_weights: dict):
        """Global leaf weights from {task: local weight vector}; None when a needed vector is missing."""
        main = task_weights.get(MAIN_TASK)
        if main is None:
            return None
        main = np.asarray(main, dtype=float)
        out = []
        for leaf, _, i, task, k in self.leaves:
            if task == MAIN_TASK:
                out.append(main[i])
            else:
                w = task_weights.get(task)
                if w is None:
                    return None
                out.append(main[i] * float(np.asarray(w, dtype=float)[k]))
        return np.asarray(out, dtype=float)

    # ------------------------------------------------------------ ranking
    @property
    def has_ranking(self) -> bool:
        return bool(self.ranking) and self.ranking.get("matrix") is not None

    def performance(self, transformed: bool = True) -> dict:
        """Ranking data as arrays: X (alternatives x indicators, transformed unless transformed=False),
        benefit mask, alternatives, indicator codes, and the leaf -> column order."""
        if not self.has_ranking:
            raise ProblemError("problem has no ranking block with a performance matrix")
        r = self.ranking
        codes = [ind["code"] for ind in r["indicators"]]
        X = np.asarray(r["matrix"], dtype=float)
        if transformed and r.get("transform"):
            X = apply_transform(X, codes, r["transform"])
        return {"X": X, "benefit": np.array([ind["direction"] == "benefit" for ind in r["indicators"]]),
                "alternatives": list(r["alternatives"]), "codes": codes,
                "names": [ind["name"] for ind in r["indicators"]],
                "column_of_leaf": [codes.index(r["criterion_indicator"][leaf]) for leaf in self.leaf_names]}

    def indicator_weights(self, leaf_weights):
        """Leaf (global) weights reordered to the indicator columns of the performance matrix."""
        p = self.performance(transformed=False)
        w = np.zeros(len(p["codes"]))
        for leaf_idx, col in enumerate(p["column_of_leaf"]):
            w[col] = float(leaf_weights[leaf_idx])
        return w

    def summary(self) -> str:
        lines = [f"problem: {self.name} ({self.title})", f"goal: {self.goal}",
                 "conditions: " + ", ".join({"lean": "direct (stored key lean)"}.get(c, c) for c in self.conditions)]
        for t in self.tasks.values():
            lines.append(f"task {t.name}: {t.n} criteria, {len(t.pairs)} questions"
                         + (f" (sub-criteria of {t.parent!r})" if t.parent else ""))
        lines.append(f"global leaves: {len(self.leaves)}")
        if self.has_ranking:
            lines.append(f"ranking: {len(self.ranking['alternatives'])} alternatives x "
                         f"{len(self.ranking['indicators'])} indicators")
        else:
            lines.append("ranking: none (stage 2 is skipped)")
        if self.descriptions:
            d = self.descriptions
            lines.append(f"descriptions: drafted by {d.get('drafted_by')}, confirmed by user: "
                         f"{'yes' if d.get('confirmed_by_user') else 'no'}")
        if self.guidance:
            g = self.guidance
            lines.append(f"guidance: source {g.get('source')}, confirmed by user: "
                         f"{'yes' if g.get('confirmed_by_user') else 'no'}")
        prov = (self.raw or {}).get("provenance") if isinstance(self.raw, dict) else None
        if isinstance(prov, dict) and "criteria_confirmed_by_user" in prov:
            lines.append(f"criteria: proposed by {prov.get('criteria_proposed_by')}, confirmed by user: "
                         f"{'yes' if prov.get('criteria_confirmed_by_user') else 'no'}")
        return "\n".join(lines)


def apply_transform(X, codes, transform: dict) -> np.ndarray:
    X = np.array(X, dtype=float)
    if transform.get("type") == "row_share_percent":
        cols = [codes.index(c) for c in transform["indicators"]]
        total = X[:, cols].sum(axis=1, keepdims=True)
        X[:, cols] = X[:, cols] / total * 100.0
        return X
    raise ProblemError(f"unknown transform {transform.get('type')!r}")


# ---------------------------------------------------------------- loading / validation

def _strip_comments(obj):
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _names(value, where: str) -> list:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ProblemError(f"{where}: expected a list of non-empty names")
    names = [v.strip() for v in value]
    if not MIN_N <= len(names) <= MAX_N:
        raise ProblemError(f"{where}: {len(names)} criteria; each matrix needs {MIN_N}-{MAX_N}")
    if len({n.lower() for n in names}) != len(names):
        raise ProblemError(f"{where}: duplicate names")
    return names


def _short(value, names, where):
    if value is None:
        return list(names)
    if not isinstance(value, list) or len(value) != len(names) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ProblemError(f"{where}: 'short' must list one label per criterion")
    if len(set(value)) != len(value):
        raise ProblemError(f"{where}: duplicate short labels")
    return list(value)


def _pairs(value, n, where):
    if value is None:
        return list(combinations(range(n), 2))
    if not isinstance(value, list):
        raise ProblemError(f"{where}: question_order must be a list of [i, j] pairs (1-based)")
    pairs, seen = [], set()
    for p in value:
        if (not isinstance(p, list) or len(p) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in p)
                or not all(1 <= x <= n for x in p) or p[0] == p[1]):
            raise ProblemError(f"{where}: bad pair {p!r} (1-based indices 1..{n}, i != j)")
        key = frozenset(p)
        if key in seen:
            raise ProblemError(f"{where}: pair {p!r} appears twice")
        seen.add(key)
        pairs.append((p[0] - 1, p[1] - 1))
    if len(pairs) != n * (n - 1) // 2:
        raise ProblemError(f"{where}: {len(pairs)} pairs, expected {n * (n - 1) // 2}")
    return pairs


def _validate_ranking(r: dict, leaves: list) -> dict:
    if not isinstance(r, dict):
        raise ProblemError("ranking: expected an object")
    alts = r.get("alternatives")
    if not isinstance(alts, list) or len(alts) < 2 or not all(isinstance(a, str) and a.strip() for a in alts) \
            or len(set(alts)) != len(alts):
        raise ProblemError("ranking.alternatives: expected at least 2 unique names")
    inds = r.get("indicators")
    if not isinstance(inds, list) or not inds:
        raise ProblemError("ranking.indicators: expected a non-empty list")
    codes = []
    for k, ind in enumerate(inds):
        if not isinstance(ind, dict) or not isinstance(ind.get("code"), str) or not ind["code"].strip():
            raise ProblemError(f"ranking.indicators[{k}]: needs a 'code'")
        if ind.get("direction") not in DIRECTIONS:
            raise ProblemError(f"ranking.indicators[{k}]: direction must be 'benefit' or 'cost'")
        ind.setdefault("name", ind["code"])
        ind.setdefault("unit", "")
        codes.append(ind["code"])
    if len(set(codes)) != len(codes):
        raise ProblemError("ranking.indicators: duplicate codes")
    mapping = r.get("criterion_indicator")
    if not isinstance(mapping, dict):
        raise ProblemError("ranking.criterion_indicator: expected an object leaf criterion -> indicator code")
    missing = [lf for lf in leaves if lf not in mapping]
    extra = [k for k in mapping if k not in leaves]
    if missing or extra:
        raise ProblemError(f"ranking.criterion_indicator: missing leaves {missing}, unknown keys {extra}")
    used = [mapping[lf] for lf in leaves]
    if sorted(used) != sorted(codes):
        raise ProblemError("ranking.criterion_indicator: every indicator must be used by exactly one leaf criterion")
    m = r.get("matrix")
    if m is not None:
        try:
            X = np.asarray(m, dtype=float)
        except (TypeError, ValueError):
            raise ProblemError("ranking.matrix: non-numeric entries") from None
        if X.shape != (len(alts), len(codes)):
            raise ProblemError(f"ranking.matrix: shape {X.shape}, expected ({len(alts)}, {len(codes)})")
        if not np.all(np.isfinite(X)) or not np.all(X > 0):
            raise ProblemError("ranking.matrix: entries must be finite and > 0 (ARAS and EDAS divide by them)")
    t = r.get("transform")
    if t is not None:
        if not isinstance(t, dict) or t.get("type") not in TRANSFORMS:
            raise ProblemError(f"ranking.transform: type must be one of {TRANSFORMS}")
        cols = t.get("indicators")
        if not isinstance(cols, list) or len(cols) < 2 or any(c not in codes for c in cols):
            raise ProblemError("ranking.transform.indicators: list of at least 2 indicator codes")
    return r


GUIDANCE_SOURCES = ("user", "orchestrator-draft")


def _validate_descriptions(d, names: list) -> dict:
    if not isinstance(d, dict) or not isinstance(d.get("criteria"), dict):
        raise ProblemError("descriptions: object with 'criteria' (criterion name -> neutral description), "
                           "'drafted_by' and 'confirmed_by_user'")
    if not isinstance(d.get("drafted_by"), str) or not d["drafted_by"].strip():
        raise ProblemError("descriptions.drafted_by: who drafted the descriptions (e.g. user, host agent, a model id)")
    if not isinstance(d.get("confirmed_by_user", False), bool):
        raise ProblemError("descriptions.confirmed_by_user: true or false")
    crit = d["criteria"]
    missing = [n for n in names if not isinstance(crit.get(n), str) or not crit[n].strip()]
    extra = [k for k in crit if k not in names]
    if missing or extra:
        raise ProblemError(f"descriptions.criteria: one non-empty text per criterion of the hierarchy; missing "
                           f"{missing}, unknown {extra}")
    return d


def _validate_guidance(g, tasks: dict) -> dict:
    if not isinstance(g, dict):
        raise ProblemError("guidance: expected an object")
    if g.get("source") not in GUIDANCE_SOURCES:
        raise ProblemError("guidance.source: 'user' (the user's own stated priorities) or 'orchestrator-draft'")
    if not isinstance(g.get("confirmed_by_user", False), bool):
        raise ProblemError("guidance.confirmed_by_user: true or false")
    order, hints = g.get("order") or {}, g.get("hints") or {}
    if not isinstance(order, dict) or not isinstance(hints, dict):
        raise ProblemError("guidance.order and guidance.hints: objects keyed by task name")
    for t, o in order.items():
        if t not in tasks:
            raise ProblemError(f"guidance.order: unknown task {t!r}; tasks: {', '.join(tasks)}")
        if not isinstance(o, list) or sorted(o) != sorted(tasks[t].criteria) or len(set(o)) != len(o):
            raise ProblemError(f"guidance.order.{t}: every criterion of the task exactly once, most important first")
    for t, h in hints.items():
        if t not in tasks:
            raise ProblemError(f"guidance.hints: unknown task {t!r}; tasks: {', '.join(tasks)}")
        if not isinstance(h, list) or not all(isinstance(x, str) and x.strip() for x in h):
            raise ProblemError(f"guidance.hints.{t}: list of non-empty texts")
    if not any(order.values()) and not any(hints.values()):
        raise ProblemError("guidance: needs an 'order' or 'hints' for at least one task")
    return g


def problem_from_dict(data: dict, path: Path | None = None, sha256: str = "") -> Problem:
    raw = data
    d = _strip_comments(data)
    if not isinstance(d, dict):
        raise ProblemError("problem file must contain a JSON object")
    for key in ("name", "goal", "role"):
        if not isinstance(d.get(key), str) or not d[key].strip():
            raise ProblemError(f"'{key}' is required (non-empty string)")
    if not _TASK_NAME.match(d["name"]):
        raise ProblemError("'name' must be lower-case letters, digits or '_' (max 40), starting with a letter")
    order = d.get("question_order") or {}
    if not isinstance(order, dict):
        raise ProblemError("question_order must be an object task -> list of pairs")
    names = _names(d.get("criteria"), "criteria")
    main = Task(MAIN_TASK, names, _short(d.get("short"), names, "criteria"),
                _pairs(order.get(MAIN_TASK), len(names), f"question_order.{MAIN_TASK}"))
    groups, parents = [], set()
    for k, g in enumerate(d.get("groups") or []):
        where = f"groups[{k}]"
        if not isinstance(g, dict):
            raise ProblemError(f"{where}: expected an object")
        tname = g.get("task")
        if not isinstance(tname, str) or not _TASK_NAME.match(tname) or tname == MAIN_TASK:
            raise ProblemError(f"{where}: 'task' must be an identifier (lower-case) other than 'main'")
        if tname in {x.name for x in groups}:
            raise ProblemError(f"{where}: duplicate task {tname!r}")
        parent = g.get("parent")
        if parent not in names:
            raise ProblemError(f"{where}: parent {parent!r} is not a main criterion")
        if parent in parents:
            raise ProblemError(f"{where}: main criterion {parent!r} already has a group")
        parents.add(parent)
        sub = _names(g.get("criteria"), f"{where}.criteria")
        groups.append(Task(tname, sub, _short(g.get("short"), sub, where),
                           _pairs(order.get(tname), len(sub), f"question_order.{tname}"), parent))
    unknown = [t for t in order if t not in {MAIN_TASK, *(g.name for g in groups)}]
    if unknown:
        raise ProblemError(f"question_order: unknown tasks {unknown}")
    p = Problem(name=d["name"], title=str(d.get("title") or d["name"]), goal=d["goal"].strip(), role=d["role"].strip(),
                main=main, groups=groups, guided=d.get("guided_condition") == "study", path=path, sha256=sha256, raw=raw)
    leaves = p.leaf_names
    if len(set(leaves)) != len(leaves):
        raise ProblemError("leaf criterion names must be unique across the hierarchy")
    if d.get("ranking") is not None:
        p.ranking = _validate_ranking(d["ranking"], leaves)
    if d.get("descriptions") is not None or d.get("guidance") is not None:
        if p.guided:
            raise ProblemError("the study problem keeps its verbatim guided prompts; descriptions and guidance blocks "
                               "are for other problems")
    if d.get("descriptions") is not None:
        p.descriptions = _validate_descriptions(d["descriptions"], p.all_criteria)
    if d.get("guidance") is not None:
        if p.descriptions is None:
            raise ProblemError("guidance needs neutral criterion descriptions ('descriptions') as well")
        p.guidance = _validate_guidance(d["guidance"], p.tasks)
    return p


def load_problem(path=None) -> Problem:
    path = Path(path) if path else STUDY_PROBLEM_PATH
    data = path.read_bytes()
    try:
        obj = json.loads(data.decode("utf-8-sig"))
    except json.JSONDecodeError as e:
        raise ProblemError(f"{path}: invalid JSON ({e.msg}, line {e.lineno})") from None
    return problem_from_dict(obj, path, hashlib.sha256(data).hexdigest())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ahp_bench.problem", description="Validate a problem file.")
    ap.add_argument("path", nargs="?", default=str(STUDY_PROBLEM_PATH))
    a = ap.parse_args(argv)
    try:
        p = load_problem(a.path)
    except (ProblemError, OSError) as e:
        print(f"invalid problem file: {e}", file=sys.stderr)
        return 2
    print(p.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
