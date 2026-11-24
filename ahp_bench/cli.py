"""Shared command-line options: --problem, --results, --models-file."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from . import problem as pb
from .config import load_panel


def add_common(ap, models: bool = True) -> None:
    ap.add_argument("--problem", default=None,
                    help="problem definition JSON (default: the study, problems/supplier_selection.json)")
    ap.add_argument("--results", default=None,
                    help="results folder; to verify or analyse the published study: the results folder of "
                         f"{STUDY_REPO_URL} (default for new runs: results/<problem name>, results for the "
                         "study problem)")
    if models:
        ap.add_argument("--models-file", default=None, help="model panel JSON (default: models.json)")


def resolve(value, root=None) -> Path:
    """Absolute paths are kept; relative ones are taken from `root` when given (tests), else from the
    current directory."""
    p = Path(value)
    if p.is_absolute():
        return p
    return (Path(root) if root else Path.cwd()) / p


def context(a, root=None, models: bool = True) -> SimpleNamespace:
    base = Path(root) if root else Path.cwd()
    problem = pb.load_problem(resolve(a.problem, root) if a.problem else None)
    if a.results:
        results = resolve(a.results, root)
    else:
        results = base / "results" if problem.is_study else base / "results" / problem.name
    panel = load_panel(resolve(a.models_file, root) if getattr(a, "models_file", None) else None) if models else None
    return SimpleNamespace(problem=problem, results=results, raw=results / "raw", panel=panel, root=base)


STUDY_REPO_URL = "https://github.com/baa-git/llm-ahp-mcdm-study"
STUDY_FILES = {"raw": ("calls.jsonl",), "data": ("expert_matrices.json", "performance_matrix.json")}


def add_data(ap) -> None:
    ap.add_argument("--data", default=None,
                    help="study only: folder with expert_matrices.json and performance_matrix.json "
                         "(default: the 'data' folder next to --results)")


STUDY_HINT = (f"The study's records and data are published separately: clone {STUDY_REPO_URL} and pass "
              "--results <study>/results --data <study>/data.")


def study_paths(a, root=None, need_data: bool = True) -> tuple[Path, Path]:
    """(results folder, data folder) of the published study, from --results / --data. Exits with a message
    naming the study repository when the files are not there (the data are not part of this repository)."""
    results = resolve(a.results, root) if getattr(a, "results", None) else None
    data = resolve(a.data, root) if getattr(a, "data", None) else (results.parent / "data" if results else None)
    if results is None:
        raise SystemExit(f"--results is required for the study. {STUDY_HINT}")
    missing = [str(results / "raw" / f) for f in STUDY_FILES["raw"] if not (results / "raw" / f).is_file()]
    if need_data:
        missing += [str(data / f) for f in STUDY_FILES["data"] if not (data / f).is_file()]
    if missing:
        raise SystemExit(f"study files not found: {', '.join(missing)}. {STUDY_HINT}")
    return results, data


def study_records(a, root=None) -> Path:
    """Results folder of the published study (only the raw records are needed)."""
    return study_paths(a, root, need_data=False)[0]


def check_problem_matches(records, problem) -> list[str]:
    """Tasks whose stored criteria differ from the problem's (records of another problem in the same file)."""
    bad = set()
    for r in records:
        t = r.get("task")
        crit = r.get("criteria")
        if crit is None:
            continue
        if t not in problem.task_names or list(crit) != problem.criteria(t):
            bad.add(str(t))
    return sorted(bad)
