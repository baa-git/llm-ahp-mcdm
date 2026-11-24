"""Stand-ins for the study's data files, so that the tests run without the study repository.

DATA is a temporary folder with
- performance_matrix.json: the ranking block of problems/supplier_selection.json in the layout of the
  study's data/performance_matrix.json (the same numbers);
- expert_matrices.json: seven synthetic, reciprocal expert matrices per task (not the study's experts).

STUDY_DIR is the local copy of https://github.com/baa-git/llm-ahp-mcdm-study when the environment variable
AHP_STUDY_DIR points to it, else None; tests that compare with the published records are skipped without it.
"""
import atexit
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = Path(os.environ["AHP_STUDY_DIR"]) if os.environ.get("AHP_STUDY_DIR") else None


def _matrix(w, noise, rng):
    w = np.asarray(w, float)
    a = w[:, None] / w[None, :]
    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            a[i, j] *= float(np.exp(rng.normal(0, noise)))
            a[j, i] = 1 / a[i, j]
    return a.tolist()


def write_data(folder: Path) -> Path:
    from ahp_bench import problem as pb
    p = pb.load_problem(pb.STUDY_PROBLEM_PATH)
    perf = p.performance(transformed=False)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "performance_matrix.json").write_text(json.dumps({
        "alternatives": perf["alternatives"], "benefit_mask": perf["benefit"].tolist(), "codes": perf["codes"],
        "criteria": perf["names"], "matrix": perf["X"].tolist(),
        "note": "test copy of the ranking block of problems/supplier_selection.json"}, indent=1), encoding="utf-8")
    rng = np.random.default_rng(7)
    experts = [f"E{k}" for k in range(1, 8)]
    main = [_matrix([0.30, 0.15, 0.30, 0.10, 0.15], 0.5, rng) for _ in experts]
    delivery = [_matrix([0.40, 0.20, 0.25, 0.15], 0.5, rng) for _ in experts]
    (folder / "expert_matrices.json").write_text(json.dumps({
        "experts": experts, "main": main, "delivery": delivery, "main_criteria": p.criteria("main"),
        "delivery_subcriteria": p.criteria("delivery"), "note": "synthetic test matrices, not the study's experts"},
        indent=1), encoding="utf-8")
    return folder


_TMP = Path(tempfile.mkdtemp(prefix="ahp_test_data_"))
atexit.register(shutil.rmtree, _TMP, True)
DATA = write_data(_TMP / "data")
