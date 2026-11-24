"""AHP computations: reciprocal matrices, eigenvector weights, consistency."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

SCALE = [str(v) for v in range(9, 0, -1)] + [f"1/{v}" for v in range(2, 10)]  # 17 levels
# Saaty's random consistency index for n = 3..9 (n = 1, 2 are always consistent: CR = 0).
RI = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def scale_value(value) -> float:
    """Numeric value of a 17-level answer ("3", "1/5"); bare integers 1-9 are also accepted."""
    if isinstance(value, bool):
        raise ValueError(f"value not on scale: {value!r}")
    if isinstance(value, (int, float)) and float(value).is_integer():
        value = str(int(value))
    if not isinstance(value, str) or value.strip() not in SCALE:
        raise ValueError(f"value not on scale: {value!r}")
    return float(Fraction(value.strip()))


def matrix_from_answers(n: int, pairs, answers) -> np.ndarray:
    """answers[k] is the judgement for pairs[k] = (i, j): a_ij = v, a_ji = 1/v."""
    if len(pairs) != len(answers):
        raise ValueError("answers and pairs differ in length")
    a = np.ones((n, n))
    for (i, j), v in zip(pairs, answers):
        a[i, j] = scale_value(v)
        a[j, i] = 1.0 / a[i, j]
    return a


def reciprocal_from_upper(m) -> np.ndarray:
    """Analysis matrix from the upper triangle: diagonal 1, a_ji = 1/a_ij."""
    m = np.asarray(m, dtype=float)
    n = m.shape[0]
    a = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a[i, j], a[j, i] = m[i, j], 1.0 / m[i, j]
    return a


def eigen_weights(a) -> tuple[np.ndarray, float]:
    """Principal right eigenvector (normalised, positive) and lambda_max."""
    vals, vecs = np.linalg.eig(np.asarray(a, dtype=float))
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    return w / w.sum(), float(vals[k].real)


def consistency(a) -> dict:
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    w, lam = eigen_weights(a)
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    cr = ci / RI[n] if n in RI else 0.0
    return {"weights": w, "lambda_max": lam, "CI": ci, "CR": cr}


def geometric_mean_matrix(matrices) -> np.ndarray:
    """Element-wise geometric mean of judgement matrices (AIJ aggregation)."""
    stack = np.array([np.asarray(m, dtype=float) for m in matrices])
    return np.exp(np.log(stack).mean(axis=0))
