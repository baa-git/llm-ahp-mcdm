"""Strict parsing of model replies. No value repair."""
from __future__ import annotations

import json
import re
from fractions import Fraction

from . import ahp


class ParseError(ValueError):
    pass


_BARE_FRACTION = re.compile(r'(?<!["\w./])(\d+)\s*/\s*(\d+)(?![\w"./])')


def _first_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ParseError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    for k in range(start, len(text)):
        ch = text[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:k + 1]
    raise ParseError("unbalanced JSON object (truncated reply?)")


def _quote_bare_fractions(raw: str) -> str:
    """Quote unquoted fractions (1/3 -> "1/3") outside JSON string literals only."""
    out, buf, in_str, esc = [], [], False, False
    for ch in raw:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            out.append(_BARE_FRACTION.sub(r'"\1/\2"', "".join(buf)))
            buf = []
            out.append(ch)
            in_str = True
        else:
            buf.append(ch)
    out.append(_BARE_FRACTION.sub(r'"\1/\2"', "".join(buf)))
    return "".join(out)


def extract_json(reply) -> tuple[dict, bool]:
    """Return (object, bare_fractions_quoted). Strips ``` fences and takes the first balanced {...}.
    If the object is invalid JSON only because of bare fractions (1/3), they are quoted ("1/3")."""
    if not isinstance(reply, str) or not reply.strip():
        raise ParseError("empty reply")
    text = re.sub(r"```(?:json|JSON)?", "", reply)
    raw = _first_object(text)
    try:
        return json.loads(raw), False
    except json.JSONDecodeError as e:
        fixed = _quote_bare_fractions(raw)
        if fixed != raw:
            try:
                return json.loads(fixed), True
            except json.JSONDecodeError:
                pass
        raise ParseError(f"invalid JSON: {e.msg}") from None


def parse_lean_cot(reply, n_questions: int, cot: bool = False) -> dict:
    obj, quoted = extract_json(reply)
    items = obj.get("answers") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        raise ParseError("missing 'answers' list")
    values = {}
    for it in items:
        if not isinstance(it, dict) or "question" not in it or "value" not in it:
            raise ParseError("answer item without question/value")
        q = it["question"]
        if isinstance(q, str) and q.strip().isdigit():
            q = int(q)
        if not isinstance(q, int) or isinstance(q, bool) or not 1 <= q <= n_questions:
            raise ParseError(f"bad question number {it['question']!r}")
        if q in values:
            raise ParseError(f"duplicate question {q}")
        try:
            ahp.scale_value(it["value"])
        except ValueError as e:
            raise ParseError(f"question {q}: {e}") from None
        v = it["value"]
        values[q] = v.strip() if isinstance(v, str) else str(int(v))
    missing = [q for q in range(1, n_questions + 1) if q not in values]
    if missing:
        raise ParseError(f"missing questions {missing}")
    out = {"answers": [values[q] for q in range(1, n_questions + 1)], "bare_fractions_quoted": quoted}
    if cot:
        out["reasoning"] = obj.get("reasoning")
        out["ranking"] = obj.get("ranking")
    return out


def _number(v) -> float:
    if isinstance(v, bool):
        raise ValueError
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return float(Fraction(v.strip().replace(" ", "")))
    raise ValueError


MATRIX_KEYS = ("matrix", "pairwise_matrix", "comparison_matrix")
# Keywords identifying each canonical criterion in a returned label (lower-case, Turkish or English).
LABEL_KEYS = {
    5: [("teslimat", "delivery", "c1"), ("finans", "financial", "c2"), ("kalite", "quality", "c3"),
        ("hacim", "volume", "c4"), ("uyum", "compliance", "c5")],
    4: [("delivered", "teslim edilen"), ("pending", "bekleyen"), ("cancel", "iptal"), ("partial", "kısm", "kism")],
}


def label_order(labels, n: int):
    """Canonical index of each returned label, or None when labels do not map one-to-one."""
    if not isinstance(labels, list) or len(labels) != n or n not in LABEL_KEYS:
        return None
    order = []
    for lab in labels:
        text = str(lab).lower()
        if n == 4 and text.lstrip().startswith(("partial", "kısm", "kism")):
            hits = [3]
        else:
            hits = [k for k, words in enumerate(LABEL_KEYS[n]) if any(w in text for w in words)]
            if n == 4 and 3 in hits and 0 in hits:      # "Partially Delivered ..." also contains "delivered"
                hits = [3]
        if len(hits) != 1:
            return None
        order.append(hits[0])
    return order if sorted(order) == list(range(n)) else None


def _find_matrix_holder(obj, n: int):
    """First dict (depth-first, document order) holding an n x n list under "matrix".
    Some models nest the requested object one level down; values are not altered."""
    if isinstance(obj, dict):
        keys = list(MATRIX_KEYS) + [k for k in obj if "matrix" in str(k).lower() and "rationale" not in str(k).lower()
                                    and "numeric" not in str(k).lower() and k not in MATRIX_KEYS]
        for key in keys:
            mat = obj.get(key)
            if isinstance(mat, list) and len(mat) == n and all(isinstance(r, list) and len(r) == n for r in mat):
                labels = obj.get("labels")
                if not isinstance(labels, list):
                    labels = next((v for k, v in obj.items() if "label" in str(k).lower()
                                   and isinstance(v, list) and len(v) == n), None)
                return {**obj, "matrix": mat, "labels": labels}
        for v in obj.values():
            found = _find_matrix_holder(v, n)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_matrix_holder(v, n)
            if found is not None:
                return found
    return None


def parse_guided(reply, n: int, tol: float = 0.01) -> dict:
    """Parse {"labels","matrix","rationales"}. Returns the numeric matrix as returned, whether it
    was reciprocal (diagonal 1 and a_ij*a_ji = 1 within tol), and the analysis matrix built
    from the upper triangle."""
    obj, quoted = extract_json(reply)
    obj = _find_matrix_holder(obj, n) or obj
    mat = obj.get("matrix") if isinstance(obj, dict) else None
    if not isinstance(mat, list) or len(mat) != n or any(not isinstance(r, list) or len(r) != n for r in mat):
        raise ParseError(f"'matrix' is not {n}x{n}")
    try:
        raw = [[_number(v) for v in row] for row in mat]
    except (ValueError, ZeroDivisionError):
        raise ParseError("non-numeric matrix entry") from None
    order = label_order(obj.get("labels"), n)
    returned_order = order
    if order is not None and order != list(range(n)):   # model listed the criteria in its own order
        inv = [order.index(k) for k in range(n)]
        raw = [[raw[inv[i]][inv[j]] for j in range(n)] for i in range(n)]
    upper = [raw[i][j] for i in range(n) for j in range(i + 1, n)]
    if any(v <= 0 for v in upper):
        raise ParseError("non-positive upper-triangle entry")
    reciprocal = all(abs(raw[i][i] - 1) <= tol for i in range(n)) and all(
        abs(raw[i][j] * raw[j][i] - 1) <= tol for i in range(n) for j in range(i + 1, n))
    scale = [ahp.scale_value(s) for s in ahp.SCALE]
    return {
        "labels": obj.get("labels"),
        "label_order": returned_order,            # canonical index of each returned label; None = unverified
        "labels_verified": returned_order is not None,
        "reordered": returned_order is not None and returned_order != list(range(n)),
        "raw_matrix": raw,
        "rationales": obj.get("rationales"),
        "reciprocal_returned": reciprocal,
        "upper_on_scale": all(any(abs(v - s) <= 1e-3 for s in scale) for v in upper),
        "matrix": ahp.reciprocal_from_upper(raw).tolist(),
        "bare_fractions_quoted": quoted,
    }
