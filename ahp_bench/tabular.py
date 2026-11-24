"""Robust reading of delimited text tables (standard library only).

read_table(path) returns a Table with the header, the rows as lists of strings (empty string = missing),
and how the file was read: encoding (utf-8-sig, utf-8 or cp1254, then latin-1 as a last resort), delimiter
(comma, semicolon, tab or pipe, chosen by field-count consistency) and the number of leading comment lines
(lines starting with "#") and malformed rows (padded or truncated to the header width).

Numbers: number(text, decimal) parses "1234.5", "1,234.5", "1234,5", "1.234,5", "12 %" and treats missing
tokens (empty, NA, N/A, null, none, nan, -, ?) as None. column_decimal() decides per column whether the
decimal mark is "." or ",".
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

ENCODINGS = ("utf-8", "cp1254")
DELIMITERS = (",", ";", "\t", "|")
MISSING = {"", "na", "n/a", "null", "none", "nan", "-", "?", "#n/a"}
_DATE_PATTERNS = (
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$"), (1, 2, 3)),
    (re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})(?: \d{1,2}:\d{2}(?::\d{2})?)?$"), (3, 2, 1)),
    (re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})(?: \d{1,2}:\d{2}(?::\d{2})?)?$"), (3, 2, 1)),
)


class TableError(ValueError):
    pass


@dataclass
class Table:
    header: list
    rows: list
    encoding: str
    delimiter: str
    comment_lines: int = 0
    malformed_rows: int = 0
    sha256: str = ""
    name: str = ""
    _decimal: dict = field(default_factory=dict)

    def index(self, column: str) -> int:
        try:
            return self.header.index(column)
        except ValueError:
            raise TableError(f"column {column!r} not found; columns: {', '.join(self.header)}") from None

    def column(self, name: str) -> list:
        i = self.index(name)
        return [r[i] for r in self.rows]

    def decimal(self, name: str) -> str:
        if name not in self._decimal:
            self._decimal[name] = column_decimal(self.column(name))
        return self._decimal[name]

    def numbers(self, name: str) -> list:
        """Column values as floats (None for missing or non-numeric cells)."""
        dec = self.decimal(name)
        return [number(v, dec) for v in self.column(name)]


def is_missing(text) -> bool:
    return text is None or str(text).strip().lower() in MISSING


def _decode(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8"), "utf-8-sig"
    for enc in ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"


def _field_counts(lines, delim) -> list:
    try:
        return [len(r) for r in csv.reader(lines, delimiter=delim) if r]
    except csv.Error:
        return []


def sniff_delimiter(lines) -> str:
    """Delimiter whose field count is most consistent with the header (header width > 1 required)."""
    sample = lines[:200]
    best, best_score = ",", (-1.0, 0)
    for d in DELIMITERS:
        counts = _field_counts(sample, d)
        if not counts or counts[0] < 2:
            continue
        score = (sum(1 for c in counts if c == counts[0]) / len(counts), counts[0])
        if score > best_score:
            best, best_score = d, score
    return best


def read_table(path) -> Table:
    path = Path(path)
    data = path.read_bytes()
    text, enc = _decode(data)
    lines = text.splitlines()
    comments = 0
    while comments < len(lines) and (lines[comments].lstrip().startswith("#") or not lines[comments].strip()):
        comments += 1
    body = lines[comments:]
    if not body:
        raise TableError(f"{path.name}: no header row")
    delim = sniff_delimiter(body)
    reader = csv.reader(io.StringIO("\n".join(body)), delimiter=delim)
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        raise TableError(f"{path.name}: no header row")
    header = [h.strip() for h in rows[0]]
    seen = {}
    for k, h in enumerate(header):  # empty or duplicate header cells get unique names
        base = h or f"column_{k + 1}"
        name, n = base, 1
        while name in seen:
            n += 1
            name = f"{base}_{n}"
        seen[name] = k
        header[k] = name
    width, out, bad = len(header), [], 0
    for r in rows[1:]:
        if len(r) != width:
            bad += 1
            r = (r + [""] * width)[:width]
        out.append([c.strip() for c in r])
    return Table(header, out, enc, delim, comments, bad, hashlib.sha256(data).hexdigest(), path.name)


def number(text, decimal: str = ".") -> float | None:
    """Parse one cell; None for missing or non-numeric cells."""
    if is_missing(text):
        return None
    s = str(text).strip().replace(" ", "").replace(" ", "")
    if s.endswith("%"):
        s = s[:-1]
    thousands = "," if decimal == "." else "."
    if thousands in s:
        if not re.fullmatch(rf"[+-]?\d{{1,3}}(\{thousands}\d{{3}})+(\{decimal}\d+)?", s):
            return None
        s = s.replace(thousands, "")
    s = s.replace(decimal, ".")
    if not re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", s):
        return None
    return float(s)


def column_decimal(values) -> str:
    """'.' or ',' as the decimal mark of a column: the one that parses more cells; on a tie, ',' only when a
    comma is followed by other than exactly three digits (1,5 or 12,75) and no value uses '.' as a decimal."""
    vals = [v for v in values if not is_missing(v)]
    dot = sum(number(v, ".") is not None for v in vals)
    com = sum(number(v, ",") is not None for v in vals)
    if com > dot:
        return ","
    if com == dot and any(re.search(r",\d{1,2}$|,\d{4,}$", v) for v in vals) \
            and not any(re.search(r"\.\d+$", v) and "," not in v for v in vals):
        return ","
    return "."


def date_ordinal(text) -> int | None:
    """Day number of an ISO (yyyy-mm-dd), dd.mm.yyyy or dd/mm/yyyy date (optional time ignored)."""
    import datetime as dt
    if is_missing(text):
        return None
    s = str(text).strip()
    for pat, (y, m, d) in _DATE_PATTERNS:
        g = pat.match(s)
        if g:
            try:
                return dt.date(int(g.group(y)), int(g.group(m)), int(g.group(d))).toordinal()
            except ValueError:
                return None
    return None
