"""Read-only audit of the stored experiment records.

CLI: python -m ahp_bench.audit --results STUDY/results [--out DIR]
     python -m ahp_bench.audit --problem PATH [--results DIR] [--out DIR] [--tag main] [--revision-tag revision]

Writes <out>/audit/audit.md and CSV files under <out>/audit/ (default out: <results>/report). No network
access. The files in <results>/raw are only read; .env is never opened (the secret scan walks the results
folder and the data/ and prompts/ folders next to it only).

With the study problem (the default --problem) this is the study's audit (fixed design: 9 models x 3
conditions x 2 tasks x 5 repetitions, revision tag revision_final); the study's records are published in
a separate repository and --results must point to its results folder. For any other problem the same
checks run on the records found there: completeness of the observed design, field completeness,
prompt integrity against the messages the code builds from the problem file, recomputation,
re-parse, secret scan of the results folder, checksums and, when revision records exist, the
revision protocol rules.

Sections
  a inventory per file and run_tag          f parsing flags, parser versions, backups
  b completeness of the main design          g recomputation and re-parse check
  c field completeness                       h secret scan
  d prompt integrity                         i checksums of results/raw
  e applied settings per model               j revision phase (revision tag)
  k parameter tests (param_tests.jsonl)      l spending per file and run_tag
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from . import ahp, prompts
from . import analysis as an
from .config import MODELS, TEMPERATURE
from .prompts import condition_label as L

RAW_FILES = ("calls.jsonl", "revisions.jsonl", "judge_calls.jsonl", "param_tests.jsonl")
BACKUP_FILES = ("calls_backup_before_reparse.jsonl", "calls.jsonl.bak")
MAIN_TAG = "main"
REVISION_TAG = "revision_final"
MAIN_REPS = 5
TEMP_REPS = 3
SCAN_DIRS = ("results", "data", "prompts")  # results folder (by its own name) and its siblings
# Patterns are assembled so that this source file does not itself contain them.
KEY_PREFIX = b"sk-" + b"or-"
BEARER = b"Bear" + b"er "
KEY_LIKE = re.compile(re.escape(KEY_PREFIX) + rb"v\d+-[A-Za-z0-9]{20,}")
BEARER_LIKE = re.compile(re.escape(BEARER) + rb"\s*[A-Za-z0-9_\-.]{20,}")
TOL = 1e-9


# ---------------------------------------------------------------- loading

def load_jsonl(path) -> tuple[list[dict], int]:
    return an.load_records(path)


def load_all(raw_dir: Path) -> dict:
    """file name -> (records, unreadable lines) for the raw and backup files that exist."""
    out = {}
    for name in RAW_FILES + BACKUP_FILES:
        p = Path(raw_dir) / name
        if p.exists():
            out[name] = load_jsonl(p)
    return out


def _tag(r: dict, fname: str) -> str:
    if fname == "param_tests.jsonl":
        return "(param test)"
    return str(r.get("run_tag"))


def _ts(r):
    v = r.get("timestamp_utc")
    if not isinstance(v, str):
        return None
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return None


def _num(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _usage(r: dict, k: str):
    if "usage" not in r and k == "reasoning_tokens":
        return r.get("reasoning_tokens")
    return (r.get("usage") or {}).get(k)


# ---------------------------------------------------------------- a inventory

def inventory(files: dict) -> list[dict]:
    rows = []
    for fname, (recs, bad) in files.items():
        groups = defaultdict(list)
        for r in recs:
            groups[_tag(r, fname)].append(r)
        for tag in sorted(groups):
            rs = groups[tag]
            if fname == "param_tests.jsonl":
                ok = [r.get("status") == 200 for r in rs]
                parsed = None
            else:
                ok = [r.get("ok") is True for r in rs]
                key = "judge_parse_ok" if fname == "judge_calls.jsonl" else "parse_ok"
                parsed = sum(1 for r in rs if r.get(key))
            ts = [t for t in map(_ts, rs) if t is not None]
            rows.append({
                "file": fname, "run_tag": tag, "records": len(rs), "calls_ok": sum(ok),
                "calls_failed": len(rs) - sum(ok), "parse_ok": parsed,
                "first_utc": min(ts).isoformat(timespec="seconds") if ts else "",
                "last_utc": max(ts).isoformat(timespec="seconds") if ts else "",
                "cost_usd": sum(_num(r.get("cost")) for r in rs),
                "prompt_tokens": int(sum(_num(_usage(r, "prompt_tokens")) for r in rs)),
                "completion_tokens": int(sum(_num(_usage(r, "completion_tokens")) for r in rs)),
                "reasoning_tokens": int(sum(_num(_usage(r, "reasoning_tokens")) for r in rs)),
                "unreadable_lines": bad, "backup": fname in BACKUP_FILES,
            })
    return rows


# ---------------------------------------------------------------- b completeness

def completeness(records, tag, models, conditions, tasks, reps) -> dict:
    expected = [(m, c, t, r) for m in models for c in conditions for t in tasks for r in range(1, reps + 1)]
    tagged = [r for r in records if r.get("run_tag") == tag]
    cnt = Counter(an.record_key(r) for r in tagged)
    ok_cnt = Counter(an.record_key(r) for r in tagged if r.get("parse_ok"))
    exp = set(expected)
    return {"tag": tag, "expected": len(expected), "records": len(tagged), "cells_present": len(exp & set(cnt)),
            "missing": [k for k in expected if k not in cnt],
            "duplicated": sorted((k, n) for k, n in cnt.items() if n > 1),
            "unexpected": sorted(k for k in cnt if k not in exp),
            "cells_without_parse_ok": [k for k in expected if k in cnt and not ok_cnt.get(k)]}


# ---------------------------------------------------------------- c field completeness

def _nonempty_str(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _messages_ok(v) -> bool:
    return isinstance(v, list) and bool(v) and all(
        isinstance(m, dict) and _nonempty_str(m.get("role")) and _nonempty_str(m.get("content")) for m in v)


FIELD_CHECKS = {
    "messages": lambda r: _messages_ok(r.get("messages")),
    "raw_reply": lambda r: _nonempty_str(r.get("raw_reply")),
    "requested_params": lambda r: isinstance(r.get("requested_params"), dict) and bool(r["requested_params"]),
    "applied_params": lambda r: isinstance(r.get("applied_params"), dict) and bool(r["applied_params"]),
    # host-mode records (source "host": answered by the coding agent's subagents) have no token counts
    "usage_tokens": lambda r: r.get("source") == "host" or all(isinstance(_usage(r, k), int)
                                                                for k in ("prompt_tokens", "completion_tokens")),
    "cost": lambda r: isinstance(r.get("cost"), (int, float)) and not isinstance(r.get("cost"), bool),
    "provider": lambda r: _nonempty_str(r.get("provider")),
    "generation_id": lambda r: _nonempty_str(r.get("generation_id")),
    "timestamp_utc": lambda r: _ts(r) is not None,
    "prompt_id": lambda r: _nonempty_str(r.get("prompt_id")),
    "prompt_sha256": lambda r: _nonempty_str(r.get("prompt_sha256")),
}
INFO_CHECKS = {"reasoning_tokens": lambda r: isinstance(_usage(r, "reasoning_tokens"), int)}
NO_PROMPT_ID = ("prompt_id", "prompt_sha256")  # judge calls carry no prompt id/hash


def result_sets(files: dict, revision_tag=REVISION_TAG) -> list[tuple[str, str, list[dict]]]:
    """(file, tag, records) for the result tags: main, temp_*, revision tag (panel and judge)."""
    out = []
    calls = files.get("calls.jsonl", ([], 0))[0]
    tags = [MAIN_TAG] + sorted({r["run_tag"] for r in calls if str(r.get("run_tag", "")).startswith("temp_")})
    for t in tags:
        out.append(("calls.jsonl", t, [r for r in calls if r.get("run_tag") == t]))
    for f in ("revisions.jsonl", "judge_calls.jsonl"):
        out.append((f, revision_tag, [r for r in files.get(f, ([], 0))[0] if r.get("run_tag") == revision_tag]))
    return out


def _ident(fname: str, r: dict) -> str:
    parts = [fname, str(r.get("run_tag")), str(r.get("chain_model_key") or r.get("model_key")),
             str(r.get("condition")), str(r.get("task")), f"rep{r.get('rep')}"]
    if r.get("revision_number") is not None:
        parts.append(f"rev{r.get('revision_number')}")
    if r.get("purpose"):
        parts.append(str(r["purpose"]))
    return " ".join(parts)


def field_completeness(sets) -> tuple[list[dict], list[dict]]:
    rows, lacking = [], []
    for fname, tag, recs in sets:
        checks = {k: f for k, f in FIELD_CHECKS.items() if not (fname == "judge_calls.jsonl" and k in NO_PROMPT_ID)}
        row = {"file": fname, "run_tag": tag, "records": len(recs)}
        for k in FIELD_CHECKS:
            row[k] = (sum(1 for r in recs if checks[k](r)) if k in checks else None)
        for k, f in INFO_CHECKS.items():
            row[k] = sum(1 for r in recs if f(r))
        row["records_lacking_any"] = 0
        for r in recs:
            miss = [k for k, f in checks.items() if not f(r)]
            if miss:
                row["records_lacking_any"] += 1
                lacking.append({"record": _ident(fname, r), "missing": ";".join(miss),
                                "ok": r.get("ok"), "error": str(r.get("error") or "")[:120]})
        rows.append(row)
    return rows, lacking


# ---------------------------------------------------------------- d prompt integrity

def sha(messages) -> str:
    return prompts.prompt_sha256(messages)


def prompt_integrity(calls, tags, problem=None) -> list[dict]:
    rows = []
    for tag in tags:
        groups = defaultdict(list)
        for r in calls:
            if r.get("run_tag") == tag:
                groups[(r.get("condition"), r.get("task"))].append(r)
        for (c, t), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            try:
                current = prompts.build_messages(c, t, problem)
            except (ValueError, KeyError):
                current = None
            rows.append({"run_tag": tag, "condition": c, "task": t, "records": len(rs),
                         "distinct_prompt_sha256": len({r.get("prompt_sha256") for r in rs}),
                         "distinct_prompt_id": len({r.get("prompt_id") for r in rs}),
                         "stored_hash_matches_stored_messages": sum(1 for r in rs if r.get("prompt_sha256") == sha(r.get("messages"))),
                         "messages_equal_current_code": sum(1 for r in rs if r.get("messages") == current),
                         "stored_hash_equals_current_code_hash": sum(1 for r in rs if r.get("prompt_sha256") == sha(current))})
    return rows


def guided_file_check(calls, tag=MAIN_TAG, guided_dir=None) -> list[dict]:
    """Stored guided messages vs the files in prompts/guided, read as raw bytes and transformed
    exactly as prompts.py does (system: UTF-8, CRLF -> LF; user: the file text, which must equal
    json.dumps(json.loads(file), ensure_ascii=False))."""
    gdir = Path(guided_dir or prompts.GUIDED_DIR)
    sys_bytes = (gdir / prompts.GUIDED_FILES["system"]).read_bytes()
    sys_text = sys_bytes.decode("utf-8").replace("\r\n", "\n")
    rows = []
    for t in prompts.TASKS:
        ub = (gdir / prompts.GUIDED_FILES[t]).read_bytes()
        file_text = ub.decode("utf-8")
        rebuilt = json.dumps(json.loads(file_text), ensure_ascii=False)
        rs = [r for r in calls if r.get("run_tag") == tag and r.get("condition") == "guided" and r.get("task") == t]
        rows.append({"task": t, "records": len(rs),
                     "system_file_sha256": hashlib.sha256(sys_bytes).hexdigest(),
                     "system_file_has_crlf": b"\r\n" in sys_bytes,
                     "user_file_sha256": hashlib.sha256(ub).hexdigest(),
                     "user_file_equals_json_dumps_of_itself": file_text == rebuilt,
                     "system_message_equals_file": sum(1 for r in rs if (r.get("messages") or [{}])[0].get("content") == sys_text),
                     "user_message_equals_file_bytes": sum(1 for r in rs if len(r.get("messages") or []) > 1
                                                           and r["messages"][1].get("content") == file_text),
                     "two_messages_only": sum(1 for r in rs if len(r.get("messages") or []) == 2)})
    return rows


# ---------------------------------------------------------------- e applied settings

def _label(v) -> str:
    if v is None:
        return "omitted"
    if isinstance(v, dict):
        return ",".join(f"{k}={v[k]}" for k in sorted(v))
    return str(v)


def _counter_text(c: Counter) -> str:
    return "; ".join(f"{k} ({n})" for k, n in sorted(c.items(), key=lambda x: (-x[1], str(x[0]))))


def applied_settings(sets) -> list[dict]:
    rows = []
    for fname, tag, recs in sets:
        groups = defaultdict(list)
        for r in recs:
            groups[r.get("model_key")].append(r)
        for m in sorted(groups, key=str):
            rs = groups[m]
            rq = [r.get("requested_params") or {} for r in rs]
            ap = [r.get("applied_params") or {} for r in rs]
            rt = [_usage(r, "reasoning_tokens") for r in rs]
            rt = [x for x in rt if isinstance(x, (int, float))]
            rows.append({
                "file": fname, "run_tag": tag, "model": m, "records": len(rs),
                "temperature_requested": _counter_text(Counter(str(p.get("temperature")) for p in rq)),
                "temperature_applied": _counter_text(Counter(str(p.get("temperature")) if "temperature" in p and p.get("temperature") is not None else "omitted" for p in ap)),
                "reasoning_requested": _counter_text(Counter(_label(p.get("reasoning")) for p in rq)),
                "reasoning_applied": _counter_text(Counter(_label(p.get("reasoning")) for p in ap)),
                "omitted": _counter_text(Counter(",".join(p.get("omitted") or []) or "none" for p in ap)),
                "calls_with_parameter_fallback": sum(1 for r in rs if len(r.get("http_attempts") or []) > 1
                                                     and len({json.dumps(a.get("params"), sort_keys=True)
                                                              for a in r["http_attempts"]}) > 1),
                "calls_with_retries": sum(1 for r in rs if len(r.get("http_attempts") or []) > 1),
                "mean_reasoning_tokens": float(np.mean(rt)) if rt else None,
                "providers": _counter_text(Counter(str(r.get("provider")) for r in rs)),
                "finish_reasons": _counter_text(Counter(str(r.get("finish_reason")) for r in rs)),
            })
    return rows


# ---------------------------------------------------------------- f parsing

def parsing(files: dict) -> tuple[list[dict], list[dict]]:
    versions, flags = [], []
    for fname in ("calls.jsonl", "revisions.jsonl") + BACKUP_FILES:
        recs = files.get(fname, ([], 0))[0]
        c = Counter((str(r.get("run_tag")), str(r.get("parser_version", "absent"))) for r in recs)
        for (tag, v), n in sorted(c.items()):
            versions.append({"file": fname, "run_tag": tag, "parser_version": v, "records": n})
    for fname in ("calls.jsonl", "revisions.jsonl"):
        recs = files.get(fname, ([], 0))[0]
        groups = defaultdict(list)
        for r in recs:
            groups[(str(r.get("run_tag")), r.get("condition"), r.get("task"))].append(r)
        for (tag, c, t), rs in sorted(groups.items(), key=str):
            row = {"file": fname, "run_tag": tag, "condition": c, "task": t, "records": len(rs),
                   "parse_ok": sum(1 for r in rs if r.get("parse_ok")),
                   "bare_fractions_quoted": sum(1 for r in rs if r.get("bare_fractions_quoted"))}
            if c == "guided":
                row.update(labels_verified=sum(1 for r in rs if r.get("labels_verified") is True),
                           labels_not_verified=sum(1 for r in rs if r.get("parse_ok") and r.get("labels_verified") is not True),
                           reordered=sum(1 for r in rs if r.get("reordered") is True),
                           reciprocal_as_returned=sum(1 for r in rs if r.get("reciprocal_returned") is True),
                           not_reciprocal_as_returned=sum(1 for r in rs if r.get("parse_ok") and r.get("reciprocal_returned") is not True),
                           upper_on_scale=sum(1 for r in rs if r.get("upper_on_scale") is True))
            flags.append(row)
    return versions, flags


def backup_comparison(files: dict) -> list[dict]:
    """Raw fields of calls.jsonl vs each backup, matched by generation_id (fallback: key + timestamp)."""
    cur = files.get("calls.jsonl", ([], 0))[0]
    raw_fields = ("raw_reply", "messages", "usage", "cost", "provider", "timestamp_utc", "requested_params",
                  "applied_params")
    def k(r):
        return r.get("generation_id") or (an.record_key(r), r.get("timestamp_utc"))
    idx = {}
    for r in cur:
        idx.setdefault(k(r), r)
    rows = []
    for fname in BACKUP_FILES:
        if fname not in files:
            continue
        recs = files[fname][0]
        matched = [(b, idx.get(k(b))) for b in recs]
        found = [(b, c) for b, c in matched if c is not None]
        rows.append({"backup": fname, "backup_records": len(recs), "matched_in_calls": len(found),
                     "raw_fields_identical": sum(1 for b, c in found if all(b.get(f) == c.get(f) for f in raw_fields)),
                     "parse_ok_backup": sum(1 for b in recs if b.get("parse_ok")),
                     "parse_ok_now_same_records": sum(1 for _, c in found if c.get("parse_ok"))})
    return rows


# ---------------------------------------------------------------- g recomputation

def _weights_vec(r: dict, names) -> np.ndarray | None:
    w = r.get("weights")
    if isinstance(w, dict) and all(n in w for n in names):
        return np.array([float(w[n]) for n in names])
    if isinstance(w, list) and len(w) == len(names):
        return np.array(w, dtype=float)
    return None


def recompute_record(r: dict, problem=None) -> dict:
    """Max abs differences between stored and recomputed matrix / weights / lambda_max / CI / CR."""
    task = r.get("task")
    p = prompts.STUDY if problem is None else problem
    names, pairs = p.criteria(task), p.pairs(task)
    if r.get("condition") == "guided":
        m = ahp.reciprocal_from_upper(r["raw_matrix"])
    else:
        m = ahp.matrix_from_answers(len(names), pairs, r["answers"])
    c = ahp.consistency(m)
    w = _weights_vec(r, names)
    return {"matrix": float(np.abs(m - np.asarray(r["matrix"], float)).max()),
            "weights": float(np.abs(c["weights"] - w).max()) if w is not None else float("inf"),
            "lambda_max": abs(c["lambda_max"] - float(r["lambda_max"])),
            "CI": abs(c["CI"] - float(r["CI"])), "CR": abs(c["CR"] - float(r["CR"]))}


def recompute(files: dict, problem=None) -> tuple[list[dict], list[str]]:
    rows, failures = [], []
    for fname in ("calls.jsonl", "revisions.jsonl"):
        groups = defaultdict(list)
        for r in files.get(fname, ([], 0))[0]:
            if r.get("parse_ok"):
                groups[str(r.get("run_tag"))].append(r)
        for tag in sorted(groups):
            mx = Counter()
            n = 0
            for r in groups[tag]:
                try:
                    d = recompute_record(r, problem)
                except Exception as e:  # noqa: BLE001 - reported, not raised
                    failures.append(f"{_ident(fname, r)}: {type(e).__name__}")
                    continue
                n += 1
                for k, v in d.items():
                    mx[k] = max(mx[k], v)
            rows.append({"file": fname, "run_tag": tag, "parse_ok_records": len(groups[tag]), "recomputed": n,
                         **{f"max_abs_diff_{k}": mx.get(k, 0.0) for k in ("matrix", "weights", "lambda_max", "CI", "CR")}})
    return rows, failures


def reparse_check(files: dict, problem=None) -> list[dict]:
    """Re-parse every stored raw reply with the current parser (run.analyse) and compare with the
    stored derived fields. Nothing is written."""
    from . import run
    from .parse import ParseError
    rows = []
    for fname in ("calls.jsonl", "revisions.jsonl"):
        groups = defaultdict(list)
        for r in files.get(fname, ([], 0))[0]:
            if r.get("ok") is not False and r.get("raw_reply") is not None:
                groups[str(r.get("run_tag"))].append(r)
        for tag in sorted(groups):
            same_ok = same_ans = diff_flags = 0
            max_cr = max_mat = 0.0
            for r in groups[tag]:
                try:
                    a = run.analyse(r["condition"], r["task"], r["raw_reply"], problem)
                    ok = True
                except (ParseError, ValueError):
                    a, ok = {}, False
                same_ok += ok == bool(r.get("parse_ok"))
                if ok and r.get("parse_ok"):
                    same_ans += a.get("answers") == r.get("answers") or (r.get("condition") == "guided"
                                                                         and a.get("raw_matrix") == r.get("raw_matrix"))
                    max_mat = max(max_mat, float(np.abs(np.asarray(a["matrix"]) - np.asarray(r["matrix"], float)).max()))
                    max_cr = max(max_cr, abs(a["CR"] - float(r["CR"])))
                    diff_flags += any(a.get(f) != r.get(f) for f in ("bare_fractions_quoted", "labels_verified",
                                                                       "reordered", "reciprocal_returned")
                                      if f in a or f in r)
            rows.append({"file": fname, "run_tag": tag, "records_with_reply": len(groups[tag]),
                         "parse_ok_agrees": same_ok, "answers_or_raw_matrix_equal": same_ans,
                         "records_with_flag_difference": diff_flags, "max_abs_diff_matrix": max_mat,
                         "max_abs_diff_CR": max_cr})
    return rows


# ---------------------------------------------------------------- h secret scan

def secret_scan(root: Path, dirs=SCAN_DIRS) -> list[dict]:
    rows = []
    for d in dirs:
        base = Path(root) / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts or p.name.startswith(".env"):
                continue
            data = p.read_bytes()
            rows.append({"file": p.relative_to(root).as_posix(), "bytes": len(data),
                         "key_prefix_count": data.count(KEY_PREFIX), "key_like_count": len(KEY_LIKE.findall(data)),
                         "bearer_count": data.count(BEARER), "bearer_token_like_count": len(BEARER_LIKE.findall(data))})
    return rows


# ---------------------------------------------------------------- i checksums

def checksums(raw_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(Path(raw_dir).iterdir()):
        if p.is_file():
            data = p.read_bytes()
            rows.append({"file": p.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                         "lines": data.count(b"\n")})
    return rows


# ---------------------------------------------------------------- j revision phase

def revision_audit(calls, revs, judges, tag=REVISION_TAG, source_tag=MAIN_TAG, max_revisions=5,
                   temperature=TEMPERATURE, problem=None) -> dict:
    """Revision protocol rules. temperature=None compares each record's requested temperature with its
    own temperature_requested field instead of a fixed value."""
    from . import revise
    rv = [r for r in revs if r.get("run_tag") == tag and r.get("source_tag") == source_tag]
    jc = [r for r in judges if r.get("run_tag") == tag and r.get("source_tag") == source_tag]
    sources = revise.source_matrices(calls, source_tag, revise.REVISION_CONDITIONS, problem=problem)
    if problem is not None:  # only chains of the models and conditions that were revised
        revised = {(r.get("model_key"), r.get("condition")) for r in rv}
        sources = {k: v for k, v in sources.items() if (k[0], k[1]) in revised}
    need = {k for k, s in sources.items() if not an.is_consistent(revise._cr(s))}
    chains = revise.revision_records(rv, tag, source_tag)
    jidx = defaultdict(list)
    for j in jc:
        jidx[(tuple(j.get("chain_key") or []), int(j.get("revision_number") or 0))].append(j)
    violations = []

    def bad(key, number, rule):
        violations.append({"chain": " ".join(map(str, key)), "revision": number, "rule": rule})

    n_judge_expected = Counter()
    for key, recs in sorted(chains.items()):
        src = sources.get(key)
        if src is None:
            bad(key, "", "no source record in source tag")
            continue
        if key not in need:
            bad(key, "", "source matrix was already consistent")
        nums = [int(r.get("revision_number") or 0) for r in recs]
        if nums != list(range(1, len(recs) + 1)):
            bad(key, "", f"revision numbers not 1..n: {nums}")
        max_rev = max(int(r.get("max_revisions") or max_revisions) for r in recs) if problem is not None else max_revisions
        if len(recs) > max_rev:
            bad(key, "", f"more than {max_rev} revisions")
        cons = [i for i, r in enumerate(recs) if r.get("parse_ok") and an.is_consistent(r["CR"])]
        if cons and cons[0] != len(recs) - 1:
            bad(key, recs[cons[0]].get("revision_number"), "calls continued after a consistent revision")
        if not cons and len(recs) < max_rev:
            bad(key, "", "chain stopped before the limit without reaching consistency")
        for i, r in enumerate(recs):
            no = r.get("revision_number")
            if abs(float(r.get("source_CR", -1)) - revise._cr(src)) > TOL:
                bad(key, no, "source_CR differs from the source matrix CR")
            best, based, cr = revise.best_so_far(src, recs[:i], problem)
            if abs(float(r.get("based_on_CR", -1)) - cr) > TOL or list(r.get("based_on_answers") or []) != list(best["answers"]):
                bad(key, no, "based_on answers/CR differ from best-so-far rule")
            if r.get("based_on_revision") is not None and int(r["based_on_revision"]) != based:
                bad(key, no, "based_on_revision differs from best-so-far rule")
            if r.get("prompt_sha256") != sha(r.get("messages")):
                bad(key, no, "prompt_sha256 does not match stored messages")
            if (r.get("requested_params") or {}).get("temperature") != (
                    r.get("temperature_requested") if temperature is None else temperature):
                bad(key, no, "requested temperature differs")
            msgs = r.get("messages") or []
            reused = bool(r.get("judge_feedback_reused"))
            if reused:
                if i == 0 or msgs != recs[i - 1].get("messages"):
                    bad(key, no, "feedback reused but conversation differs from previous call")
            else:
                expect_prefix = list(src["messages"])
                if msgs[:len(expect_prefix)] != expect_prefix or len(msgs) != len(expect_prefix) + 2 \
                        or msgs[-2].get("role") != "assistant" or msgs[-1].get("role") != "user":
                    bad(key, no, "conversation is not original messages + one assistant turn + one user turn")
                else:
                    if r.get("judge_feedback_ok"):
                        expect = revise.judge_revision_message(r.get("judge_feedback") or "", src["task"], best["answers"],
                                                               problem)
                    else:
                        expect = revise.revision_prompt(src["task"], best["answers"], cr, problem)
                    if msgs[-1].get("content") != expect:
                        bad(key, no, "last user message differs from the code-built revision message")
                    reply = (revise.serialise_answers(src["condition"], best) if based
                             else src.get("raw_reply") or src.get("content"))
                    if msgs[-2].get("content") != reply:
                        bad(key, no, "assistant turn differs from the best-so-far answers")
            js = jidx.get((tuple(key), int(no or 0)), [])
            nf = sum(1 for j in js if j.get("purpose") == "feedback")
            nv = sum(1 for j in js if j.get("purpose") == "verdict")
            if problem is not None and "judge_model_id" not in r:
                continue  # revision without the judge (other problems); the study always used the judge
            ef, ev = (0 if reused else 1), (1 if r.get("parse_ok") else 0)
            n_judge_expected["feedback"] += ef
            n_judge_expected["verdict"] += ev
            if nf != ef or nv != ev:
                bad(key, no, f"judge calls feedback={nf} verdict={nv}, expected {ef}/{ev}")
            if r.get("parse_ok") and (r.get("judge_verdict_ok") is True) != (r.get("judge_verdict") is not None):
                bad(key, no, "judge_verdict_ok inconsistent with judge_verdict")
    rev_keys = {(tuple(k), n) for k, rs in chains.items() for n in (int(r.get("revision_number") or 0) for r in rs)}
    orphan = [j for j in jc if (tuple(j.get("chain_key") or []), int(j.get("revision_number") or 0)) not in rev_keys]
    not_reused = [r for r in rv if not r.get("judge_feedback_reused")]
    return {
        "tag": tag, "source_tag": source_tag, "judge_models": sorted({str(r.get("judge_model_id")) for r in rv}),
        "judge_model_ids_in_calls": sorted({str(j.get("model_id")) for j in jc}),
        "chains_expected": len(need), "chains_present": len(chains),
        "chains_missing": sorted(" ".join(map(str, k)) for k in need - set(chains)),
        "calls": len(rv), "calls_ok": sum(1 for r in rv if r.get("ok")), "parse_ok": sum(1 for r in rv if r.get("parse_ok")),
        "judge_calls": len(jc), "judge_calls_by_purpose": dict(Counter(str(j.get("purpose")) for j in jc)),
        "judge_calls_ok": sum(1 for j in jc if j.get("ok")), "judge_parse_ok": sum(1 for j in jc if j.get("judge_parse_ok")),
        "judge_calls_expected": dict(n_judge_expected), "orphan_judge_calls": len(orphan),
        "feedback_new": len(not_reused), "feedback_reused": len(rv) - len(not_reused),
        "feedback_fallback_code_prompt": sum(1 for r in not_reused if not r.get("judge_feedback_ok")),
        "verdict_valid": sum(1 for r in rv if r.get("judge_verdict_ok") is True),
        "verdict_invalid": sum(1 for r in rv if r.get("judge_verdict_ok") is False),
        "verdict_not_requested": sum(1 for r in rv if r.get("judge_verdict_ok") is None),
        "verdict_counts": dict(Counter((r.get("judge_verdict") or {}).get("logic_preserved") for r in rv
                                       if isinstance(r.get("judge_verdict"), dict))),
        "chains_reached": sum(1 for rs in chains.values() if any(r.get("parse_ok") and an.is_consistent(r["CR"]) for r in rs)),
        "cost_panel": sum(_num(r.get("cost")) for r in rv), "cost_judge": sum(_num(j.get("cost")) for j in jc),
        "cost_judge_from_revision_records": sum(_num(r.get("judge_cost")) for r in rv),
        "violations": violations,
    }


# ---------------------------------------------------------------- k parameter tests

def answer_set(content):
    from .parse import ParseError, parse_lean_cot
    try:
        return tuple(parse_lean_cot(content, len(prompts.PAIRS["main"]))["answers"])
    except ParseError:
        return None


def param_tests(recs) -> list[dict]:
    groups = defaultdict(list)
    for r in recs:
        groups[(r.get("model_key"), json.dumps(r.get("extra"), sort_keys=True))].append(r)
    rows = []
    for (m, extra), rs in sorted(groups.items(), key=str):
        sets = [answer_set(r.get("content")) for r in rs if r.get("status") == 200]
        rows.append({"model": m, "extra_params": extra, "calls": len(rs),
                     "http_200": sum(1 for r in rs if r.get("status") == 200),
                     "parsed_answer_sets": sum(1 for s in sets if s is not None),
                     "distinct_answer_sets": len({s for s in sets if s is not None}),
                     "errors": _counter_text(Counter(json.loads(r["error"]).get("message", "") if isinstance(r.get("error"), str)
                                                     and r["error"].startswith("{") else str(r.get("error"))
                                                     for r in rs if r.get("status") != 200)),
                     "mean_reasoning_tokens": float(np.mean([r["reasoning_tokens"] for r in rs
                                                             if isinstance(r.get("reasoning_tokens"), (int, float))]))
                     if any(isinstance(r.get("reasoning_tokens"), (int, float)) for r in rs) else None,
                     "cost_usd": sum(_num(r.get("cost")) for r in rs)})
    return rows


# ---------------------------------------------------------------- run

def run_audit(results_dir: Path, revision_tag: str = REVISION_TAG) -> dict:
    """The study's audit of the records in <results_dir>/raw."""
    results_dir = Path(results_dir)
    raw = results_dir / "raw"
    scan_dirs = tuple(dict.fromkeys((results_dir.name,) + SCAN_DIRS[1:]))
    files = load_all(raw)
    calls = files.get("calls.jsonl", ([], 0))[0]
    temp_tags = sorted({r["run_tag"] for r in calls if str(r.get("run_tag", "")).startswith("temp_")})
    sets = result_sets(files, revision_tag)
    comp = [completeness(calls, MAIN_TAG, list(MODELS), prompts.CONDITIONS, prompts.TASKS, MAIN_REPS)]
    comp += [completeness(calls, t, list(MODELS), ("lean",), prompts.TASKS, TEMP_REPS) for t in temp_tags]
    fc, lacking = field_completeness(sets)
    versions, flags = parsing(files)
    rec_rows, rec_fail = recompute(files)
    return {
        "inventory": inventory(files), "completeness": comp, "fields": fc, "fields_lacking": lacking,
        "prompt_integrity": prompt_integrity(calls, [MAIN_TAG] + temp_tags),
        "guided_files": guided_file_check(calls),
        "revision_prompt_hash": [{"file": f, "run_tag": t, "records": len(rs),
                                  "stored_hash_matches_stored_messages": sum(1 for r in rs if r.get("prompt_sha256") == sha(r.get("messages")))}
                                 for f, t, rs in sets if f == "revisions.jsonl"],
        "settings": applied_settings(sets), "parser_versions": versions, "parse_flags": flags,
        "backups": backup_comparison(files), "recompute": rec_rows, "recompute_failures": rec_fail,
        "reparse": reparse_check(files), "secrets": secret_scan(results_dir.parent, scan_dirs),
        "scan_dirs": [d for d in scan_dirs if (results_dir.parent / d).is_dir()], "checksums": checksums(raw),
        "revision": revision_audit(calls, files.get("revisions.jsonl", ([], 0))[0],
                                   files.get("judge_calls.jsonl", ([], 0))[0], revision_tag),
        "param_tests": param_tests(files.get("param_tests.jsonl", ([], 0))[0]),
        "temp_tags": temp_tags, "revision_tag": revision_tag,
    }


def run_problem_audit(problem, results_dir: Path, tag: str = MAIN_TAG, revision_tag: str = "revision") -> dict:
    """Audit of the records of any problem in a results folder (same sections as the study audit;
    the design is the one observed in the records, guided checks are omitted)."""
    raw = Path(results_dir) / "raw"
    files = load_all(raw)
    calls = files.get("calls.jsonl", ([], 0))[0]
    tagged = [r for r in calls if r.get("run_tag") == tag]
    models = sorted({r.get("model_key") for r in tagged}, key=str)
    conds = [c for c in problem.conditions if any(r.get("condition") == c for r in tagged)]
    tasks = [t for t in problem.task_names if any(r.get("task") == t for r in tagged)]
    reps = max([int(r.get("rep") or 0) for r in tagged] or [0])
    revs = files.get("revisions.jsonl", ([], 0))[0]
    judges = files.get("judge_calls.jsonl", ([], 0))[0]
    has_rev = any(r.get("run_tag") == revision_tag for r in revs)
    sets = [("calls.jsonl", tag, tagged)]
    if has_rev:
        sets += [(f, revision_tag, [r for r in files.get(f, ([], 0))[0] if r.get("run_tag") == revision_tag])
                 for f in ("revisions.jsonl", "judge_calls.jsonl")]
        sets = [x for x in sets if x[2]]
    fc, lacking = field_completeness(sets)
    versions, flags = parsing(files)
    rec_rows, rec_fail = recompute(files, problem)
    from .cli import check_problem_matches
    return {
        "problem": problem.name, "problem_sha256": problem.sha256,
        "foreign_tasks": check_problem_matches(calls + revs, problem),
        "inventory": inventory(files),
        "completeness": [completeness(calls, tag, models, conds, tasks, reps)] if tagged else [],
        "fields": fc, "fields_lacking": lacking,
        "prompt_integrity": prompt_integrity(calls, [tag], problem), "guided_files": [],
        "revision_prompt_hash": [{"file": f, "run_tag": t, "records": len(rs),
                                  "stored_hash_matches_stored_messages": sum(1 for r in rs if r.get("prompt_sha256") == sha(r.get("messages")))}
                                 for f, t, rs in sets if f == "revisions.jsonl"],
        "settings": applied_settings(sets), "parser_versions": versions, "parse_flags": flags,
        "backups": [], "recompute": rec_rows, "recompute_failures": rec_fail,
        "reparse": reparse_check(files, problem), "secrets": secret_scan(Path(results_dir).parent, (Path(results_dir).name,)),
        "checksums": checksums(raw) if raw.exists() else [],
        "revision": (revision_audit(calls, revs, judges, revision_tag, tag, temperature=None, problem=problem)
                     if has_rev else None),
        "param_tests": [], "temp_tags": [], "revision_tag": revision_tag,
    }


def expectations(res: dict) -> list[dict]:
    """Checks with expected and observed values; status PASS/FAIL."""
    out = []

    def check(name, observed, expected, ok=None):
        out.append({"check": name, "expected": str(expected), "observed": str(observed),
                    "status": "PASS" if (observed == expected if ok is None else ok) else "FAIL"})

    for c in res["completeness"]:
        check(f"b {c['tag']}: cells present", c["cells_present"], c["expected"])
        check(f"b {c['tag']}: missing / duplicated / unexpected cells",
              (len(c["missing"]), len(c["duplicated"]), len(c["unexpected"])), (0, 0, 0))
        check(f"b {c['tag']}: cells without a parsed record", len(c["cells_without_parse_ok"]), 0)
    for f in res["fields"]:
        check(f"c {f['file']} {f['run_tag']}: records lacking a required field", f["records_lacking_any"], 0)
    if "foreign_tasks" in res:
        check("problem file matches the stored criteria", res["foreign_tasks"], [])
    for p in res["prompt_integrity"]:
        pre = f"d {p['run_tag']} {L(p['condition'])}/{p['task']}"
        check(f"{pre}: distinct prompt_sha256", p["distinct_prompt_sha256"], 1)
        check(f"{pre}: stored hash = sha256(stored messages)", p["stored_hash_matches_stored_messages"], p["records"])
        check(f"{pre}: stored messages = current code", p["messages_equal_current_code"], p["records"])
    for g in res["guided_files"]:
        check(f"d guided/{g['task']}: system message = file (CRLF->LF)", g["system_message_equals_file"], g["records"])
        check(f"d guided/{g['task']}: user message = file bytes", g["user_message_equals_file_bytes"], g["records"])
        check(f"d guided/{g['task']}: file = json.dumps(payload)", g["user_file_equals_json_dumps_of_itself"], True)
    for p in res["revision_prompt_hash"]:
        check(f"d {p['run_tag']} revisions: stored hash = sha256(stored messages)",
              p["stored_hash_matches_stored_messages"], p["records"])
    for r in res["recompute"]:
        worst = max(v for k, v in r.items() if k.startswith("max_abs_diff_"))
        check(f"g recompute {r['file']} {r['run_tag']}: max abs diff <= 1e-9", f"{worst:.3g}", "<= 1e-9", worst <= 1e-9)
    check("g recompute failures", len(res["recompute_failures"]), 0)
    for r in res["reparse"]:
        check(f"g re-parse {r['file']} {r['run_tag']}: parse_ok agrees", r["parse_ok_agrees"], r["records_with_reply"])
        check(f"g re-parse {r['file']} {r['run_tag']}: flag differences", r["records_with_flag_difference"], 0)
    keys = sum(s["key_like_count"] for s in res["secrets"])
    bearer_tok = sum(s["bearer_token_like_count"] for s in res["secrets"])
    check("h key-like strings (OpenRouter key format)", keys, 0)
    check("h bearer followed by a token-like string", bearer_tok, 0)
    for b in res["backups"]:
        check(f"f backup {b['backup']}: raw fields identical to calls.jsonl", b["raw_fields_identical"], b["backup_records"])
    rv = res["revision"]
    if rv is not None:
        check(f"j {rv['tag']}: chains present = chains needing revision", rv["chains_present"], rv["chains_expected"])
        check(f"j {rv['tag']}: protocol rule violations", len(rv["violations"]), 0)
        check(f"j {rv['tag']}: judge calls without a revision record", rv["orphan_judge_calls"], 0)
    return out


# ---------------------------------------------------------------- output

def _fmt(v) -> str:
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        return f"{v:.6g}" if abs(v) < 1e-3 and v != 0 else (f"{v:.4f}" if not v.is_integer() else f"{v:.1f}")
    if v is None:
        return ""
    return str(v)


def _keys(rows: list[dict]) -> list[str]:
    """Ordered union of the keys of all rows."""
    out = []
    for r in rows:
        out += [k for k in r if k not in out]
    return out


def _write_csv(path: Path, rows: list[dict], header=None) -> None:
    from .report import write_csv
    header = header or (_keys(rows) if rows else ["(no rows)"])
    write_csv(path, header, [[_fmt(r.get(h)) for h in header] for r in rows])


def _md_table(rows: list[dict], cols=None) -> str:
    if not rows:
        return "(no rows)"
    cols = cols or _keys(rows)
    esc = lambda s: _fmt(s).replace("|", "\\|")
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(esc(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def _labelled(rows: list[dict]) -> list[dict]:
    """Rows with the condition column shown by its display name (direct for the stored key lean) and the guidance and technique axes."""
    from .prompts import condition_guidance, condition_technique
    out = []
    for r in rows:
        if "condition" not in r:
            out.append(r)
            continue
        d = {}
        for k, v in r.items():
            if k == "condition":
                d.update(guidance=condition_guidance(v), technique=condition_technique(v), condition=L(v))
            else:
                d[k] = v
        out.append(d)
    return out


def write_outputs(res: dict, out_dir: Path) -> Path:
    from .report import write_text
    d = Path(out_dir) / "audit"
    exp = expectations(res)
    res = {**res, "prompt_integrity": _labelled(res["prompt_integrity"]), "parse_flags": _labelled(res["parse_flags"])}
    comp_rows = []
    for c in res["completeness"]:
        for k in c["missing"]:
            comp_rows.append({"run_tag": c["tag"], "issue": "missing", "cell": " ".join(map(str, k))})
        for k, n in c["duplicated"]:
            comp_rows.append({"run_tag": c["tag"], "issue": f"duplicated x{n}", "cell": " ".join(map(str, k))})
        for k in c["unexpected"]:
            comp_rows.append({"run_tag": c["tag"], "issue": "unexpected", "cell": " ".join(map(str, k))})
    rv = dict(res["revision"] or {"tag": f"{res.get('revision_tag')} (no records)"})
    viol = rv.pop("violations", [])
    files = {
        "a_inventory.csv": res["inventory"],
        "b_completeness.csv": [{k: (len(v) if isinstance(v, list) else v) for k, v in c.items()} for c in res["completeness"]],
        "b_completeness_issues.csv": comp_rows,
        "c_field_completeness.csv": res["fields"], "c_records_lacking_fields.csv": res["fields_lacking"],
        "d_prompt_integrity.csv": res["prompt_integrity"], "d_guided_files.csv": res["guided_files"],
        "d_revision_prompt_hash.csv": res["revision_prompt_hash"],
        "e_applied_settings.csv": res["settings"],
        "f_parser_versions.csv": res["parser_versions"], "f_parse_flags.csv": res["parse_flags"],
        "f_backups.csv": res["backups"],
        "g_recompute.csv": res["recompute"], "g_reparse.csv": res["reparse"],
        "h_secret_scan.csv": res["secrets"], "i_checksums.csv": res["checksums"],
        "j_revision_summary.csv": [{"item": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}
                                   for k, v in rv.items()],
        "j_revision_violations.csv": viol,
        "k_param_tests.csv": res["param_tests"], "expectations.csv": exp,
    }
    headers = {"b_completeness_issues.csv": ["run_tag", "issue", "cell"],
               "c_records_lacking_fields.csv": ["record", "missing", "ok", "error"],
               "j_revision_violations.csv": ["chain", "revision", "rule"]}
    for name, rows in files.items():
        _write_csv(d / name, rows, headers.get(name))

    fails = [e for e in exp if e["status"] == "FAIL"]
    sec = res["secrets"]
    md = ["# Audit of stored records", "",
          "Generated by `python -m ahp_bench.audit`. Read-only; no network access; `.env` is not read.", "",
          "Conditions: guided (zero-shot, guided), direct (zero-shot, unguided; stored in the records under the key "
          "`lean`) and CoT (chain-of-thought, unguided; key `cot`); prompt ids and hashes keep the keys. The tables "
          "below show the short labels with the guidance and technique columns.", ""]
    if res.get("problem"):
        md += [f"Problem: `{res['problem']}` (problem file sha256 {res['problem_sha256']}).", ""]
    md += [
          f"Expectation checks: {len(exp)}, passed {len(exp) - len(fails)}, failed {len(fails)}.", ""]
    if fails:
        md += ["## Failed expectations", "", _md_table(fails), ""]
    md += ["## a Inventory", "", _md_table(res["inventory"]), "",
           "## b Completeness", "", _md_table(files["b_completeness.csv"]), "",
           (_md_table(comp_rows) if comp_rows else "No missing, duplicated or unexpected cells."), "",
           "## c Field completeness (result tags)", "",
           "Counts of records with a non-empty value. `usage_tokens` = prompt and completion tokens are integers. "
           "`reasoning_tokens` is informative and not part of `records_lacking_any`. Judge calls carry no "
           "prompt id or hash (empty cells).", "", _md_table(res["fields"]), "",
           (_md_table(res["fields_lacking"]) if res["fields_lacking"] else "No record lacks a required field."), "",
           "## d Prompt integrity", "", _md_table(res["prompt_integrity"]), "", _md_table(res["guided_files"]), "",
           _md_table(res["revision_prompt_hash"]), "",
           "## e Applied settings per model", "", _md_table(res["settings"]), "",
           "## f Parsing", "", _md_table(res["parser_versions"]), "", _md_table(res["parse_flags"]), "",
           "Backups compared with calls.jsonl (matched by generation id; raw fields = raw reply, messages, usage, "
           "cost, provider, timestamp, requested and applied parameters):", "", _md_table(res["backups"]), "",
           "## g Recomputation", "",
           "Weights and CR recomputed from the stored answers (direct, CoT) or the stored returned matrix (guided).", "",
           _md_table(res["recompute"]), "", "Re-parse of stored raw replies with the current parser:", "",
           _md_table(res["reparse"]), "",
           "## h Secret scan", "",
           f"Files scanned under {', '.join(res.get('scan_dirs', SCAN_DIRS))} (excluding __pycache__): {len(sec)}. "
           "Patterns: the "
           "OpenRouter key prefix and the token `Bearer` with a trailing space; key-like = full key format or a "
           "token of 20+ characters after `Bearer`. Matched content is not printed.", "",
           _md_table([s for s in sec if s["key_prefix_count"] or s["bearer_count"]]) if any(
               s["key_prefix_count"] or s["bearer_count"] for s in sec) else "No file contains either pattern.", "",
           "## i Checksums (raw records)", "", _md_table(res["checksums"]), "",
           f"## j Revision phase ({rv['tag']})", "",
           _md_table([{"item": k, "value": v["value"]} for k, v in zip(rv, files["j_revision_summary.csv"])]), "",
           (_md_table(viol) if viol else "No protocol rule violation found."), "",
           "## k Parameter tests (param_tests.jsonl)", "", _md_table(res["param_tests"]), "",
           "## Expectations", "", _md_table(exp), ""]
    write_text(d / "audit.md", "\n".join(md))
    return d


def main(argv=None) -> int:
    from . import cli
    ap = argparse.ArgumentParser(prog="python -m ahp_bench.audit")
    cli.add_common(ap, models=False)
    ap.add_argument("--out", default=None, help="default: <results>/report")
    ap.add_argument("--tag", default=MAIN_TAG, help="run_tag of the first answers (other problems)")
    ap.add_argument("--revision-tag", default=None,
                    help=f"default: {REVISION_TAG} for the study, 'revision' for other problems")
    a = ap.parse_args(argv)
    ctx = cli.context(a, models=False)
    if ctx.problem.is_study:
        results = cli.study_records(a)
        res = run_audit(results, a.revision_tag or REVISION_TAG)
    else:
        results = ctx.results
        res = run_problem_audit(ctx.problem, results, a.tag, a.revision_tag or "revision")
        res["scan_dirs"] = [results.name]
    out = Path(a.out) if a.out else results / "report"
    d = write_outputs(res, out)
    exp = expectations(res)
    fails = [e for e in exp if e["status"] == "FAIL"]
    print(f"audit written to {d}: {len(exp)} checks, {len(fails)} failed")
    for e in fails:
        print(f"  FAIL {e['check']}: expected {e['expected']}, observed {e['observed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
