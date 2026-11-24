"""Phase 2 revision chains for inconsistent matrices (prompts/PROMPT_TASLAK.md, "Phase 2"), protocol v2.

CLI:
  python -m ahp_bench.revise --source-tag main --conditions direct,cot --tag revision \
      --max-revisions 5 --budget USD [--models ...] [--dry-run] [--temperature FLOAT] [--judge] \
      [--problem PATH] [--results DIR] [--models-file PATH]

Source records are read (read-only) from <results>/raw/calls.jsonl; one record per revision call
is appended to <results>/raw/revisions.jsonl. Consistent means round(CR, 4) <= 0.10
(analysis.is_consistent), everywhere. Each revision is built from the BEST valid answers seen so far
in the chain (lowest CR among the original answers and all parsed revisions; ties keep the earlier
one). The conversation sent is
  original messages + ONE assistant turn (the original reply, or the best answers serialised in the
  condition's JSON format when they come from a revision) + revision prompt;
earlier turns are not accumulated. The revision prompt lists the three most inconsistent triads and
every circular (ordinal) triad (max 3). A chain stops at the first consistent revision or after
--max-revisions calls; its final answers are the best valid answers. An unparsable reply (or failed
call) counts as a used revision; the next call re-sends the same conversation (the best answers did
not change).

--judge: an LLM judge (config.JUDGE) writes the revision message (feedback call; code still computes
CR and the contradictory triads and appends the previous answers and the JSON-format instruction)
and, after every parsable revised reply, rates whether the original logic was preserved (verdict
call). The stopping rule stays "consistent" or --max-revisions. Every judge call is appended to
results/raw/judge_calls.jsonl. When a reply was unparsable the same conversation is re-sent, so the
previous feedback is reused without a new feedback call.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np

from . import ahp, cli, estimate, prompts, store
from . import analysis as an
from .client import api_key_for, call_model
from .config import JUDGE, MAX_TOKENS, load_env, reasoning_for
from .parse import ParseError, extract_json
from .run import _conditions, _csv, analyse

REVISION_CONDITIONS = ("lean", "cot", "described", "described_cot", "user_guided", "user_guided_cot")
REVISION_PROMPT_VERSION = "v2"
CR_LIMIT = an.CR_THRESHOLD
MAX_TRIADS = 3
MAX_CIRCULAR = 3
is_consistent = an.is_consistent
SCALE_VALUES_LINE = "Use only these values: 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9."
DIRECTION_LINE = "You may change the direction of an answer if that is what resolves a contradiction."

# Verbatim from prompts/PROMPT_TASLAK.md, "Phase 2 — revision prompt".
REVISION_TEMPLATE = (
    "The comparison matrix built from your answers has a consistency ratio CR = {cr}.\n"
    "The accepted limit is 0.10; your answers contradict each other.\n"
    "\n"
    "Most contradictory comparisons:\n"
    "{contradictions}\n"
    "\n"
    "Your previous answers:\n"
    "{previous_answers}\n"
    "\n"
    "Review your answers. You do not have to change your view; make only the changes needed\n"
    "to remove the contradiction.\n"
    + SCALE_VALUES_LINE + "\n"
    + DIRECTION_LINE + "\n"
    "Answer all questions again in the same JSON format."
)


# ---------------------------------------------------------------- prompt text

def _prob(problem):
    """The problem whose criteria and question order are used (default: the study)."""
    return prompts.STUDY if problem is None else problem


def scale_string(value: Fraction) -> str:
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def reciprocal_string(value: str) -> str:
    """'3' -> '1/3', '1/5' -> '5', '1' -> '1'."""
    return scale_string(1 / Fraction(str(value).strip()))


def _pair_value(task: str, answers, x: int, y: int, problem=None) -> str:
    """Scale string of criterion x vs criterion y; reciprocal when the question is stored as (y, x)."""
    pairs = _prob(problem).pairs(task)
    if (x, y) in pairs:
        return str(answers[pairs.index((x, y))]).strip()
    return reciprocal_string(answers[pairs.index((y, x))])


def _question(task: str, x: int, y: int, problem=None) -> int:
    pairs = _prob(problem).pairs(task)
    return (pairs.index((x, y)) if (x, y) in pairs else pairs.index((y, x))) + 1


def triad_deviations(task: str, answers, problem=None) -> list[tuple[float, int, int, int]]:
    """(|log a_ij + log a_jk - log a_ik|, i, j, k) for every triple i < j < k, sorted by
    decreasing deviation, ties broken by (i, j, k)."""
    p = _prob(problem)
    n = p.task(task).n
    a = ahp.matrix_from_answers(n, p.pairs(task), answers)
    out = []
    for i, j, k in combinations(range(n), 3):
        d = abs(math.log(a[i, j]) + math.log(a[j, k]) - math.log(a[i, k]))
        out.append((d, i, j, k))
    return sorted(out, key=lambda x: (-round(x[0], 12), x[1], x[2], x[3]))


def contradiction_lines(task: str, answers, top: int = MAX_TRIADS, problem=None) -> list[str]:
    """Up to `top` most inconsistent triads (zero-deviation triads are not listed)."""
    names = _prob(problem).criteria(task)
    lines = []
    for d, i, j, k in triad_deviations(task, answers, problem)[:top]:
        if d <= 1e-12:
            break
        qa, qb, qc = _question(task, i, j, problem), _question(task, j, k, problem), _question(task, i, k, problem)
        lines.append(f"Questions {qa}, {qb} and {qc}: {names[i]} vs {names[j]} = "
                     f"{_pair_value(task, answers, i, j, problem)}, "
                     f"{names[j]} vs {names[k]} = {_pair_value(task, answers, j, k, problem)}, "
                     f"but {names[i]} vs {names[k]} = {_pair_value(task, answers, i, k, problem)}")
    return lines


def _direction(task: str, answers, x: int, y: int, problem=None) -> int:
    """+1 if x is preferred over y (value > 1), -1 if y over x (value < 1), 0 if equal (value = 1)."""
    v = Fraction(_pair_value(task, answers, x, y, problem))
    return (v > 1) - (v < 1)


def circular_triads(task: str, answers, problem=None) -> list[tuple[int, int, int]]:
    """Ordinal cycles among i < j < k with strict directions, as (X, Y, Z) with X over Y, Y over Z and
    Z over X: (i, j, k) when i>j, j>k, k>i; (i, k, j) for the reverse orientation. Listed in the order of
    triad_deviations (largest deviation first)."""
    out = []
    for _, i, j, k in triad_deviations(task, answers, problem):
        dij, djk, dik = (_direction(task, answers, i, j, problem), _direction(task, answers, j, k, problem),
                         _direction(task, answers, i, k, problem))
        if dij == djk == 1 and dik == -1:
            out.append((i, j, k))
        elif dij == djk == -1 and dik == 1:
            out.append((i, k, j))
    return out


def circular_lines(task: str, answers, top: int = MAX_CIRCULAR, problem=None) -> list[str]:
    names = _prob(problem).criteria(task)
    lines = []
    for x, y, z in circular_triads(task, answers, problem)[:top]:
        i, j, k = sorted((x, y, z))
        qa, qb, qc = _question(task, i, j, problem), _question(task, j, k, problem), _question(task, i, k, problem)
        lines.append(f"Questions {qa}, {qb} and {qc} form a circular preference: {names[x]} over {names[y]}, "
                     f"{names[y]} over {names[z]}, {names[z]} over {names[x]}.")
    return lines


def code_block_lines(task: str, answers, problem=None) -> list[str]:
    """Code-computed contradiction block: the most inconsistent triads, then circular triads."""
    return contradiction_lines(task, answers, problem=problem) + circular_lines(task, answers, problem=problem)


def previous_answer_lines(task: str, answers, problem=None) -> list[str]:
    p = _prob(problem)
    names = p.criteria(task)
    return [f"Q{q}: Left {names[i]} | Right {names[j]} = {str(v).strip()}"
            for q, ((i, j), v) in enumerate(zip(p.pairs(task), answers), 1)]


def revision_prompt(task: str, answers, cr: float, problem=None) -> str:
    return REVISION_TEMPLATE.format(cr=f"{cr:.3f}",
                                    contradictions="\n".join(code_block_lines(task, answers, problem)),
                                    previous_answers="\n".join(previous_answer_lines(task, answers, problem)))


# ---------------------------------------------------------------- judge prompts

JUDGE_FEEDBACK_SYSTEM = (
    "You coordinate an AHP elicitation. You never tell the respondent which criterion should be more "
    "important; you only point out contradictions among their own answers. Respond only in JSON."
)
JUDGE_FEEDBACK_INSTRUCTION = (
    "Write a short message (max 120 words) to the respondent that explains which of their answers "
    "contradict each other and asks them to reconsider those answers changing only the answers that need "
    "to change, keeping their own view of what matters. Do not suggest values. Do not state or imply which "
    "criterion deserves more weight. If the list contains a circular preference, say so explicitly and state "
    "that at least one of those three answers must move toward the opposite direction; do not say which one. "
    'JSON: {"feedback": "..."}'
)
JUDGE_VERDICT_SYSTEM = "You evaluate answer revisions in an AHP elicitation. Respond only in JSON."
JUDGE_VERDICT_INSTRUCTION = (
    "Judge whether the revised answers preserve the logic of the original answers (same view of what "
    "matters most and least) or whether priorities were changed merely to reach numerical consistency. "
    'JSON: {"logic_preserved": "yes" | "partly" | "no", "changed_priorities": ["..."], "comment": "max 60 words"}'
)
SCALE_NOTE = ("Values use the AHP 1-9 scale: a value v > 1 means Left is more important than Right, "
              "1/v means Right is more important than Left, and 1 means equally important.")
JUDGE_FORMAT_TAIL = ("\n\nYour previous answers:\n{previous_answers}\n\n" + SCALE_VALUES_LINE + "\n" + DIRECTION_LINE
                     + "\nAnswer all questions again in the same JSON format.")
VERDICTS = ("yes", "partly", "no")


def answer_weights(task: str, answers, problem=None) -> np.ndarray:
    p = _prob(problem)
    return np.asarray(ahp.consistency(ahp.matrix_from_answers(p.task(task).n, p.pairs(task), answers))["weights"],
                      float)


def ranking_line(task: str, answers, problem=None) -> str:
    """Criteria from most to least important by the eigenvector weights of the answers (3 dp)."""
    names, w = _prob(problem).criteria(task), answer_weights(task, answers, problem)
    return " > ".join(f"{names[i]} ({w[i]:.3f})" for i in an.ranking_order(w))


def _criteria_block(task: str, problem=None) -> str:
    return "Criteria:\n" + "\n".join(f"- {n}" for n in _prob(problem).criteria(task))


def _stated_ranking(ranking):
    if isinstance(ranking, list) and ranking:
        return " > ".join(map(str, ranking))
    if isinstance(ranking, str) and ranking.strip():
        return ranking.strip()
    return None


def judge_feedback_messages(task: str, answers, cr: float, stated_ranking=None, problem=None) -> list[dict]:
    parts = [_criteria_block(task, problem), SCALE_NOTE,
             "Respondent's current answers:\n" + "\n".join(previous_answer_lines(task, answers, problem)),
             f"Consistency ratio CR = {cr:.3f} (accepted limit 0.10).",
             "Most contradictory comparisons (computed by code):\n"
             + "\n".join(code_block_lines(task, answers, problem))]
    stated = _stated_ranking(stated_ranking)
    if stated:
        parts.append(f"Respondent's own stated ranking (most to least important): {stated}")
    parts.append(JUDGE_FEEDBACK_INSTRUCTION)
    return [{"role": "system", "content": JUDGE_FEEDBACK_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def judge_verdict_messages(task: str, original, revised, cr_original: float, cr_before: float, cr_after: float,
                           original_reasoning=None, problem=None) -> list[dict]:
    parts = [_criteria_block(task, problem), SCALE_NOTE,
             "ORIGINAL answers (first answers):\n" + "\n".join(previous_answer_lines(task, original, problem))
             + "\nImplied ranking (most to least important, weights): " + ranking_line(task, original, problem)]
    if isinstance(original_reasoning, str) and original_reasoning.strip():
        parts.append("Respondent's original stated reasoning:\n" + original_reasoning.strip())
    parts += ["REVISED answers:\n" + "\n".join(previous_answer_lines(task, revised, problem))
              + "\nImplied ranking (most to least important, weights): " + ranking_line(task, revised, problem),
              f"CR of the original answers = {cr_original:.3f}; CR before this revision = {cr_before:.3f}; "
              f"CR after this revision = {cr_after:.3f}.",
              JUDGE_VERDICT_INSTRUCTION]
    return [{"role": "system", "content": JUDGE_VERDICT_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def judge_revision_message(feedback: str, task: str, answers, problem=None) -> str:
    """Panel-model message: judge feedback + code-written previous answers and format instruction."""
    return feedback.strip() + JUDGE_FORMAT_TAIL.format(
        previous_answers="\n".join(previous_answer_lines(task, answers, problem)))


def _judge_object(content):
    try:
        obj, _ = extract_json(content)
    except ParseError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_feedback(content):
    """Feedback string from a judge reply, or None."""
    obj = _judge_object(content)
    fb = obj.get("feedback") if obj else None
    return fb.strip() if isinstance(fb, str) and fb.strip() else None


def parse_verdict(content):
    """Normalised verdict dict from a judge reply, or None when logic_preserved is not yes/partly/no."""
    obj = _judge_object(content)
    if not obj:
        return None
    lp = obj.get("logic_preserved")
    lp = lp.strip().lower() if isinstance(lp, str) else None
    if lp not in VERDICTS:
        return None
    cp = obj.get("changed_priorities")
    cp = [str(x) for x in cp] if isinstance(cp, list) else ([cp] if isinstance(cp, str) and cp.strip() else [])
    comment = obj.get("comment")
    return {"logic_preserved": lp, "changed_priorities": cp, "comment": comment if isinstance(comment, str) else None}


def logic_metrics(task: str, original, revised, problem=None) -> dict:
    """Deterministic comparison of revised answers with the chain's original answers (eigenvector weights)."""
    w0, w1 = answer_weights(task, original, problem), answer_weights(task, revised, problem)
    o0, o1 = an.ranking_order(w0), an.ranking_order(w1)
    return {"n_answers_changed": int(sum(Fraction(str(a).strip()) != Fraction(str(b).strip())
                                         for a, b in zip(original, revised))),
            "spearman": an.spearman(w0, w1),
            "top_criterion_unchanged": o0[0] == o1[0],
            "bottom_criterion_unchanged": o0[-1] == o1[-1],
            "ranking_unchanged": o0 == o1,
            "max_abs_weight_change_pp": float(np.abs(w1 - w0).max() * 100.0)}


# ---------------------------------------------------------------- chains

def chain_key(r: dict) -> tuple:
    return (r.get("model_key"), r.get("condition"), r.get("task"), int(r.get("rep") or 0))


def source_matrices(records, source_tag: str, conditions=REVISION_CONDITIONS, models=None, problem=None) -> dict:
    """(model, condition, task, rep) -> first usable source record (analysis.select_records rules)
    with answers, restricted to the given conditions/models."""
    selected, _, _ = an.select_records(records, source_tag, problem)
    return {k: r for k, r in sorted(selected.items())
            if k[1] in conditions and (models is None or k[0] in models) and isinstance(r.get("answers"), list)}


def _cr(r: dict) -> float:
    return an.priority(r["matrix"])[1]


def revision_records(records, tag: str, source_tag: str) -> dict:
    """chain key -> list of revision records (file order) for this tag and source tag."""
    out: dict = {}
    for r in records:
        if r.get("run_tag") == tag and r.get("source_tag") == source_tag:
            out.setdefault(chain_key(r), []).append(r)
    return out


def chain_finished(recs, max_revisions: int) -> bool:
    return any((r.get("parse_ok") and is_consistent(r["CR"])) or int(r.get("revision_number") or 0) >= max_revisions
               for r in recs)


def _valid_revision(r: dict, problem=None) -> bool:
    p = _prob(problem)
    n = p.task(r["task"]).n if r.get("task") in p.task_names else None
    return bool(r.get("parse_ok")) and isinstance(r.get("answers"), list) and an.valid_matrix(r.get("matrix"), n)


def best_so_far(source: dict, recs, problem=None) -> tuple[dict, int, float]:
    """(record, revision number (0 = original), CR) of the valid answers with the lowest CR among the
    source record and all parsed revisions; ties keep the earlier one."""
    best, number, cr = source, 0, _cr(source)
    for r in recs:
        if _valid_revision(r, problem):
            c = float(r["CR"])
            if c < cr:
                best, number, cr = r, int(r.get("revision_number") or 0), c
    return best, number, cr


def serialise_answers(condition: str, record: dict) -> str:
    """Answers of a record in the condition's JSON response format (assistant turn)."""
    items = [{"question": q, "value": str(v).strip()} for q, v in enumerate(record["answers"], 1)]
    if prompts.is_cot(condition):
        ranking = record.get("ranking")
        return json.dumps({"reasoning": record.get("reasoning") or "",
                           "ranking": ranking if isinstance(ranking, list) else [], "answers": items},
                          ensure_ascii=False)
    return json.dumps({"answers": items}, ensure_ascii=False)


def build_conversation(source: dict, best: dict, based_on_revision: int, user_message: str) -> list[dict]:
    """original messages + ONE assistant turn + new user message. The assistant turn is the original
    reply, or the best answers so far when those come from a revision (never two assistant turns
    in a row: some providers reject or merge consecutive same-role messages)."""
    if based_on_revision:
        reply = serialise_answers(source["condition"], best)
    else:
        reply = source.get("raw_reply") or source.get("content")
    return list(source["messages"]) + [{"role": "assistant", "content": reply},
                                       {"role": "user", "content": user_message}]


def next_state(source: dict, recs, problem=None) -> dict:
    """Next call of a chain: messages (ending with the code-written revision prompt), best answers,
    their CR, the best record, based_on_revision, the next revision number and whether the previous
    conversation is re-sent unchanged (last reply unparsable)."""
    task = source["task"]
    best, based_on, cr = best_so_far(source, recs, problem)
    nxt = int(recs[-1].get("revision_number") or len(recs)) + 1 if recs else 1
    state = {"answers": list(best["answers"]), "CR": cr, "best": best, "based_on_revision": based_on,
             "number": nxt, "resend": False}
    if recs and not recs[-1].get("parse_ok") and recs[-1].get("messages"):
        state.update(messages=list(recs[-1]["messages"]), resend=True)
    else:
        state["messages"] = build_conversation(source, best, based_on,
                                               revision_prompt(task, best["answers"], cr, problem))
    return state


def plan_chains(records, revision_recs, source_tag, conditions, models, max_revisions, problem=None):
    """Return (chains_needing_revision, todo) as lists of (key, source record)."""
    sources = source_matrices(records, source_tag, conditions, models, problem)
    need = [(k, r) for k, r in sources.items() if not is_consistent(_cr(r))]
    todo = [(k, r) for k, r in need if not chain_finished(revision_recs.get(k, []), max_revisions)]
    return need, todo


def chain_summary(source_records, revision_records_list, source_tag="main", revision_tag="revision",
                  conditions=REVISION_CONDITIONS, max_revisions: int = 5, problem=None) -> list[dict]:
    """One dict per (model, condition, task, rep) source matrix: initial CR, revisions used (revision
    calls until the first consistent result, else all calls made), final CR and answers (the BEST valid
    answers: lowest CR among the original and all parsable revisions), reached (final answers
    consistent), the revision number at which consistency was first reached, the revision the final
    answers come from (0 = original), chain completeness, and initial/final weights and matrices."""
    sources = source_matrices(source_records, source_tag, conditions, problem=problem)
    revs = revision_records(revision_records_list, revision_tag, source_tag)
    out = []
    for key, src in sources.items():
        names = _prob(problem).criteria(key[2])
        w0, cr0 = an.priority(src["matrix"])
        needs = not is_consistent(cr0)
        recs = revs.get(key, []) if needs else []
        best, best_no, _ = best_so_far(src, recs, problem)
        final = best if best_no else None
        w1, cr1 = an.priority(final["matrix"]) if final else (w0, cr0)
        reached_at = next((int(r["revision_number"]) for r in recs
                           if _valid_revision(r, problem) and is_consistent(r["CR"])), None)
        out.append({
            "model_key": key[0], "condition": key[1], "task": key[2], "rep": key[3],
            "needs_revision": needs, "initial_CR": cr0,
            "revisions_used": reached_at if reached_at is not None else len(recs),
            "revision_calls": len(recs), "final_CR": cr1, "reached": is_consistent(cr1), "reached_at": reached_at,
            "final_revision": best_no,
            "complete": not needs or chain_finished(recs, max_revisions),
            "has_judge": any("judge_model_id" in r for r in recs),
            "initial_weights": dict(zip(names, map(float, w0))), "final_weights": dict(zip(names, map(float, w1))),
            "initial_answers": list(src["answers"]),
            "final_answers": list(final["answers"] if final else src["answers"]),
            "final_record": final,
            "initial_matrix": np.asarray(src["matrix"], dtype=float).tolist(),
            "final_matrix": np.asarray(final["matrix"] if final else src["matrix"], dtype=float).tolist(),
        })
    return out


# ---------------------------------------------------------------- CLI

def _progress(k, chain_no, n_chains, rec, max_revisions, spent):
    cr = f"{rec['CR']:.3f}" if rec.get("parse_ok") else "PARSE_FAIL"
    u = rec.get("usage") or {}
    print(f"[call {k} | chain {chain_no}/{n_chains}] {rec['model_key']} {prompts.condition_label(rec['condition'])} {rec['task']} "
          f"rep{rec['rep']} rev{rec['revision_number']}/{max_revisions} CR {rec['based_on_CR']:.3f} -> {cr} "
          f"tok={u.get('prompt_tokens')}/{u.get('completion_tokens')} "
          f"cost=${float(rec.get('cost') or 0):.4f} "
          + (f"judge=${float(rec.get('judge_cost') or 0):.4f} " if "judge_cost" in rec else "")
          + f"total=${spent:.4f}")



def _judge_call(purpose: str, msgs, base: dict, path: Path, transport, parse, judge=None, panel=None,
                max_tokens: int = MAX_TOKENS):
    """One judge call, appended to judge_calls.jsonl. Returns (parsed reply or None, cost)."""
    judge = judge or JUDGE
    try:
        call = call_model(judge["key"], msgs, temperature=judge["temperature"], reasoning=judge["reasoning"],
                          max_tokens=max_tokens, transport=transport, model=judge, panel=panel)
    except RuntimeError as e:
        call = {"model_key": judge["key"], "model_id": judge["id"], "ok": False, "error": str(e), "content": None,
                "usage": {}, "cost": None, "applied_params": None}
    parsed = parse(call["content"]) if call.get("ok") else None
    store.append_jsonl(path, {**base, "purpose": purpose, "messages": msgs, **call, "raw_reply": call.get("content"),
                              "judge_parsed": parsed, "judge_parse_ok": parsed is not None})
    return parsed, float(call.get("cost") or 0)


def main(argv=None, transport=None, root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ahp_bench.revise")
    cli.add_common(ap)
    ap.add_argument("--source-tag", default="main")
    ap.add_argument("--conditions", default="direct,cot",
                    help="comma-separated: direct (zero-shot, unguided; stored as lean) and/or cot (default: both)")
    ap.add_argument("--tag", default="revision", help="run_tag of the revision records")
    ap.add_argument("--max-revisions", type=int, default=5)
    ap.add_argument("--budget", type=float, required=True, help="USD cap per revision run_tag")
    ap.add_argument("--models", default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--temperature", type=float, default=None,
                    help="requested temperature (default: models file, 0.6 in the study)")
    ap.add_argument("--judge", action="store_true",
                    help="LLM judge (models file 'judge' entry) writes the revision feedback and rates logic "
                         "preservation")
    a = ap.parse_args(argv)
    if a.max_revisions < 1:
        raise SystemExit("--max-revisions must be >= 1")

    ctx = cli.context(a, root)
    p, panel = ctx.problem, ctx.panel
    pm, jm = panel["models"], panel["judge"]
    temperature = panel["temperature"] if a.temperature is None else a.temperature
    models = _csv(a.models, list(pm))
    conditions = _conditions(a.conditions, REVISION_CONDITIONS)
    src_path = ctx.raw / "calls.jsonl"
    out_path = ctx.raw / "revisions.jsonl"
    judge_path = ctx.raw / "judge_calls.jsonl"
    records, bad = an.load_records(src_path)
    clash = cli.check_problem_matches(records, p)
    if clash:
        print(f"{src_path} holds records whose criteria differ from problem {p.name!r} (tasks: {', '.join(clash)})",
              file=sys.stderr)
        return 2
    rev_all, bad_rev = an.load_records(out_path)
    revs = revision_records(rev_all, a.tag, a.source_tag)
    need, todo = plan_chains(records, revs, a.source_tag, conditions, set(models), a.max_revisions, p)
    spent = sum(float(r.get("cost") or 0) for r in rev_all if r.get("run_tag") == a.tag)
    if judge_path.exists():  # judge spending of this tag (every judge call, also from interrupted steps)
        spent += sum(float(r.get("cost") or 0) for r in an.load_records(judge_path)[0] if r.get("run_tag") == a.tag)
    print(f"source_tag={a.source_tag} tag={a.tag}: {len(need)} inconsistent matrices (round(CR, 4) > {CR_LIMIT:.2f}), "
          f"{len(need) - len(todo)} chains already finished, {len(todo)} to run; spent ${spent:.4f}"
          + (f"; {bad + bad_rev} unreadable lines skipped" if bad + bad_rev else ""))

    if a.dry_run:
        for c in conditions:
            for t in p.task_names:
                n_need = sum(1 for k, _ in need if k[1] == c and k[2] == t)
                n_todo = sum(1 for k, _ in todo if k[1] == c and k[2] == t)
                print(f"  {prompts.condition_label(c)} {t}: {n_need} needing revision, {n_todo} chains planned")
            print(f"  {prompts.condition_label(c)} total: {sum(1 for k, _ in todo if k[1] == c)} chains planned")
        remaining = [(k, a.max_revisions - (next_state(r, revs.get(k, []), p)["number"] - 1)) for k, r in todo]
        n_calls = sum(n for _, n in remaining)
        print(f"dry run: {len(todo)} chains planned, at most "
              f"{n_calls} calls; "
              f"T={temperature}, max_revisions={a.max_revisions}; no calls made")
        ref_panel, ref_judge = estimate.revision_reference()
        planned = [((pm[k[0]]["id"], k[1]), k[0], len(p.pairs(k[2]))) for k, n in remaining for _ in range(n)]
        total, unknown = estimate.estimate(planned, ref_panel)
        if a.judge:
            print(f"judge: {jm['id']} (T={jm['temperature']}), at most {2 * n_calls} judge calls "
                  f"(1 feedback + 1 verdict per revision call)")
            jt, ju = estimate.estimate([(jm["id"], jm["key"], len(p.pairs(k[2]))) for k, n in remaining
                                        for _ in range(2 * n)], ref_judge)
            total, unknown = total + jt, sorted(set(unknown) | set(ju))
        print("upper bound: " + estimate.describe(total, unknown, n_calls * (3 if a.judge else 1)))
        return 0

    load_env(ctx.root / ".env")
    missing = sorted({pm[k[0]].get("env", "?") for k, _ in todo if not api_key_for(pm[k[0]], pm)})
    if a.judge and not api_key_for(jm, pm):
        missing.append(f"{jm.get('env')} (or any panel-model key)")
    if missing:
        print(f"missing API key variables: {', '.join(missing)} (or OPENROUTER_API_KEY)", file=sys.stderr)
        return 2

    k_call = 0
    for chain_no, (key, src) in enumerate(todo, 1):
        m, c, t, rep = key
        recs = list(revs.get(key, []))
        source_cr = _cr(src)
        while not chain_finished(recs, a.max_revisions):
            if spent >= a.budget:
                print(f"budget reached (${spent:.4f} >= ${a.budget:.4f}); stopping")
                return 0
            st = next_state(src, recs, p)
            msgs, answers, cr, number = st["messages"], st["answers"], st["CR"], st["number"]
            judge = {}
            if a.judge:
                base = {"run_tag": a.tag, "source_tag": a.source_tag, "chain_key": list(key), "chain_model_key": m,
                        "condition": c, "task": t, "rep": rep, "revision_number": number}
                judge_cost = 0.0
                if st["resend"]:  # same conversation re-sent: reuse its feedback
                    prev = recs[-1]
                    judge.update(judge_feedback=prev.get("judge_feedback"),
                                 judge_feedback_ok=prev.get("judge_feedback_ok"), judge_feedback_reused=True)
                else:
                    fb, cost = _judge_call("feedback", judge_feedback_messages(
                        t, answers, cr, st["best"].get("ranking") if prompts.is_cot(c) else None, p), base, judge_path,
                        transport, parse_feedback, jm, pm, panel["max_tokens"])
                    judge_cost += cost
                    if fb:
                        msgs[-1] = {"role": "user", "content": judge_revision_message(fb, t, answers, p)}
                    judge.update(judge_feedback=fb, judge_feedback_ok=fb is not None, judge_feedback_reused=False)
            pid = f"revision_{'judge_' if a.judge else ''}{c}_{t}_{REVISION_PROMPT_VERSION}"
            rec = {"run_tag": a.tag, "source_tag": a.source_tag, "model_key": m, "model_id": pm[m]["id"],
                   "developer": pm[m]["developer"], "condition": c, "task": t, "rep": rep,
                   "revision_number": number, "max_revisions": a.max_revisions, "source_CR": source_cr,
                   "based_on_answers": list(answers), "based_on_CR": cr,
                   "based_on_revision": st["based_on_revision"],
                   "prompt_id": pid if p.is_study else f"{p.name}:{pid}",
                   "prompt_sha256": prompts.prompt_sha256(msgs), "messages": msgs,
                   "criteria": p.criteria(t), "temperature_requested": temperature}
            if not p.is_study:
                rec.update(problem=p.name, problem_sha256=p.sha256)
            call = call_model(m, msgs, temperature=temperature,
                              reasoning=reasoning_for(m, pm, panel["default_reasoning"]),
                              max_tokens=panel["max_tokens"], transport=transport, model=pm[m], panel=pm)
            rec.update(call)
            rec["raw_reply"] = call["content"]
            try:
                if not call["ok"]:
                    raise ParseError(f"call failed: {call['error']}")
                rec.update(analyse(c, t, call["content"], p), parse_ok=True, parse_error=None)
            except ParseError as e:
                rec.update(parse_ok=False, parse_error=str(e))
            rec["best_CR_so_far"] = min(cr, float(rec["CR"])) if _valid_revision(rec, p) else cr
            if a.judge:
                verdict, verdict_ok, metrics = None, None, None  # None: no verdict call (unparsable reply)
                if rec["parse_ok"]:
                    verdict, cost = _judge_call("verdict", judge_verdict_messages(
                        t, src["answers"], rec["answers"], source_cr, cr, float(rec["CR"]),
                        src.get("reasoning") if prompts.is_cot(c) else None, p), base, judge_path, transport, parse_verdict,
                        jm, pm, panel["max_tokens"])
                    judge_cost += cost
                    verdict_ok = verdict is not None
                    metrics = logic_metrics(t, src["answers"], rec["answers"], p)
                rec.update(judge_model_id=jm["id"], **judge, judge_verdict=verdict, judge_verdict_ok=verdict_ok,
                           judge_cost=judge_cost, logic_metrics=metrics)
            store.append_jsonl(out_path, rec)
            recs.append(rec)
            spent += float(call.get("cost") or 0) + float(rec.get("judge_cost") or 0)
            k_call += 1
            _progress(k_call, chain_no, len(todo), rec, a.max_revisions, spent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
