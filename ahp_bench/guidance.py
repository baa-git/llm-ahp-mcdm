"""Prompt guidance levels for problems other than the study: neutral descriptions and user priorities.

Three guidance levels, chosen with the condition option of `run` and `host prompts` (technique and guidance
are independent; each level exists zero-shot and with CoT):

  unguided   direct / cot                    criterion names only (default)
  described  described / described-cot       names plus neutral descriptions (what the criterion measures,
                                             unit, direction, source indicator); importance cues are refused
                                             unless --allow-cues is given
  guided     guided / guided-cot             descriptions plus the user's priority order and hints
             (stored keys user_guided, user_guided_cot)

The problem file stores
  "descriptions": {"drafted_by": "...", "confirmed_by_user": false, "criteria": {"<criterion>": "<text>"}}
  "guidance": {"source": "user" | "orchestrator-draft", "confirmed_by_user": false,
               "user_statement": "<the user's own words>", "order": {"<task>": [criteria, most important first]},
               "hints": {"<task>": ["..."]}}
The priority order must come from the user's own stated priorities: an agent may only turn the user's words
into the block and show it for confirmation; it must not invent an order. Weighting with a guidance block
whose source is not "user" or that is not confirmed prints a prominent warning, and the report carries it.

ahp-bench guidance draft-descriptions --problem P            neutral drafts from the indicator table (code)
ahp-bench guidance descriptions-prompt --problem P --out DIR  host mode: prompt file for a subagent
ahp-bench guidance descriptions-ingest --problem P --dir DIR  host mode: check and store the reply
ahp-bench guidance descriptions-propose --problem P --model ID --budget USD [--dry-run]   API mode
ahp-bench guidance set-order --problem P --task main --order "A | B | C" --source user [--user-statement TEXT]
ahp-bench guidance confirm --problem P --criteria | --descriptions | --guidance
ahp-bench guidance check --problem P [--allow-cues]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import problem as pb

CUE_PATTERNS = (
    r"most important", r"least important", r"more important", r"less important", r"as important",
    r"importan(?:t|ce|tly)", r"critical(?:ly)?", r"crucial(?:ly)?", r"key", r"priorit(?:y|ies|ise|ize|ised|ized)",
    r"vital", r"essential(?:ly)?", r"paramount", r"primary", r"secondary", r"highest", r"lowest", r"top",
    r"rank(?:s|ed|ing|ings)?", r"significan(?:t|tly|ce)", r"matters? (?:most|more|less)", r"decisive",
    r"fundamental", r"minor", r"marginal", r"negligible", r"major", r"above all", r"first and foremost",
    r"önem\w*", r"kritik\w*", r"öncelik\w*",
)
_CUE_RE = re.compile(r"(?<![\w-])(" + "|".join(CUE_PATTERNS) + r")(?![\w-])", re.I)
DESCRIBED = ("described", "described_cot")
GUIDED = ("user_guided", "user_guided_cot")
GUIDED_NOTE = ("Weights obtained under guidance largely reproduce the guidance and should be read as the guide "
               "author's priorities expressed in AHP form, not as independent judgments; run the unguided condition "
               "alongside for contrast.")


def importance_cues(text: str) -> list:
    """Words or phrases in a text that signal importance or rank (case-insensitive, whole words)."""
    return sorted({m.group(1).lower() for m in _CUE_RE.finditer(text or "")})


def description_cues(problem) -> dict:
    """criterion -> cues found in its description (only criteria with cues)."""
    d = (problem.descriptions or {}).get("criteria") or {}
    out = {}
    for c, text in d.items():
        cues = importance_cues(text)
        if cues:
            out[c] = cues
    return out


def check_cues(problem, conditions, allow_cues: bool) -> str | None:
    """Error message when a described condition is requested and the descriptions carry importance cues."""
    if not any(c in DESCRIBED for c in conditions):
        return None
    found = description_cues(problem)
    if not found or allow_cues:
        return None
    lines = "; ".join(f"{c}: {', '.join(v)}" for c, v in found.items())
    return ("the descriptions contain importance cues, which the described condition must not have (" + lines
            + "). Rewrite them neutrally (ahp-bench guidance check), or pass --allow-cues to run anyway (recorded).")


def weighting_warnings(problem, conditions) -> list:
    """Warnings printed by the weighting commands (they never block)."""
    out = []
    raw = problem.raw if isinstance(problem.raw, dict) else {}
    prov = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else None
    if prov is not None and prov.get("criteria_confirmed_by_user") is False:
        out.append(f"the criteria (proposed by {prov.get('criteria_proposed_by')}) are not confirmed by the user: show "
                   "the hierarchy, directions and performance matrix and record the confirmation "
                   "(ahp-bench guidance confirm --criteria, or criteria build --confirmed)")
    if any(c in DESCRIBED + GUIDED for c in conditions) and problem.descriptions \
            and not problem.descriptions.get("confirmed_by_user"):
        out.append(f"the criterion descriptions (drafted by {problem.descriptions.get('drafted_by')}) are not confirmed "
                   "by the user (ahp-bench guidance confirm --descriptions)")
    if any(c in GUIDED for c in conditions):
        out.append(guidance_warning(problem) or "")
        out = [x for x in out if x]
    return out


def guidance_warning(problem) -> str | None:
    g = problem.guidance or {}
    if g.get("source") == "user" and g.get("confirmed_by_user"):
        return None
    why = []
    if g.get("source") != "user":
        why.append(f"source is {g.get('source')!r}, not the user's own stated priorities")
    if not g.get("confirmed_by_user"):
        why.append("not confirmed by the user")
    return ("GUIDANCE NOT CONFIRMED AS THE USER'S OWN PRIORITIES (" + "; ".join(why) + "). Weights under the guided "
            "condition reproduce this order; it must come from the user, not from an agent.")


def print_warnings(problem, conditions) -> None:
    for w in weighting_warnings(problem, conditions):
        prominent = w.startswith("GUIDANCE")
        bar = "!" * 78
        print((f"{bar}\nWARNING: {w}\n{bar}" if prominent else f"warning: {w}"), file=sys.stderr)


# ---------------------------------------------------------------- drafting descriptions

def code_drafts(problem) -> dict:
    """Neutral descriptions written by code from the indicator table (leaves) and the hierarchy (parents)."""
    out = {}
    inds = {}
    if problem.ranking:
        by_code = {i["code"]: i for i in problem.ranking["indicators"]}
        inds = {leaf: by_code[code] for leaf, code in problem.ranking["criterion_indicator"].items()}
    for c in problem.main.criteria:
        g = problem.group_of(c)
        if g is not None:
            out[c] = "Groups the sub-criteria " + ", ".join(g.criteria) + "."
    for leaf in problem.leaf_names:
        i = inds.get(leaf)
        if i is None:
            out[leaf] = f"REVIEW: what {leaf} measures, its unit and whether higher or lower values are better."
            continue
        text = f"Measured by {i['name']}"
        if i.get("unit"):
            text += f" ({i['unit']})"
        text += "; " + ("higher" if i["direction"] == "benefit" else "lower") + " values are better"
        if i.get("source"):
            text += f"; computed as {i.get('aggregation', 'value')} of {i['source']}"
            if i.get("filter"):
                text += f" where {i['filter']}"
        out[leaf] = text + f" (indicator {i['code']})."
    return {c: out[c] for c in problem.all_criteria}


DESC_SYSTEM = ("You write neutral descriptions of decision criteria for pairwise-comparison questions. "
               "Respond only in the requested JSON format.")


def descriptions_messages(problem) -> list:
    lines = [f"Write one neutral description for each criterion used in {problem.goal}.", "", "Criteria:"]
    inds = {}
    if problem.ranking:
        by_code = {i["code"]: i for i in problem.ranking["indicators"]}
        inds = {leaf: by_code[code] for leaf, code in problem.ranking["criterion_indicator"].items()}
    for c in problem.all_criteria:
        g = problem.group_of(c)
        if g is not None:
            lines.append(f"- {c}: groups the sub-criteria {', '.join(g.criteria)}")
        elif c in inds:
            i = inds[c]
            bits = [f"indicator {i['code']}: {i['name']}"]
            if i.get("unit"):
                bits.append(f"unit {i['unit']}")
            bits.append(f"{'higher' if i['direction'] == 'benefit' else 'lower'} values are better")
            if i.get("source"):
                bits.append(f"computed as {i.get('aggregation', 'value')} of {i['source']}")
            lines.append(f"- {c}: " + "; ".join(bits))
        else:
            lines.append(f"- {c}")
    lines += ["", "Rules:",
              "1. One or two sentences per criterion: what it measures, its unit, whether higher or lower values are "
              "better, and the source indicator when one is given.",
              "2. Do not signal importance or rank: no words such as important, critical, key, priority, vital, "
              "essential, primary, major, minor, top, and no comparison with other criteria.",
              "3. Do not recommend anything and do not mention the alternatives.", "",
              "Response format (only this JSON):",
              '{"descriptions": {"<criterion name>": "<description>", ...}}']
    return [{"role": "system", "content": DESC_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def parse_descriptions(text: str, problem) -> dict:
    from .criteria import SpecError, parse_reply
    obj = parse_reply(text)
    d = obj.get("descriptions", obj)
    if not isinstance(d, dict):
        raise SpecError(["'descriptions' must be an object criterion -> text"])
    missing = [c for c in problem.all_criteria if not isinstance(d.get(c), str) or not d[c].strip()]
    if missing:
        raise SpecError([f"no description for: {', '.join(missing)}"])
    return {c: d[c].strip() for c in problem.all_criteria}


# ---------------------------------------------------------------- problem file editing

def _load_raw(path: Path) -> dict:
    return json.loads(Path(path).read_bytes().decode("utf-8-sig"))


def save_problem(path: Path, raw: dict) -> None:
    pb.problem_from_dict(json.loads(json.dumps(raw)))  # must stay valid
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")


def store_descriptions(path: Path, texts: dict, drafted_by: str, force: bool = False) -> None:
    raw = _load_raw(path)
    if raw.get("descriptions") and not force:
        raise SystemExit(f"{path} already has descriptions; use --force to replace them")
    raw["descriptions"] = {"drafted_by": drafted_by, "confirmed_by_user": False, "criteria": texts}
    save_problem(path, raw)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(path: Path, rec: dict) -> None:
    from . import store
    store.append_jsonl(path, {"kind": "descriptions_draft", "timestamp_utc": _now(), **rec})


def _ingest_text(problem, path, raw_reply, msgs, drafted_by, log, extra, force) -> int:
    from .criteria import SpecError, prompt_sha256
    rec = {**extra, "problem": problem.name, "prompt_sha256": prompt_sha256(msgs), "messages": msgs,
           "raw_reply": raw_reply}
    try:
        texts = parse_descriptions(raw_reply, problem)
    except SpecError as e:
        _log(log, {**rec, "parse_ok": False, "parse_error": str(e)})
        print(f"reply not usable: {e}", file=sys.stderr)
        return 2
    cues = {c: importance_cues(t) for c, t in texts.items() if importance_cues(t)}
    _log(log, {**rec, "parse_ok": True, "descriptions": texts, "importance_cues": cues})
    store_descriptions(path, texts, drafted_by, force)
    print(f"stored descriptions drafted by {drafted_by} in {path} (confirmed_by_user: false); log: {log}")
    if cues:
        print("importance cues found (the described condition refuses them):", file=sys.stderr)
        for c, v in cues.items():
            print(f"  - {c}: {', '.join(v)}", file=sys.stderr)
        return 2
    return 0


def _order(text: str, names: list) -> list:
    parts = [x.strip() for x in re.split(r"\|", text) if x.strip()]
    if len(parts) == 1 and "," in text and all(x.strip().isdigit() for x in text.split(",")):
        parts = [x.strip() for x in text.split(",")]
    if parts and all(x.isdigit() for x in parts):
        try:
            parts = [names[int(x) - 1] for x in parts]
        except IndexError:
            raise SystemExit(f"--order: indices must be 1..{len(names)}") from None
    lower = {n.lower(): n for n in names}
    out = []
    for x in parts:
        if x.lower() not in lower:
            raise SystemExit(f"--order: unknown criterion {x!r}; criteria: {' | '.join(names)}")
        out.append(lower[x.lower()])
    if sorted(out) != sorted(names):
        raise SystemExit(f"--order must name every criterion of the task exactly once: {' | '.join(names)}")
    return out


def show(problem) -> str:
    lines = [f"problem: {problem.name}"]
    d = problem.descriptions
    if d:
        lines.append(f"descriptions (drafted by {d.get('drafted_by')}, confirmed by user: "
                     f"{'yes' if d.get('confirmed_by_user') else 'no'}):")
        cues = description_cues(problem)
        for c in problem.all_criteria:
            lines.append(f"  {c}: {d['criteria'][c]}" + (f"   [importance cues: {', '.join(cues[c])}]" if c in cues else ""))
    else:
        lines.append("descriptions: none (conditions described / guided unavailable)")
    g = problem.guidance
    if g:
        lines.append(f"guidance (source {g.get('source')}, confirmed by user: {'yes' if g.get('confirmed_by_user') else 'no'}):")
        if g.get("user_statement"):
            lines.append(f"  user statement: {g['user_statement']}")
        for t, o in (g.get("order") or {}).items():
            lines.append(f"  order {t}: " + " > ".join(o))
        for t, h in (g.get("hints") or {}).items():
            lines += [f"  hint {t}: {x}" for x in h]
    else:
        lines.append("guidance: none (condition guided unavailable)")
    lines.append("conditions: " + ", ".join(problem.conditions))
    return "\n".join(lines)


def main(argv=None, transport=None) -> int:
    ap = argparse.ArgumentParser(prog="ahp-bench guidance", description="Neutral criterion descriptions and user "
                                 "priorities for the described and guided conditions.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("draft-descriptions", help="neutral drafts from the indicator table (written by code)")
    p.add_argument("--problem", required=True)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("descriptions-prompt", help="host mode: prompt file asking a subagent for descriptions")
    p.add_argument("--problem", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("descriptions-ingest", help="host mode: check the reply and store the descriptions")
    p.add_argument("--problem", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--drafted-by", default="host agent")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("descriptions-propose", help="API mode: one OpenRouter model drafts descriptions")
    p.add_argument("--problem", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--models-file", default=None)
    p.add_argument("--log", default=None, help="default: descriptions_calls.jsonl next to the problem file")
    p.add_argument("--tag", default="descriptions")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("set-order", help="store the user's priority order (and hints) for one task")
    p.add_argument("--problem", required=True)
    p.add_argument("--task", default=pb.MAIN_TASK)
    p.add_argument("--order", default=None, help="criteria separated by '|', most important first (or 1-based indices)")
    p.add_argument("--hint", action="append", default=[], help="a hint in the user's words (repeatable)")
    p.add_argument("--source", required=True, choices=pb.GUIDANCE_SOURCES,
                   help="user = the user's own stated priorities; orchestrator-draft = drafted by an agent")
    p.add_argument("--user-statement", default=None, help="the user's own words the order was taken from")
    p.add_argument("--confirmed", action="store_true", help="the user has seen and confirmed this block")
    p = sub.add_parser("confirm", help="record that the user confirmed criteria, descriptions or guidance")
    p.add_argument("--problem", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--criteria", action="store_true")
    g.add_argument("--descriptions", action="store_true")
    g.add_argument("--guidance", action="store_true")
    p = sub.add_parser("check", help="show descriptions and guidance; flag importance cues")
    p.add_argument("--problem", required=True)
    p.add_argument("--allow-cues", action="store_true")
    a = ap.parse_args(argv)
    path = Path(a.problem)
    try:
        prob = pb.load_problem(path)
    except (pb.ProblemError, OSError) as e:
        print(f"invalid problem file: {e}", file=sys.stderr)
        return 2
    if prob.is_study:
        print("the study problem keeps its verbatim prompts; use a copy for new problems", file=sys.stderr)
        return 2

    if a.cmd == "draft-descriptions":
        store_descriptions(path, code_drafts(prob), "ahp-bench (from the indicator table)", a.force)
        print(show(pb.load_problem(path)))
        print("drafts stored with confirmed_by_user: false; show them to the user, edit if needed, then "
              "ahp-bench guidance confirm --descriptions")
        return 0
    if a.cmd == "descriptions-prompt":
        from .host import render
        out = Path(a.out)
        msgs = descriptions_messages(prob)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "descriptions_prompt.txt").open("w", encoding="utf-8", newline="\n") as f:
            f.write(render(msgs))
        from .criteria import prompt_sha256, write_json
        write_json(out / "descriptions_manifest.json", {"problem": prob.name, "problem_sha256": prob.sha256,
                                                        "prompt_sha256": prompt_sha256(msgs), "messages": msgs,
                                                        "reply_file": "descriptions_reply.json"})
        print(f"wrote {out / 'descriptions_prompt.txt'}; give it verbatim to ONE fresh subagent and save its JSON "
              f"reply as {out / 'descriptions_reply.json'}")
        return 0
    if a.cmd == "descriptions-ingest":
        folder = Path(a.dir)
        m = json.loads((folder / "descriptions_manifest.json").read_text(encoding="utf-8"))
        if m["problem"] != prob.name:
            raise SystemExit(f"the manifest belongs to problem {m['problem']!r}")
        reply = folder / m["reply_file"]
        if not reply.is_file():
            raise SystemExit(f"{reply} not found")
        return _ingest_text(prob, path, reply.read_text(encoding="utf-8-sig"), m["messages"], a.drafted_by,
                            folder / "descriptions_calls.jsonl", {"source": "host"}, a.force)
    if a.cmd == "descriptions-propose":
        from . import store
        from .client import api_key_for, call_model
        from .config import SHARED_KEY_ENV, load_env, load_panel, reasoning_for
        msgs = descriptions_messages(prob)
        panel = load_panel(Path(a.models_file) if a.models_file else None)
        pm = panel["models"]
        model = pm.get(a.model) or {"id": a.model, "env": SHARED_KEY_ENV, "developer": a.model.split("/")[0]}
        log = Path(a.log) if a.log else path.parent / "descriptions_calls.jsonl"
        spent = sum(float(r.get("cost") or 0) for r in store.read_jsonl(log) if r.get("run_tag") == a.tag)
        reasoning = reasoning_for(a.model, pm, panel["default_reasoning"]) if a.model in pm else panel["default_reasoning"]
        if a.dry_run:
            for m in msgs:
                print(f"--- {m['role']} ---\n{m['content']}")
            print(f"dry run: 1 planned call to {model['id']}; cost unknown (no reference); budget ${a.budget:.4f}, "
                  f"spent ${spent:.4f} under tag {a.tag!r}")
            return 0
        if spent >= a.budget:
            print(f"budget reached (${spent:.4f} >= ${a.budget:.4f}); no call", file=sys.stderr)
            return 2
        load_env(Path.cwd() / ".env")
        if not api_key_for(model, pm):
            print(f"missing API key variable {model.get('env')} (or {SHARED_KEY_ENV})", file=sys.stderr)
            return 2
        call = call_model(a.model, msgs, temperature=panel["temperature"], reasoning=reasoning,
                          max_tokens=panel["max_tokens"], transport=transport, model=model, panel=pm)
        extra = {"run_tag": a.tag, "source": "api", **{k: v for k, v in call.items() if k != "content"}}
        if not call["ok"]:
            _log(log, {**extra, "messages": msgs, "parse_ok": False, "parse_error": call["error"]})
            print(f"call failed: {call['error']}", file=sys.stderr)
            return 2
        return _ingest_text(prob, path, call["content"], msgs, model["id"], log, extra, a.force)
    if a.cmd == "set-order":
        if a.task not in prob.task_names:
            raise SystemExit(f"unknown task {a.task!r}; tasks: {', '.join(prob.task_names)}")
        if not prob.descriptions:
            raise SystemExit("store neutral descriptions first (ahp-bench guidance draft-descriptions); the guided "
                             "level builds on them")
        raw = _load_raw(path)
        gd = raw.get("guidance") or {}
        if a.order:
            gd.setdefault("order", {})[a.task] = _order(a.order, prob.criteria(a.task))
        if a.hint:
            gd.setdefault("hints", {})[a.task] = list(a.hint)
        if not a.order and not a.hint:
            raise SystemExit("give --order and/or --hint")
        gd["source"] = a.source
        gd["confirmed_by_user"] = bool(a.confirmed)
        if a.user_statement:
            gd["user_statement"] = a.user_statement
        raw["guidance"] = gd
        save_problem(path, raw)
        print(show(pb.load_problem(path)))
        if a.source != "user":
            print("note: an order not taken from the user's own words is marked orchestrator-draft; weighting with it "
                  "prints a warning and the report carries it", file=sys.stderr)
        return 0
    if a.cmd == "confirm":
        raw = _load_raw(path)
        if a.criteria:
            if not isinstance(raw.get("provenance"), dict):
                raise SystemExit("the problem file has no provenance block (it was not built from a data set)")
            raw["provenance"]["criteria_confirmed_by_user"] = True
        else:
            key = "descriptions" if a.descriptions else "guidance"
            if not raw.get(key):
                raise SystemExit(f"the problem file has no {key} block")
            raw[key]["confirmed_by_user"] = True
        save_problem(path, raw)
        print(f"recorded the user's confirmation in {path}")
        return 0
    # check
    print(show(prob))
    cues = description_cues(prob)
    if cues and not a.allow_cues:
        print("importance cues in descriptions: the described condition refuses them unless --allow-cues is given",
              file=sys.stderr)
        return 2
    w = guidance_warning(prob) if prob.guidance else None
    if w:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
