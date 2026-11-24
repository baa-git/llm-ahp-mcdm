"""Host mode: the AHP questions are answered by the coding agent's own model (Claude Code, Codex, ...)
through independent subagents instead of API calls. No network access, no API key.

    python -m ahp_bench.host prompts --problem P --conditions direct,cot --panelists 5 --reps 1 --out DIR
        writes DIR/prompts/<id>.txt (one prompt per panelist x condition x task x repetition: the system
        message and the user message, exactly the text the API path sends) and DIR/manifest.json.
        Each subagent receives ONE prompt file and writes its reply (the JSON object only) to
        DIR/replies/<id>.json.
    python -m ahp_bench.host ingest --problem P --dir DIR --results RESULTS --tag main --model-name NAME
        parses every reply with the same parser as the API path and appends records in the same schema to
        RESULTS/raw/calls.jsonl (source "host", provider "host", cost 0, temperature null), so report,
        audit and revision work unchanged.
    python -m ahp_bench.host revise-prompts --problem P --dir DIR --results RESULTS --source-tag main --tag revision
        for every inconsistent matrix (CR > 0.10) whose revision chain is not finished: the code computes
        CR and the most contradictory triads and writes the next revision conversation to
        DIR/revise/<id>.txt and DIR/revise_manifest.json. With --judge a judge prompt
        (DIR/judge/<id>.txt) is written first; once its reply DIR/judge_replies/<id>.json exists, the
        next call of revise-prompts uses that feedback (logic preservation is then measured by code only).
    python -m ahp_bench.host revise-ingest --problem P --dir DIR --results RESULTS --tag revision --model-name NAME
        appends the revision replies to RESULTS/raw/revisions.jsonl; repeat revise-prompts / revise-ingest
        until no chain is pending.

    python -m ahp_bench.host criteria-prompt --eda DIR/eda.json --out CRIT
    python -m ahp_bench.host criteria-ingest --dir CRIT --data FILE.csv --out spec.json
        stage 1: one prompt file asking a subagent for a criteria spec (no weights); the reply is checked
        against the data, weight fields are removed, and prompt, reply and spec are logged to
        CRIT/criteria_calls.jsonl (see ahp_bench.criteria).

Limits of host mode: one underlying model; temperature and reasoning settings cannot be set or recorded;
panelists are repetitions of the same model, not different models; token counts and costs are unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import analysis as an
from . import cli, prompts, revise, store
from .parse import ParseError
from .run import _conditions, _csv, analyse

SOURCE = "host"
MANIFEST = "manifest.json"
REVISE_MANIFEST = "revise_manifest.json"
HOST_NOTE = ("Answered by the host agent's own model through independent subagents (host mode): temperature, "
             "reasoning, tokens and cost are not controlled or reported.")


def render(messages) -> str:
    """Prompt file text: every message under a role header, contents verbatim."""
    return "\n\n".join(f"=== {m['role'].upper()} ===\n{m['content']}" for m in messages) + "\n"


def instructions(reply_dir: str) -> str:
    return ("Host-mode prompt files. Give each file in this folder to ONE fresh subagent, verbatim and with "
            "nothing else; the subagent answers as the respondent described in the SYSTEM part and replies "
            f"with the requested JSON object only, which is saved unchanged as {reply_dir}/<same name>.json. "
            "Never answer the questions yourself and never show one subagent another subagent's answers.\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _read_reply(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8-sig")
    return text if text.strip() else None


def _slug(condition) -> str:
    """File-name label of a condition: direct, cot, guided, described, described_cot, guided_cot."""
    return prompts.condition_label(condition).lower().replace(" ", "_")


def _problem_label(p) -> str:
    return str(p.path) if p.path else p.name


# ---------------------------------------------------------------- first answers

def write_prompts(problem, conditions, tasks, panelists: int, reps: int, out: Path, allow_cues: bool = False) -> dict:
    out = Path(out)
    entries = []
    for c in conditions:
        for t in tasks:
            msgs = prompts.build_messages(c, t, problem)
            sha = prompts.prompt_sha256(msgs)
            for panelist in range(1, panelists + 1):
                for r in range(1, reps + 1):
                    pid = f"{_slug(c)}_{t}_p{panelist:02d}_r{r:02d}"
                    _write(out / "prompts" / f"{pid}.txt", render(msgs))
                    entries.append({"id": pid, "condition": c, "task": t, "panelist": panelist, "rep_in_panelist": r,
                                    "rep": (panelist - 1) * reps + r, "prompt_file": f"prompts/{pid}.txt",
                                    "reply_file": f"replies/{pid}.json", "prompt_id": prompts.prompt_id(c, t, problem),
                                    "prompt_sha256": sha})
    manifest = {"mode": SOURCE, "problem": problem.name, "problem_file": _problem_label(problem),
                "problem_sha256": problem.sha256, "panelists": panelists, "reps_per_panelist": reps,
                "conditions": list(conditions), "tasks": list(tasks), "created_utc": _now(), "note": HOST_NOTE,
                "entries": entries}
    if allow_cues:
        manifest["allow_cues"] = True
    _write(out / MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    _write(out / "prompts" / "README.txt", instructions("replies"))
    (out / "replies").mkdir(parents=True, exist_ok=True)
    return manifest


def _load_manifest(path: Path, problem) -> dict:
    if not path.is_file():
        raise SystemExit(f"{path} not found; write the prompts first")
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("problem") != problem.name:
        raise SystemExit(f"{path} belongs to problem {m.get('problem')!r}, not {problem.name!r} (use --problem)")
    return m


def _host_fields(model_name: str) -> dict:
    return {"source": SOURCE, "model_key": model_name, "model_id": f"host/{model_name}", "developer": SOURCE,
            "provider": SOURCE, "cost": 0.0, "usage": {}, "temperature_requested": None,
            "requested_params": {"source": SOURCE, "temperature": None, "reasoning": None},
            "applied_params": {"source": SOURCE, "temperature": None, "reasoning": None},
            "http_attempts": [], "latency_s": None, "finish_reason": None, "host_note": HOST_NOTE}


def ingest(problem, folder: Path, results: Path, tag: str, model_name: str) -> dict:
    folder, raw = Path(folder), Path(results) / "raw"
    m = _load_manifest(folder / MANIFEST, problem)
    path = raw / "calls.jsonl"
    existing = store.read_jsonl(path)
    clash = cli.check_problem_matches(existing, problem)
    if clash:
        raise SystemExit(f"{path} holds records of another problem (tasks: {', '.join(clash)}); use another --results")
    done = {r.get("host_prompt_id") for r in existing if r.get("run_tag") == tag and r.get("source") == SOURCE}
    out = {"written": 0, "parsed": 0, "missing": [], "already": 0, "unparsed": []}
    for e in m["entries"]:
        if e["id"] in done:
            out["already"] += 1
            continue
        reply = _read_reply(folder / e["reply_file"])
        if reply is None:
            out["missing"].append(e["id"])
            continue
        msgs = prompts.build_messages(e["condition"], e["task"], problem)
        if prompts.prompt_sha256(msgs) != e["prompt_sha256"]:
            raise SystemExit(f"{e['id']}: the prompt built from the problem file differs from the manifest "
                             "(problem file changed after the prompts were written)")
        rec = {"run_tag": tag, **_host_fields(model_name), "condition": e["condition"], "task": e["task"],
               "rep": e["rep"], "panelist": e["panelist"], "host_prompt_id": e["id"],
               "prompt_id": e["prompt_id"], "prompt_sha256": e["prompt_sha256"], "messages": msgs,
               "criteria": problem.criteria(e["task"]), "ok": True, "error": None, "content": reply,
               "raw_reply": reply, "generation_id": f"host-{e['id']}", "timestamp_utc": _now(),
               "reply_sha256": hashlib.sha256(reply.encode("utf-8")).hexdigest()}
        if not problem.is_study:
            rec.update(problem=problem.name, problem_sha256=problem.sha256,
                       **prompts.record_fields(e["condition"], problem, bool(m.get("allow_cues"))))
        try:
            rec.update(analyse(e["condition"], e["task"], reply, problem), parse_ok=True, parse_error=None)
            out["parsed"] += 1
        except ParseError as err:
            rec.update(parse_ok=False, parse_error=str(err))
            out["unparsed"].append(e["id"])
        store.append_jsonl(path, rec)
        out["written"] += 1
    return out


# ---------------------------------------------------------------- revision

def _chain_id(key, number: int) -> str:
    m, c, t, rep = key
    return f"{_slug(c)}_{t}_rep{rep:03d}_rev{number}"


def revise_prompts(problem, folder: Path, results: Path, source_tag: str, tag: str, conditions, max_revisions: int,
                   judge: bool = False) -> dict:
    folder, raw = Path(folder), Path(results) / "raw"
    records, _ = an.load_records(raw / "calls.jsonl")
    rev_all, _ = an.load_records(raw / "revisions.jsonl")
    revs = revise.revision_records(rev_all, tag, source_tag)
    need, todo = revise.plan_chains(records, revs, source_tag, conditions, None, max_revisions, problem)
    entries, judge_pending = [], []
    for key, src in todo:
        recs = list(revs.get(key, []))
        st = revise.next_state(src, recs, problem)
        cid = _chain_id(key, st["number"])
        msgs = st["messages"]
        fb_fields = {}
        if judge and not st["resend"]:
            jmsgs = revise.judge_feedback_messages(key[2], st["answers"], st["CR"],
                                                   st["best"].get("ranking") if prompts.is_cot(key[1]) else None, problem)
            jreply = _read_reply(folder / "judge_replies" / f"{cid}.json")
            if jreply is None:
                _write(folder / "judge" / f"{cid}.txt", render(jmsgs))
                judge_pending.append(cid)
                continue
            fb = revise.parse_feedback(jreply)
            if fb:
                msgs = msgs[:-1] + [{"role": "user", "content": revise.judge_revision_message(fb, key[2], st["answers"],
                                                                                              problem)}]
            fb_fields = {"judge_source": SOURCE, "judge_feedback": fb, "judge_feedback_ok": fb is not None,
                         "judge_feedback_reused": False, "judge_messages": jmsgs, "judge_raw_reply": jreply}
        elif judge and st["resend"]:
            prev = recs[-1]
            fb_fields = {k: prev.get(k) for k in ("judge_source", "judge_feedback", "judge_feedback_ok")}
            fb_fields["judge_feedback_reused"] = True
        _write(folder / "revise" / f"{cid}.txt", render(msgs))
        entries.append({"id": cid, "chain_key": list(key), "revision_number": st["number"],
                        "based_on_answers": list(st["answers"]), "based_on_CR": st["CR"],
                        "based_on_revision": st["based_on_revision"], "source_CR": revise._cr(src),
                        "max_revisions": max_revisions, "messages": msgs, "prompt_sha256": prompts.prompt_sha256(msgs),
                        "prompt_file": f"revise/{cid}.txt", "reply_file": f"revise_replies/{cid}.json", **fb_fields})
    manifest = {"mode": SOURCE, "problem": problem.name, "problem_sha256": problem.sha256, "source_tag": source_tag,
                "tag": tag, "max_revisions": max_revisions, "judge": judge, "created_utc": _now(), "note": HOST_NOTE,
                "chains_needing_revision": len(need), "entries": entries, "judge_pending": judge_pending}
    _write(folder / REVISE_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    (folder / "revise_replies").mkdir(parents=True, exist_ok=True)
    _write(folder / "revise" / "README.txt", instructions("revise_replies"))
    if judge:
        (folder / "judge_replies").mkdir(parents=True, exist_ok=True)
        _write(folder / "judge" / "README.txt", instructions("judge_replies").replace(
            "answers as the respondent described in the SYSTEM part", "acts as described in the SYSTEM part"))
    return manifest


def revise_ingest(problem, folder: Path, results: Path, tag: str, model_name: str) -> dict:
    folder, raw = Path(folder), Path(results) / "raw"
    m = _load_manifest(folder / REVISE_MANIFEST, problem)
    if m["tag"] != tag:
        raise SystemExit(f"{folder / REVISE_MANIFEST} was written for tag {m['tag']!r}, not {tag!r}")
    path = raw / "revisions.jsonl"
    rev_all, _ = an.load_records(path)
    revs = revise.revision_records(rev_all, tag, m["source_tag"])
    out = {"written": 0, "parsed": 0, "missing": [], "stale": [], "consistent": 0}
    for e in m["entries"]:
        key = tuple(e["chain_key"])
        if len(revs.get(key, [])) + 1 != e["revision_number"]:
            out["stale"].append(e["id"])   # already ingested or the chain moved on: write prompts again
            continue
        reply = _read_reply(folder / e["reply_file"])
        if reply is None:
            out["missing"].append(e["id"])
            continue
        mk, c, t, rep = key
        cr = e["based_on_CR"]
        pid = f"revision_{'judge_' if e.get('judge_feedback_ok') is not None else ''}{c}_{t}_{revise.REVISION_PROMPT_VERSION}"
        rec = {"run_tag": tag, "source_tag": m["source_tag"], **_host_fields(model_name), "model_key": mk,
               "model_id": f"host/{model_name}", "condition": c, "task": t, "rep": rep,
               "revision_number": e["revision_number"], "max_revisions": e["max_revisions"], "source_CR": e["source_CR"],
               "based_on_answers": e["based_on_answers"], "based_on_CR": cr, "based_on_revision": e["based_on_revision"],
               "prompt_id": pid if problem.is_study else f"{problem.name}:{pid}", "prompt_sha256": e["prompt_sha256"],
               "messages": e["messages"], "criteria": problem.criteria(t), "host_prompt_id": e["id"], "ok": True,
               "error": None, "content": reply, "raw_reply": reply, "generation_id": f"host-{e['id']}",
               "timestamp_utc": _now()}
        for k in ("judge_source", "judge_feedback", "judge_feedback_ok", "judge_feedback_reused",
                  "judge_messages", "judge_raw_reply"):
            if k in e:
                rec[k] = e[k]
        if not problem.is_study:
            rec.update(problem=problem.name, problem_sha256=problem.sha256)
        try:
            rec.update(analyse(c, t, reply, problem), parse_ok=True, parse_error=None)
            out["parsed"] += 1
        except ParseError as err:
            rec.update(parse_ok=False, parse_error=str(err))
        valid = revise._valid_revision(rec, problem)
        rec["best_CR_so_far"] = min(cr, float(rec["CR"])) if valid else cr
        if valid:
            sources = revise.source_matrices(an.load_records(raw / "calls.jsonl")[0], m["source_tag"], (c,), {mk}, problem)
            src = sources.get(key)
            if src is not None:
                rec["logic_metrics"] = revise.logic_metrics(t, src["answers"], rec["answers"], problem)
            out["consistent"] += an.is_consistent(rec["CR"])
        store.append_jsonl(path, rec)
        revs.setdefault(key, []).append(rec)
        out["written"] += 1
    return out


# ---------------------------------------------------------------- criteria proposal (stage 1)

CRITERIA_MANIFEST = "criteria_manifest.json"
CRITERIA_REPLY = "criteria_reply.json"


def criteria_host(a) -> int:
    from . import criteria as cr
    if a.cmd == "criteria-prompt":
        eda_path = cli.resolve(a.eda)
        eda = json.loads(eda_path.read_text(encoding="utf-8-sig"))
        msgs = cr.criteria_messages(eda, a.goal)
        out = cli.resolve(a.out)
        _write(out / "criteria_prompt.txt", render(msgs))
        _write(out / CRITERIA_MANIFEST, json.dumps({
            "mode": SOURCE, "kind": "criteria_proposal", "prompt_id": cr.PROMPT_ID,
            "prompt_sha256": cr.prompt_sha256(msgs), "eda_file": eda_path.name, "data_file": eda.get("data_file"),
            "data_sha256": eda.get("data_sha256"), "goal": a.goal, "messages": msgs, "reply_file": CRITERIA_REPLY,
            "created_utc": _now()}, ensure_ascii=False, indent=1) + "\n")
        _write(out / "README.txt", "Give criteria_prompt.txt verbatim to ONE fresh subagent and save its JSON reply "
               f"unchanged as {CRITERIA_REPLY}. The reply proposes criteria and indicators only; weights come from the "
               "pairwise step.\n")
        print(f"wrote {out / 'criteria_prompt.txt'}; give it verbatim to ONE fresh subagent and save the reply as "
              f"{out / CRITERIA_REPLY}; then: ahp-bench host criteria-ingest --dir {a.out} --data <data file> --out <spec>")
        return 0
    folder = cli.resolve(a.dir)
    mpath = folder / CRITERIA_MANIFEST
    if not mpath.is_file():
        raise SystemExit(f"{mpath} not found; write the prompt first (ahp-bench host criteria-prompt)")
    m = json.loads(mpath.read_text(encoding="utf-8"))
    reply = _read_reply(folder / m["reply_file"])
    if reply is None:
        raise SystemExit(f"{folder / m['reply_file']} not found or empty")
    from .tabular import read_table
    data = cli.resolve(a.data)
    table = read_table(data)
    if m.get("data_sha256") and table.sha256 != m["data_sha256"]:
        raise SystemExit(f"{data.name} differs from the data file profiled for the prompt ({m.get('data_file')})")
    eda = {"data_sha256": m.get("data_sha256"), "data_file": m.get("data_file"), "prompt_eda_file": m.get("eda_file")}
    return cr.finish_proposal(reply, m["messages"], eda, data, cli.resolve(a.out), a.proposed_by,
                              a.name or cr._name_from(data), a.goal or m.get("goal"), a.role,
                              folder / "criteria_calls.jsonl",
                              {"source": SOURCE, "model_id": f"host/{a.proposed_by}", "host_note": HOST_NOTE})


# ---------------------------------------------------------------- CLI

def _problem(a):
    from . import problem as pb
    if not a.problem:
        raise SystemExit("--problem is required (try the bundled example: " + str(pb.EXAMPLE_PROBLEM_PATH) + ")")
    return pb.load_problem(cli.resolve(a.problem))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ahp_bench.host", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts", help="write one prompt file per panelist x condition x task x repetition")
    p.add_argument("--problem", required=True)
    p.add_argument("--conditions", default=None,
                   help="comma-separated: direct (zero-shot, unguided; stored as lean), cot (chain-of-thought, unguided), "
                        "guided; with descriptions in the problem file also described, described-cot, guided-cot; "
                        "default: direct,cot")
    p.add_argument("--allow-cues", action="store_true",
                   help="described condition: write prompts even if the descriptions contain importance cues (recorded)")
    p.add_argument("--tasks", default=None, help="default: all tasks of the problem")
    p.add_argument("--panelists", type=int, default=5, help="independent subagents per condition and task")
    p.add_argument("--reps", type=int, default=1, help="repetitions per panelist (default 1)")
    p.add_argument("--out", required=True, help="folder for prompts/, replies/ and manifest.json")
    p = sub.add_parser("ingest", help="parse reply files and append records to <results>/raw/calls.jsonl")
    p.add_argument("--problem", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--tag", default="main")
    p.add_argument("--model-name", required=True, help="name of the host model, e.g. the model your agent runs on")
    p = sub.add_parser("revise-prompts", help="write the next revision prompt of every unfinished chain")
    p.add_argument("--problem", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--source-tag", default="main")
    p.add_argument("--tag", default="revision")
    p.add_argument("--conditions", default="direct,cot", help="direct (stored as lean) and/or cot")
    p.add_argument("--max-revisions", type=int, default=5)
    p.add_argument("--judge", action="store_true",
                   help="the host agent writes the feedback as judge (only if the user asks); default: code feedback")
    p = sub.add_parser("revise-ingest", help="append revision replies to <results>/raw/revisions.jsonl")
    p.add_argument("--problem", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--tag", default="revision")
    p.add_argument("--model-name", required=True)
    p = sub.add_parser("criteria-prompt", help="stage 1: prompt file asking a subagent for a criteria spec (from eda.json)")
    p.add_argument("--eda", required=True, help="eda.json written by ahp-bench eda")
    p.add_argument("--out", required=True, help="folder for criteria_prompt.txt and criteria_manifest.json")
    p.add_argument("--goal", default=None, help="noun phrase naming the decision (else the subagent proposes one)")
    p = sub.add_parser("criteria-ingest", help="stage 1: validate the subagent's reply against the data, write the spec")
    p.add_argument("--dir", required=True, help="folder of criteria-prompt (reply: criteria_reply.json)")
    p.add_argument("--data", required=True, help="the data file profiled by ahp-bench eda")
    p.add_argument("--out", required=True, help="criteria spec to write")
    p.add_argument("--proposed-by", default="host agent")
    p.add_argument("--name", default=None, help="problem identifier (default: from the data file name)")
    p.add_argument("--goal", default=None)
    p.add_argument("--role", default=None, help="respondent line for the weighting prompts")
    a = ap.parse_args(argv)
    if a.cmd in ("criteria-prompt", "criteria-ingest"):
        return criteria_host(a)
    prob = _problem(a)
    if a.cmd == "prompts":
        if a.panelists < 1 or a.reps < 1:
            raise SystemExit("--panelists and --reps must be >= 1")
        default = ",".join(c for c in prob.default_conditions if c != "guided")
        conds = _conditions(a.conditions or default, prob.conditions)
        tasks = _csv(a.tasks or ",".join(prob.task_names), prob.task_names)
        from . import guidance
        err = guidance.check_cues(prob, conds, a.allow_cues)
        if err:
            print(err, file=sys.stderr)
            return 2
        guidance.print_warnings(prob, conds)
        m = write_prompts(prob, conds, tasks, a.panelists, a.reps, cli.resolve(a.out), a.allow_cues)
        print(f"wrote {len(m['entries'])} prompt files to {cli.resolve(a.out) / 'prompts'}; replies go to "
              f"{cli.resolve(a.out) / 'replies'} (one JSON reply per prompt, same file name, .json)")
    elif a.cmd == "ingest":
        r = ingest(prob, cli.resolve(a.dir), cli.resolve(a.results), a.tag, a.model_name)
        print(f"tag={a.tag}: {r['written']} records written ({r['parsed']} parsed), {r['already']} already ingested, "
              f"{len(r['missing'])} replies missing" + (f"; not parsed: {', '.join(r['unparsed'])}" if r["unparsed"] else ""))
    elif a.cmd == "revise-prompts":
        if a.max_revisions < 1:
            raise SystemExit("--max-revisions must be >= 1")
        conds = _conditions(a.conditions, revise.REVISION_CONDITIONS)
        m = revise_prompts(prob, cli.resolve(a.dir), cli.resolve(a.results), a.source_tag, a.tag, conds,
                           a.max_revisions, a.judge)
        print(f"{m['chains_needing_revision']} matrices with CR > 0.10; {len(m['entries'])} revision prompts written to "
              f"{cli.resolve(a.dir) / 'revise'}" + (f"; {len(m['judge_pending'])} judge prompts pending in "
                                                     f"{cli.resolve(a.dir) / 'judge'}" if m["judge_pending"] else ""))
    else:
        r = revise_ingest(prob, cli.resolve(a.dir), cli.resolve(a.results), a.tag, a.model_name)
        print(f"tag={a.tag}: {r['written']} revision records written ({r['parsed']} parsed, {r['consistent']} now "
              f"CR <= 0.10), {len(r['missing'])} replies missing, {len(r['stale'])} stale entries skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
