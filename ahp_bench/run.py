"""CLI: python -m ahp_bench.run [--problem PATH] [--results DIR] [--models-file PATH]
       --models all --conditions direct,cot[,guided] --tasks main,... --reps N --budget USD [--dry-run]

Conditions default to all conditions of the study problem, and to direct and cot for other problems; tasks
default to all tasks of the problem. Conditions: direct (zero-shot, unguided; stored key "lean"), cot
(chain-of-thought, unguided), guided (the study: its verbatim prompts). Other problems also have described,
described-cot (neutral criterion descriptions) and guided, guided-cot (the user's priorities; stored keys
user_guided, user_guided_cot) when the problem file has descriptions / a guidance block (see
ahp_bench.guidance). Aliases: direct, unguided-direct (lean); unguided-cot (cot).
Records are appended to <results>/raw/calls.jsonl; --dry-run writes the exact messages to
<results>/dry_run/ and prints the planned calls and a rough cost estimate without calling any model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ahp, cli, estimate, prompts, store
from .client import api_key_for, call_model
from .config import load_env, reasoning_for
from .parse import ParseError, parse_guided, parse_lean_cot


def _csv(value: str, allowed) -> list[str]:
    items = list(allowed) if value == "all" else [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in allowed]
    if bad:
        raise SystemExit(f"unknown value(s): {', '.join(bad)}; allowed: {', '.join(allowed)}")
    return items


def _conditions(value: str, allowed) -> list[str]:
    """Condition list from the command line; aliases such as "direct" map to the stored key "lean"."""
    if value != "all":
        keys = [prompts.condition_key(v.strip()) for v in value.split(",") if v.strip()]
        # "guided" names the study's verbatim prompts for the study and the user-priority level elsewhere
        keys = ["user_guided" if k == "guided" and "guided" not in allowed and "user_guided" in allowed else k
                for k in keys]
        value = ",".join(keys)
    items = list(allowed) if value == "all" else [v for v in value.split(",") if v]
    bad = [v for v in items if v not in allowed]
    if bad:
        shown = ", ".join(prompts.condition_label(c) for c in allowed)
        raise SystemExit(f"unknown condition(s): {', '.join(bad)}; allowed: {shown}")
    return items


def analyse(condition: str, task: str, reply, problem=None) -> dict:
    """Parse a reply and compute weights/consistency. Raises ParseError."""
    p = prompts.STUDY if problem is None else problem
    names, pairs = p.criteria(task), p.pairs(task)
    out: dict = {}
    if condition == "guided":
        g = parse_guided(reply, len(names))
        out.update(g, guided_labels_expected=prompts.guided_labels(task))
        matrix = g["matrix"]
    else:
        q = parse_lean_cot(reply, len(pairs), cot=prompts.is_cot(condition))
        out.update(q)
        matrix = ahp.matrix_from_answers(len(names), pairs, q["answers"]).tolist()
        out["matrix"] = matrix
    c = ahp.consistency(matrix)
    out.update(weights=dict(zip(names, map(float, c["weights"]))),
               lambda_max=c["lambda_max"], CI=c["CI"], CR=c["CR"])
    return out


def plan(models, conditions, tasks, reps):
    return [(m, c, t, r) for m in models for c in conditions for t in tasks for r in range(1, reps + 1)]


def dry_run(combos, conditions, tasks, out_dir: Path, temperature: float, problem=None, panel=None,
            print_prompts: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in conditions:
        for t in tasks:
            msgs = prompts.build_messages(c, t, problem)
            data = {"prompt_id": prompts.prompt_id(c, t, problem), "prompt_sha256": prompts.prompt_sha256(msgs),
                    "messages": msgs}
            (out_dir / f"{c}_{t}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            if print_prompts:
                print(f"===== {prompts.condition_label(c)} / {t} =====")
                for m in msgs:
                    print(f"--- {m['role']} ---\n{m['content']}")
    models = (panel or {}).get("models")
    default = (panel or {}).get("default_reasoning", {"effort": "none"})
    for k, (m, c, t, r) in enumerate(combos, 1):
        print(f"[{k}/{len(combos)}] {m} {prompts.condition_label(c)} {t} rep{r} reasoning={reasoning_for(m, models, default)} T={temperature}")
    p = prompts.STUDY if problem is None else problem
    planned = [((models[m]["id"], c), m, len(p.pairs(t))) for m, c, t, _ in combos if c != "guided"]
    total, unknown = estimate.estimate(planned, estimate.panel_reference())
    n_guided = sum(1 for x in combos if x[1] == "guided")
    print(f"dry run: {len(combos)} planned calls; messages written to {out_dir}")
    print(estimate.describe(total, unknown, len(planned))
          + (f"; {n_guided} guided calls not estimated" if n_guided else ""))


def main(argv=None, transport=None, root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ahp_bench.run")
    cli.add_common(ap)
    ap.add_argument("--models", default="all")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated: direct (zero-shot, unguided; stored as lean), cot (chain-of-thought, "
                         "unguided), guided; other problems with descriptions also described, described-cot, guided-cot; "
                         "default: all conditions of the study, direct,cot otherwise")
    ap.add_argument("--tasks", default=None, help="default: all tasks of the problem")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--budget", type=float, required=True, help="USD cap per run_tag")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-prompts", action="store_true", help="with --dry-run: print the messages")
    ap.add_argument("--tag", default="default", help="run_tag (budget and resume are per tag)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="requested sampling temperature (default: models file, 0.6 in the study)")
    ap.add_argument("--allow-cues", action="store_true",
                    help="described condition: run even if the descriptions contain importance cues (recorded)")
    a = ap.parse_args(argv)

    ctx = cli.context(a, root)
    p, panel = ctx.problem, ctx.panel
    pm = panel["models"]
    temperature = panel["temperature"] if a.temperature is None else a.temperature
    models = _csv(a.models, list(pm))
    conditions = _conditions(a.conditions or ",".join(p.default_conditions), p.conditions)
    tasks = _csv(a.tasks or ",".join(p.task_names), p.task_names)
    combos = plan(models, conditions, tasks, a.reps)
    from . import guidance
    err = guidance.check_cues(p, conditions, a.allow_cues)
    if err:
        print(err, file=sys.stderr)
        return 2
    guidance.print_warnings(p, conditions)
    if a.dry_run:
        dry_run(combos, conditions, tasks, ctx.results / "dry_run", temperature, p, panel, a.print_prompts)
        return 0

    path = ctx.raw / "calls.jsonl"
    existing = store.read_jsonl(path)
    clash = cli.check_problem_matches(existing, p)
    if clash:
        print(f"{path} holds records whose criteria differ from problem {p.name!r} (tasks: {', '.join(clash)}); "
              "use another --results folder", file=sys.stderr)
        return 2
    load_env(ctx.root / ".env")
    missing = sorted({pm[m].get("env", "?") for m in models if not api_key_for(pm[m], pm)})
    if missing:
        print(f"missing API key variables: {', '.join(missing)} (or OPENROUTER_API_KEY)", file=sys.stderr)
        return 2

    previous = [r for r in existing if r.get("run_tag") == a.tag]
    done = {(r["model_key"], r["condition"], r["task"], r["rep"]) for r in previous if r.get("parse_ok")}
    spent = sum(float(r.get("cost") or 0) for r in previous)
    todo = [x for x in combos if x not in done]
    print(f"run_tag={a.tag}: {len(combos)} planned, {len(combos) - len(todo)} already done, spent ${spent:.4f}")

    for k, (m, c, t, r) in enumerate(todo, 1):
        if spent >= a.budget:
            print(f"budget reached (${spent:.4f} >= ${a.budget:.4f}); stopping")
            break
        msgs = prompts.build_messages(c, t, p)
        rec = {"run_tag": a.tag, "model_key": m, "model_id": pm[m]["id"], "developer": pm[m]["developer"],
               "condition": c, "task": t, "rep": r, "prompt_id": prompts.prompt_id(c, t, p),
               "prompt_sha256": prompts.prompt_sha256(msgs), "messages": msgs,
               "criteria": p.criteria(t), "temperature_requested": temperature}
        if not p.is_study:
            rec.update(problem=p.name, problem_sha256=p.sha256, **prompts.record_fields(c, p, a.allow_cues))
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
        store.append_jsonl(path, rec)
        spent += float(call.get("cost") or 0)
        u = call["usage"]
        cr = f"{rec['CR']:.3f}" if rec.get("parse_ok") else "PARSE_FAIL"
        print(f"[{k}/{len(todo)}] {m} {prompts.condition_label(c)} {t} rep{r} CR={cr} tok={u['prompt_tokens']}/{u['completion_tokens']}"
              f" cost=${float(call.get('cost') or 0):.4f} total=${spent:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
