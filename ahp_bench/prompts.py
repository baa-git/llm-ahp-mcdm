"""Message builders for the prompt conditions: unguided (stored key "lean"), CoT ("cot") and, for the
study only, guided.

unguided/CoT messages are built from a problem definition (ahp_bench.problem). With the study problem
(problems/supplier_selection.json, the default) they are byte-identical to the prompts used in the
study; python -m ahp_bench.audit checks this against every stored record. The guided condition uses
the original thesis prompt files in problems/guided and exists only for the study problem.
"""
from __future__ import annotations

import hashlib
import json
import re

from . import problem as pb
PROMPT_VERSION = "v2"
GUIDED_DIR = pb.PROBLEMS_DIR / "guided"   # the study's original guided prompt files (byte-exact)

STUDY = pb.load_problem(pb.STUDY_PROBLEM_PATH)

MAIN_CRITERIA = STUDY.criteria("main")
DELIVERY_SUBCRITERIA = STUDY.criteria("delivery")
CRITERIA = {t: STUDY.criteria(t) for t in STUDY.task_names}
# Question order = the order of the questions the experts were asked: (0,1),(0,2),...,(n-2,n-1)
PAIRS = {t: STUDY.pairs(t) for t in STUDY.task_names}
CONDITIONS = ("lean", "cot", "guided")
TASKS = STUDY.task_names
# Conditions are described on two axes: prompting technique (zero-shot = no worked examples and no reasoning
# instruction, or chain-of-thought) and weight guidance (guided = priority order and importance hints in the
# prompt, or unguided). Three conditions were run: guided (zero-shot, guided), direct (stored key "lean":
# zero-shot, unguided) and CoT (stored key "cot": chain-of-thought, unguided); CoT x guided was not run.
# Records, prompt ids and hashes keep the keys; every output a person reads uses the labels below:
#   condition_label     short label in "Condition" cells: guided / direct / CoT
#   condition_guidance  "Guidance" cells: guided / unguided
#   condition_technique "Technique" cells: zero-shot / CoT
#   condition_full      full label in titles and row labels
#   condition_column    label inside column names of wide tables: guided / unguided_direct / unguided_cot
# The command line accepts the keys and the aliases below, case-insensitively.
# Problems other than the study have two more guidance levels, each with both techniques (stored keys):
#   described / described_cot       criterion names plus the neutral descriptions of the problem file
#   user_guided / user_guided_cot   descriptions plus the user's priority order and hints (guidance block)
# The study's "guided" key keeps its verbatim prompt files; on the command line "guided" means user_guided
# for every other problem.
CONDITION_LABELS = {"lean": "direct", "cot": "CoT", "guided": "guided", "described": "described",
                    "described_cot": "described CoT", "user_guided": "guided", "user_guided_cot": "guided CoT"}
CONDITION_STRATEGY = {"lean": "unguided", "cot": "unguided", "guided": "guided", "described": "described",
                      "described_cot": "described", "user_guided": "guided", "user_guided_cot": "guided"}
CONDITION_TECHNIQUE = {"lean": "zero-shot", "cot": "CoT", "guided": "zero-shot", "described": "zero-shot",
                       "described_cot": "CoT", "user_guided": "zero-shot", "user_guided_cot": "CoT"}
CONDITION_FULL = {"lean": "Unguided, zero-shot (direct)", "cot": "Unguided, CoT", "guided": "Guided (zero-shot)",
                  "described": "Described, zero-shot", "described_cot": "Described, CoT",
                  "user_guided": "Guided by user priorities, zero-shot", "user_guided_cot": "Guided by user priorities, CoT"}
CONDITION_COLUMN = {"lean": "unguided_direct", "cot": "unguided_cot", "guided": "guided",
                    "described": "described_direct", "described_cot": "described_cot",
                    "user_guided": "guided_direct", "user_guided_cot": "guided_cot"}
CONDITION_ALIASES = {"lean": "lean", "direct": "lean", "unguided-direct": "lean", "unguided_direct": "lean",
                     "unguided:direct": "lean", "cot": "cot", "unguided-cot": "cot", "unguided_cot": "cot",
                     "unguided:cot": "cot", "guided": "guided",
                     "described": "described", "described-direct": "described", "described_direct": "described",
                     "described-cot": "described_cot", "described_cot": "described_cot",
                     "user_guided": "user_guided", "user-guided": "user_guided", "guided-direct": "user_guided",
                     "guided_direct": "user_guided", "guided-cot": "user_guided_cot", "guided_cot": "user_guided_cot",
                     "user_guided_cot": "user_guided_cot", "user-guided-cot": "user_guided_cot"}
GENERIC_LEVELS = {"described": "described", "described_cot": "described", "user_guided": "guided",
                  "user_guided_cot": "guided"}
_KEY_OF_LABEL = {}
for _d in (CONDITION_LABELS, CONDITION_FULL, CONDITION_COLUMN):
    for _k, _v in _d.items():
        _KEY_OF_LABEL.setdefault(_v, _k)


def is_cot(condition) -> bool:
    """Chain-of-thought technique (reply carries reasoning and a ranking)."""
    return condition in ("cot", "described_cot", "user_guided_cot")


def guidance_level(condition) -> str:
    """unguided / described / guided."""
    return CONDITION_STRATEGY.get(condition, "")


def condition_label(condition) -> str:
    """Short display label of a stored condition key ("lean" -> "direct", "cot" -> "CoT")."""
    return CONDITION_LABELS.get(condition, condition)


def _key(condition):
    return condition if condition in CONDITION_STRATEGY else _KEY_OF_LABEL.get(condition)


def condition_guidance(condition) -> str:
    """Weight guidance of a stored key or display label ("lean", "direct", "CoT" -> "unguided"); "" if none."""
    return CONDITION_STRATEGY.get(_key(condition), "")


def condition_technique(condition) -> str:
    """Prompting technique of a stored key or display label ("cot", "CoT" -> "CoT"; others "zero-shot")."""
    return CONDITION_TECHNIQUE.get(_key(condition), "")


def condition_full(condition) -> str:
    """Full display label ("lean" -> "Unguided, zero-shot (direct)")."""
    return CONDITION_FULL.get(condition, condition)


def condition_column(condition) -> str:
    """Label used inside column names of wide tables ("lean" -> "unguided_direct")."""
    return CONDITION_COLUMN.get(condition, condition)


def condition_key(name: str) -> str:
    """Stored key of a condition name given on the command line ("direct", "unguided-direct" -> "lean")."""
    n = str(name).strip().lower().replace(" ", "")
    return CONDITION_ALIASES.get(n, name)

SYSTEM_TAIL = "\nAnswer the following questions. Respond only in the requested JSON format."

SCALE_TEXT = """Answer scale (left criterion = A, right criterion = B):
9   : A is extremely more important than B.
8   : A is between very strongly and extremely more important than B. (Intermediate value)
7   : A is very strongly more important than B.
6   : A is between strongly and very strongly more important than B. (Intermediate value)
5   : A is strongly more important than B.
4   : A is between moderately and strongly more important than B. (Intermediate value)
3   : A is moderately more important than B.
2   : A is between equally and moderately more important than B. (Intermediate value)
1   : A and B are equally important.
1/2 : B is between equally and moderately more important than A. (Intermediate value)
1/3 : B is moderately more important than A.
1/4 : B is between moderately and strongly more important than A. (Intermediate value)
1/5 : B is strongly more important than A.
1/6 : B is between strongly and very strongly more important than A. (Intermediate value)
1/7 : B is very strongly more important than A.
1/8 : B is between very strongly and extremely more important than A. (Intermediate value)
1/9 : B is extremely more important than A."""

LEAN_FORMAT = (
    "Response format (only this JSON):\n"
    '{"answers": [{"question": 1, "value": "3"}, {"question": 2, "value": "1/5"}, ...]}'
)

COT_TEMPLATE = """Before answering, think step by step:
1. State in one sentence what each criterion represents in {goal}.
2. Rank the criteria from most to least important in your view.
3. Choose each value so that it is consistent with this ranking
   (for example, if A is more important than B and B than C, A must be even more important than C).

Response format (only this JSON):
{{"reasoning": "your brief reasoning for steps 1-3",
 "ranking": ["most important criterion", "...", "least important criterion"],
 "answers": [{{"question": 1, "value": "3"}}, ...]}}"""

GUIDED_FILES = {
    "system": "eda_ahp_system_prompt.txt",
    "main": "eda_main_criteria_user_prompt.json",
    "delivery": "eda_delivery_subcriteria_user_prompt.json",
}


def _p(problem):
    return STUDY if problem is None else problem


def lean_system(problem=None) -> str:
    return _p(problem).role + SYSTEM_TAIL


def cot_format(problem=None) -> str:
    return COT_TEMPLATE.format(goal=_p(problem).goal)


def intro(task: str, problem=None) -> str:
    p = _p(problem)
    return p.task(task).intro(p.goal)


LEAN_SYSTEM = lean_system()
COT_FORMAT = cot_format()
INTRO = {t: intro(t) for t in TASKS}


def conditions_for(problem=None) -> tuple:
    return _p(problem).conditions


def question_lines(task: str, problem=None) -> list[str]:
    p = _p(problem)
    names, pairs = p.criteria(task), p.pairs(task)
    width = len(str(len(pairs))) + 1  # pad "1." so that it aligns with "10."
    return [f"{f'{k}.':<{width}} Left: {names[i]} | Right: {names[j]}" for k, (i, j) in enumerate(pairs, 1)]


def criteria_block(task: str, problem) -> str:
    """Neutral descriptions of the task's criteria (described and user_guided conditions)."""
    d = problem.descriptions["criteria"]
    word = "sub-criteria" if problem.task(task).parent else "criteria"
    return f"The {word}:\n" + "\n".join(f"- {c}: {d[c].strip()}" for c in problem.criteria(task))


def guidance_block(task: str, problem) -> str | None:
    """The user's stated priorities for a task (user_guided conditions); None when the task has none."""
    g = problem.guidance
    order, hints = (g.get("order") or {}).get(task), (g.get("hints") or {}).get(task) or []
    if not order and not hints:
        return None
    lines = ["Priorities stated by the decision maker:"]
    if order:
        lines.append("Order of importance, most important first: "
                     + ", ".join(f"{k}. {c}" for k, c in enumerate(order, 1)))
    lines += [f"- {h.strip()}" for h in hints]
    return "\n".join(lines)


def _generic_user(condition: str, task: str, problem) -> str:
    if not problem.descriptions:
        raise ValueError(f"condition {condition_label(condition)!r} needs criterion descriptions in the problem file")
    parts = [intro(task, problem), criteria_block(task, problem)]
    if GENERIC_LEVELS[condition] == "guided":
        if not problem.guidance:
            raise ValueError("the guided condition needs a guidance block in the problem file")
        block = guidance_block(task, problem)
        if block:
            parts.append(block)
    fmt = cot_format(problem) if is_cot(condition) else LEAN_FORMAT
    return "\n\n".join(parts + [SCALE_TEXT, "Questions:\n" + "\n".join(question_lines(task, problem)), fmt])


def _lean_cot_user(condition: str, task: str, problem=None) -> str:
    fmt = LEAN_FORMAT if condition == "lean" else cot_format(problem)
    return "\n\n".join([intro(task, problem), SCALE_TEXT, "Questions:\n" + "\n".join(question_lines(task, problem)), fmt])


def guided_system() -> str:
    # The export was written on Windows in text mode (CRLF); the thesis code sent the
    # Python string HIERARCHICAL_AHP_SYSTEM_PROMPT with LF, so CRLF is normalised to LF.
    return (GUIDED_DIR / GUIDED_FILES["system"]).read_bytes().decode("utf-8").replace("\r\n", "\n")


def guided_payload(task: str) -> dict:
    return json.loads((GUIDED_DIR / GUIDED_FILES[task]).read_bytes().decode("utf-8"))


def guided_user(task: str) -> str:
    # Thesis (src/llm/prompts.py): {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    # default separators, no indent. The exported file is byte-identical to this string.
    return json.dumps(guided_payload(task), ensure_ascii=False)


def guided_prompted_order(task: str) -> dict:
    """What the guided prompt says about the order of the criteria of a task, read from the prompt file:
    main -> the numbered "typical priority order" (instructions.priority_guidance) as indices into the
    task's criteria, best first; delivery -> only the criterion named "the PRIMARY positive indicator"
    (instructions.example_for_delivery), no full order. Returns {"order": [..] or None, "top": i, "source": ...}."""
    names = [n.lower() for n in CRITERIA[task]]
    ins = guided_payload(task).get("instructions", {})
    if "priority_guidance" in ins:
        words = re.findall(r"\d\.\s*[^()\d]*\(([A-Za-z]+)\)", ins["priority_guidance"])
        order = [next(i for i, n in enumerate(names) if n.startswith(w.lower())) for w in words]
        if sorted(order) != list(range(len(names))):
            raise ValueError(f"priority order of the guided {task} prompt does not name every criterion once")
        return {"order": order, "top": order[0], "source": "instructions.priority_guidance"}
    m = re.search(r"-\s*([A-Za-z ]+?) is the PRIMARY", ins.get("example_for_delivery", ""))
    if not m:
        raise ValueError(f"the guided {task} prompt states no order")
    return {"order": None, "top": names.index(m.group(1).strip().lower()), "source": "instructions.example_for_delivery"}


def guided_labels(task: str) -> list[str]:
    p = guided_payload(task)
    return [c["name"] for c in p["main_criteria"]] if task == "main" else list(p["sub_criteria"])


def build_messages(condition: str, task: str, problem=None) -> list[dict]:
    p = _p(problem)
    if task not in p.task_names:
        raise ValueError(f"unknown task {task!r}")
    if condition == "guided":
        if not p.guided or task not in GUIDED_FILES:
            raise ValueError("the guided condition exists only for the study problem (supplier_selection)")
        return [{"role": "system", "content": guided_system()}, {"role": "user", "content": guided_user(task)}]
    if condition in ("lean", "cot"):
        return [{"role": "system", "content": lean_system(p)},
                {"role": "user", "content": _lean_cot_user(condition, task, p)}]
    if condition in GENERIC_LEVELS:
        if p.is_study:
            raise ValueError("the study problem has only the conditions direct, CoT and its verbatim guided prompts")
        return [{"role": "system", "content": lean_system(p)},
                {"role": "user", "content": _generic_user(condition, task, p)}]
    raise ValueError(f"unknown condition {condition!r}")


def prompt_id(condition: str, task: str, problem=None) -> str:
    p = _p(problem)
    pid = f"{condition}_{task}_{'thesis' if condition == 'guided' else PROMPT_VERSION}"
    return pid if p.is_study else f"{p.name}:{pid}"


def block_sha256(block) -> str | None:
    return None if block is None else hashlib.sha256(json.dumps(block, ensure_ascii=False, sort_keys=True)
                                                    .encode("utf-8")).hexdigest()


def record_fields(condition: str, problem, cues_allowed: bool = False) -> dict:
    """Optional record fields for problems other than the study: guidance level and technique, and for the
    described / guided levels the hashes and status of the descriptions and guidance blocks. The study's
    records get none (their schema is unchanged)."""
    if problem is None or problem.is_study:
        return {}
    out = {"guidance_level": guidance_level(condition), "technique": condition_technique(condition)}
    if condition in GENERIC_LEVELS:
        d = problem.descriptions or {}
        out.update(descriptions_sha256=block_sha256(d.get("criteria")), descriptions_drafted_by=d.get("drafted_by"),
                   descriptions_confirmed_by_user=bool(d.get("confirmed_by_user")))
        if GENERIC_LEVELS[condition] == "described":
            out["cues_allowed"] = bool(cues_allowed)
        else:
            g = problem.guidance or {}
            out.update(guidance_sha256=block_sha256({k: g.get(k) for k in ("order", "hints")}),
                       guidance_source=g.get("source"), guidance_confirmed_by_user=bool(g.get("confirmed_by_user")))
    return out


def prompt_sha256(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()
