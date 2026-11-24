"""Command line: `ahp-bench <command> [options]` (or `python -m ahp_bench <command> ...`)."""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "example": ("example", "copy the synthetic example data set and criteria spec to a folder"),
    "eda": ("eda", "stage 0: profile a tabular data set (eda.md, eda.json)"),
    "criteria": ("criteria", "stages 1-2: criteria spec template / validate / build problem file / propose (API)"),
    "hierarchy": ("hierarchy", "show the AHP hierarchy (text tree, hierarchy.md with Mermaid, hierarchy.png)"),
    "guidance": ("guidance", "neutral criterion descriptions and user priorities (described / guided conditions)"),
    "new-problem": ("new_problem", "write a commented problem template"),
    "problem": ("problem", "validate a problem file"),
    "host": ("host", "subscription mode: prompt files for subagents, ingest replies, revision loop (no API)"),
    "run": ("run", "API mode: ask the model panel through OpenRouter (needs a key and --budget)"),
    "revise": ("revise", "API mode: revise inconsistent matrices (optional judge model)"),
    "rank": ("rank", "stage 5 with weights you already have: TOPSIS, ARAS, EDAS rankings"),
    "report": ("report", "tables, figures and report.md from the stored records"),
    "audit": ("audit", "read-only checks of the stored records"),
    "tech-report": ("tech_report", "Turkish technical report of the published study"),
    "reparse": ("reparse", "re-parse stored raw replies with the current parser"),
    "install-skill": ("install_skill", "copy the agent skill to Claude Code, Codex or a project"),
}


def usage() -> str:
    from . import __version__
    lines = [f"ahp-bench {__version__}: LLM respondents in AHP (criteria weights) with optional TOPSIS/ARAS/EDAS ranking",
             "", "usage: ahp-bench <command> [options]   (ahp-bench <command> -h for the options)", "", "commands:"]
    lines += [f"  {name:<14} {desc}" for name, (_, desc) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    if argv[0] in ("-V", "--version"):
        from . import __version__
        print(__version__)
        return 0
    if argv[0] not in COMMANDS:
        print(f"unknown command {argv[0]!r}\n\n{usage()}", file=sys.stderr)
        return 2
    mod = importlib.import_module(f"ahp_bench.{COMMANDS[argv[0]][0]}")
    old = sys.argv[0]
    sys.argv[0] = f"ahp-bench {argv[0]}"
    try:
        return int(mod.main(argv[1:]) or 0)
    finally:
        sys.argv[0] = old


if __name__ == "__main__":
    raise SystemExit(main())
