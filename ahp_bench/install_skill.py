"""Copy the llm-ahp-mcdm agent skill into an agent's skills directory (no network).

ahp-bench install-skill --claude     # ~/.claude/skills/llm-ahp-mcdm (Claude Code, all projects)
ahp-bench install-skill --codex      # ~/.codex/skills/llm-ahp-mcdm (Codex)
ahp-bench install-skill --project    # ./.claude/skills/llm-ahp-mcdm and a pointer in ./AGENTS.md
ahp-bench install-skill DIR          # DIR/llm-ahp-mcdm (any other agent)
Options: --force overwrites an existing copy; --dry-run only prints what would be copied.
(In a clone without installing: python -m ahp_bench.install_skill ... or python scripts/install_skill.py ...)

Skill locations can change between tool versions; check your tool's documentation. The skill runs the
`ahp-bench` command (pip install) or `python -m ahp_bench` in a clone of the repository.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .config import resource

NAME = "llm-ahp-mcdm"
SKILL = resource("skills") / NAME
AGENTS_HINT = ("\n## LLM-AHP decision studies\n\nFor decisions from a data set (EDA, criteria, AHP hierarchy, performance "
               "matrix), AHP criteria weighting with LLM respondents and TOPSIS / ARAS / EDAS ranking, follow "
               "`.claude/skills/llm-ahp-mcdm/SKILL.md`.\n")


def install(target_dir: Path, force: bool = False, dry_run: bool = False, source: Path = SKILL) -> Path:
    source = Path(source)
    if not (source / "SKILL.md").is_file():
        raise FileNotFoundError(f"skill folder not found: {source}")
    dest = Path(target_dir).expanduser() / source.name
    files = sorted(p.relative_to(source) for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if dest.exists() and not force:
        raise FileExistsError(f"{dest} exists; use --force to overwrite")
    for f in files:
        print(f"{'would copy' if dry_run else 'copy'} {f.as_posix()} -> {dest / f}")
    if not dry_run:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dest


def add_agents_hint(folder: Path, dry_run: bool = False) -> Path:
    """Append a pointer to the project skill to folder/AGENTS.md unless it already mentions the skill."""
    path = Path(folder) / "AGENTS.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if "llm-ahp-mcdm/SKILL.md" in text:
        return path
    print(f"{'would add' if dry_run else 'add'} a pointer to the skill in {path}")
    if not dry_run:
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(("# Instructions for coding agents\n" if not text else "") + AGENTS_HINT)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ahp-bench install-skill", description="Install the llm-ahp-mcdm agent skill.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("target", nargs="?", help="skills directory of your agent")
    g.add_argument("--claude", action="store_true", help="~/.claude/skills (Claude Code, all projects)")
    g.add_argument("--codex", action="store_true", help="~/.codex/skills (Codex)")
    g.add_argument("--project", action="store_true",
                   help="./.claude/skills in the current folder, plus a pointer in ./AGENTS.md")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.claude:
        target = Path.home() / ".claude" / "skills"
    elif a.codex:
        target = Path.home() / ".codex" / "skills"
    elif a.project:
        target = Path.cwd() / ".claude" / "skills"
    else:
        target = Path(a.target)
    try:
        dest = install(target, a.force, a.dry_run)
    except (FileExistsError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 2
    if a.project:
        add_agents_hint(Path.cwd(), a.dry_run)
    print(f"{'dry run: nothing copied' if a.dry_run else 'installed'}: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
