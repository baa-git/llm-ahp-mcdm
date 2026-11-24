"""Copy the llm-ahp-mcdm skill folder into an agent's skills directory (clone without pip install).

python scripts/install_skill.py --claude | --codex | --project | TARGET_DIR [--force] [--dry-run]

Same as `ahp-bench install-skill` (see ahp_bench/install_skill.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ahp_bench.install_skill import add_agents_hint, install, main  # noqa: E402,F401

if __name__ == "__main__":
    raise SystemExit(main())
