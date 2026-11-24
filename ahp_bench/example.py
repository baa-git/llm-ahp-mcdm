"""Copy the bundled SYNTHETIC example (supplier orders, a criteria spec, the generator script) to a folder.

ahp-bench example --out DIR [--force]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .config import resource

EXAMPLES = resource("examples")
FILES = ("orders_synthetic.csv", "orders_spec.json", "make_orders_synthetic.py", "README.md")

WALKTHROUGH = """Synthetic example copied to {out} (invented data, see README.md). Walk-through, no key, no cost:
  cd {out}
  ahp-bench eda --data orders_synthetic.csv --out eda                                  # stage 0: profile
  ahp-bench criteria validate --spec orders_spec.json --data orders_synthetic.csv      # stage 1-2: hierarchy + matrix
  ahp-bench criteria build --spec orders_spec.json --data orders_synthetic.csv --out problems/orders_synthetic.json
  ahp-bench hierarchy --problem problems/orders_synthetic.json --out hierarchy         # show the hierarchy
  ahp-bench host prompts --problem problems/orders_synthetic.json --conditions direct --panelists 3 --out work
  (one fresh subagent per file in work/prompts/, reply saved as work/replies/<same name>.json)
  ahp-bench host ingest --problem problems/orders_synthetic.json --dir work --results results/orders --tag main --model-name <host model>
  ahp-bench report --problem problems/orders_synthetic.json --results results/orders  # stages 4-6
  ahp-bench audit --problem problems/orders_synthetic.json --results results/orders"""


def copy_example(out: Path, force: bool = False) -> list:
    out = Path(out)
    clash = [f for f in FILES if (out / f).exists()]
    if clash and not force:
        raise FileExistsError(f"{', '.join(clash)} already in {out}; use --force to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for f in FILES:
        shutil.copyfile(EXAMPLES / f, out / f)
        done.append(out / f)
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ahp-bench example", description="Copy the synthetic example to a folder.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    try:
        copy_example(Path(a.out), a.force)
    except (FileExistsError, OSError) as e:
        print(e, file=sys.stderr)
        return 2
    print(WALKTHROUGH.format(out=a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
