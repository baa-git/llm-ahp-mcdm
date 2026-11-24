"""Show the AHP hierarchy of a problem: text tree, Mermaid diagram, indicator table and a grayscale figure.

ahp-bench hierarchy --problem P [--out DIR]

Prints the text tree; with --out also writes DIR/hierarchy.md (Mermaid diagram, text tree and indicator
table: criterion, indicator, unit, direction, source column, aggregation) and DIR/hierarchy.png (grayscale,
square boxes, orthogonal connectors; only when matplotlib is installed).
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from . import problem as pb


def indicator_rows(problem) -> list:
    """One row per leaf criterion: main criterion, leaf, and its indicator fields ("" when absent)."""
    by_code, mapping = {}, {}
    if problem.ranking:
        by_code = {i["code"]: i for i in problem.ranking["indicators"]}
        mapping = problem.ranking["criterion_indicator"]
    rows = []
    for leaf, _, mi, task, _ in problem.leaves:
        ind = by_code.get(mapping.get(leaf), {})
        rows.append({"main": problem.main.criteria[mi], "criterion": leaf, "level": "main" if task == pb.MAIN_TASK
                     else "sub", "code": ind.get("code", ""), "indicator": ind.get("name", ""),
                     "unit": ind.get("unit", ""), "direction": ind.get("direction", ""),
                     "source": ind.get("source", ""), "aggregation": ind.get("aggregation", ""),
                     "filter": ind.get("filter", "")})
    return rows


def _ind_text(r: dict) -> str:
    if not r["code"]:
        return ""
    bits = [b for b in (r["unit"], r["direction"]) if b]
    src = f"{r['aggregation']}({r['source']})" if r["aggregation"] else r["source"]
    if r["filter"]:
        src += f" where {r['filter']}"
    return f"  [{r['code']}: {r['indicator']}" + (f" ({', '.join(bits)})" if bits else "") + (f"; {src}" if src else "") + "]"


def text_tree(problem) -> str:
    """ASCII tree: goal, main criteria, sub-criteria, each leaf with its indicator."""
    rows = {r["criterion"]: r for r in indicator_rows(problem)}
    lines = [f"Goal: {problem.goal}"]
    mains = problem.main.criteria
    for i, c in enumerate(mains):
        last = i == len(mains) - 1
        g = problem.group_of(c)
        lines.append(f"{'`-- ' if last else '+-- '}{c}" + ("" if g else _ind_text(rows[c])))
        if g:
            pad = "    " if last else "|   "
            for k, s in enumerate(g.criteria):
                lines.append(f"{pad}{'`-- ' if k == len(g.criteria) - 1 else '+-- '}{s}{_ind_text(rows[s])}")
    return "\n".join(lines)


def _q(text: str) -> str:
    return str(text).replace('"', "#quot;")


def mermaid(problem) -> str:
    rows = {r["criterion"]: r for r in indicator_rows(problem)}
    out = ["```mermaid", "graph TD", f'  G["{_q(problem.goal)}"]']
    links = []

    def leaf(node, name):
        r = rows[name]
        if r["code"]:
            out.append(f'  I_{node}["{_q(r["code"])}: {_q(r["indicator"])}<br/>{_q(r["unit"])}, {r["direction"]}"]')
            links.append(f"  {node} -.- I_{node}")

    for i, c in enumerate(problem.main.criteria, 1):
        out.append(f'  M{i}["{_q(c)}"]')
        links.append(f"  G --> M{i}")
        g = problem.group_of(c)
        if g:
            for k, s in enumerate(g.criteria, 1):
                out.append(f'  M{i}S{k}["{_q(s)}"]')
                links.append(f"  M{i} --> M{i}S{k}")
                leaf(f"M{i}S{k}", s)
        else:
            leaf(f"M{i}", c)
    out += links + ["  classDef default fill:#ffffff,stroke:#222222,color:#000000;", "```"]
    return "\n".join(out)


def indicator_table_md(problem) -> str:
    esc = lambda s: str(s).replace("|", "\\|")
    lines = ["| Main criterion | Criterion | Indicator | Unit | Direction | Source column(s) | Aggregation | Filter |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in indicator_rows(problem):
        ind = f"{r['code']}: {r['indicator']}" if r["code"] else ""
        lines.append("| " + " | ".join(esc(x) for x in (r["main"] if r["level"] == "sub" else "", r["criterion"], ind,
                                                          r["unit"], r["direction"], r["source"], r["aggregation"],
                                                          r["filter"])) + " |")
    return "\n".join(lines)


def markdown(problem, png: bool = False) -> str:
    md = [f"# Decision hierarchy: {problem.title}", "",
          f"Goal: {problem.goal}. {len(problem.main.criteria)} main criteria, {len(problem.groups)} sub-criteria "
          f"groups, {len(problem.leaves)} leaf criteria" + (f", {len(problem.ranking['alternatives'])} alternatives"
                                                             if problem.has_ranking else "") + ".", "",
          mermaid(problem), "", "```", text_tree(problem), "```", ""]
    if png:
        md += ["![hierarchy](hierarchy.png)", ""]
    md += ["## Indicators", "", indicator_table_md(problem), ""]
    return "\n".join(md)


def write_png(problem, path: Path) -> bool:
    """Grayscale figure with square boxes and orthogonal connectors; False when matplotlib is missing."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
    except ImportError:
        return False
    rows = {r["criterion"]: r for r in indicator_rows(problem)}
    has_sub = bool(problem.groups)
    has_ind = any(r["code"] for r in rows.values())
    levels = 2 + has_sub + has_ind
    y = {"goal": levels - 1, "main": levels - 2, "sub": levels - 3 if has_sub else None, "ind": 0}
    leaves = problem.leaves
    n = len(leaves)
    bw, bh, gap = 1.0, 0.62, 1.25
    fig = Figure(figsize=(max(6.0, 1.35 * n + 0.6), 1.25 * levels + 0.4))
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.75, (n - 1) * gap + 0.75)
    ax.set_ylim(-0.55, levels - 1 + 0.55)
    ax.axis("off")

    def box(x, yy, text, fill, size=7.5, width=16, scale=1.0):
        w = bw * scale
        ax.add_patch(Rectangle((x - w / 2, yy - bh / 2), w, bh, facecolor=fill, edgecolor="black", linewidth=0.9))
        ax.text(x, yy, textwrap.fill(str(text), width), ha="center", va="center", fontsize=size, color="black")

    def connect(px, py, children):
        top, bottom = py - bh / 2, max(cy for _, cy in children) + bh / 2
        mid = (top + bottom) / 2
        ax.plot([px, px], [top, mid], color="black", linewidth=0.8)
        xs = [cx for cx, _ in children] + [px]
        ax.plot([min(xs), max(xs)], [mid, mid], color="black", linewidth=0.8)
        for cx, cy in children:
            ax.plot([cx, cx], [mid, cy + bh / 2], color="black", linewidth=0.8)

    xs = {leaf: k * gap for k, (leaf, *_rest) in enumerate(leaves)}
    main_pos = []
    for c in problem.main.criteria:
        g = problem.group_of(c)
        kids = g.criteria if g else [c]
        mx = sum(xs[k] for k in kids) / len(kids)
        main_pos.append((mx, y["main"]))
        box(mx, y["main"], c, "#f2f2f2")
        if g:
            subs = [(xs[s], y["sub"]) for s in g.criteria]
            connect(mx, y["main"], subs)
            for s in g.criteria:
                box(xs[s], y["sub"], s, "#ffffff")
    gx = sum(x for x, _ in main_pos) / len(main_pos)
    box(gx, y["goal"], problem.goal, "#d9d9d9", size=8.5, width=34, scale=min(2.4, max(1.0, n * gap / 2.5)))
    connect(gx, y["goal"], main_pos)
    if has_ind:
        for leaf, _, _, task, _ in leaves:
            r = rows[leaf]
            if not r["code"]:
                continue
            ly = y["main"] if task == pb.MAIN_TASK else y["sub"]
            ax.plot([xs[leaf], xs[leaf]], [ly - bh / 2, y["ind"] + bh / 2], color="black", linewidth=0.8,
                    linestyle=(0, (3, 2)))
            unit = f" ({r['unit']})" if r["unit"] else ""
            box(xs[leaf], y["ind"], f"{r['code']}{unit}, {r['direction']}", "#e6e6e6", size=6.5, width=18)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    return True


def write(problem, out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    png = write_png(problem, out / "hierarchy.png")
    with (out / "hierarchy.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write(markdown(problem, png))
    return {"md": out / "hierarchy.md", "png": out / "hierarchy.png" if png else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ahp-bench hierarchy", description="Show the AHP hierarchy of a problem.")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--out", default=None, help="folder for hierarchy.md and hierarchy.png (default: print only)")
    a = ap.parse_args(argv)
    try:
        p = pb.load_problem(a.problem)
    except (pb.ProblemError, OSError) as e:
        print(f"invalid problem file: {e}", file=sys.stderr)
        return 2
    print(text_tree(p))
    if a.out:
        res = write(p, Path(a.out))
        print(f"wrote {res['md']}" + (f" and {res['png']}" if res["png"] else " (no hierarchy.png: matplotlib is not "
                                                                                 "installed)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
