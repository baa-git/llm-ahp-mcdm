"""Offline tests for ahp_bench.analysis and ahp_bench.report (synthetic records only)."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ahp_bench import analysis as an  # noqa: E402
from ahp_bench import prompts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_data import DATA, STUDY_DIR  # noqa: E402


def consistent(w, noise=None, seed=0):
    """Reciprocal matrix a_ij = w_i/w_j, optionally perturbed (still reciprocal)."""
    w = np.asarray(w, float)
    a = w[:, None] / w[None, :]
    if noise:
        rng = np.random.default_rng(seed)
        n = len(w)
        for i in range(n):
            for j in range(i + 1, n):
                a[i, j] *= float(np.exp(rng.normal(0, noise)))
                a[j, i] = 1 / a[i, j]
    return a


def rec(model, cond, task, rep, matrix=None, ok=True, tag="main", cost=0.01, **extra):
    r = {"run_tag": tag, "model_key": model, "model_id": f"x/{model}", "developer": "Dev",
         "condition": cond, "task": task, "rep": rep, "parse_ok": ok,
         "matrix": None if matrix is None else np.asarray(matrix).tolist(),
         "usage": {"reasoning_tokens": 0}, "cost": cost, "provider": "P",
         "requested_params": {"temperature": 0.6}, "applied_params": {"reasoning": {"effort": "none"}}}
    r.update(extra)
    return r


class TestAggregation(unittest.TestCase):
    def test_geometric_mean_then_eigenvector(self):
        a = np.array([[1, 2], [0.5, 1]])
        b = np.array([[1, 8], [1 / 8, 1]])
        w, cr, g = an.group_priority([a, b])
        np.testing.assert_allclose(g, [[1, 4], [0.25, 1]])
        np.testing.assert_allclose(w, [0.8, 0.2])

    def test_consistent_matrix_recovers_weights(self):
        w0 = np.array([0.4, 0.3, 0.2, 0.1])
        w, cr = an.priority(consistent(w0))
        np.testing.assert_allclose(w, w0, atol=1e-12)
        self.assertAlmostEqual(cr, 0.0, places=10)

    def test_global_weights_sum_to_one(self):
        main = an.priority(consistent([0.3, 0.25, 0.2, 0.15, 0.1], 0.3, 1))[0]
        deliv = an.priority(consistent([0.4, 0.3, 0.2, 0.1], 0.3, 2))[0]
        g = an.global_weights(main, deliv)
        self.assertEqual(g.shape, (8,))
        self.assertAlmostEqual(g.sum(), 1.0, places=12)
        np.testing.assert_allclose(g[:4], main[0] * deliv)
        np.testing.assert_allclose(g[4:], main[1:])

    def test_expert_weights_from_data_file(self):
        ex = an.expert_weights(an.load_expert_data(DATA / "expert_matrices.json"))
        self.assertEqual(len(ex["cr_main"]), 7)
        self.assertAlmostEqual(ex["global"].sum(), 1.0, places=12)


class TestPerformance(unittest.TestCase):
    def test_rates_conversion(self):
        X = np.array([[10, 5, 3, 2, 8.0, 1.0, 100.0, 5.0], [1, 1, 1, 1, 7.0, 2.0, 90.0, 4.0]])
        R = an.delivery_rates(X)
        np.testing.assert_allclose(R[0, :4], [50, 25, 15, 10])
        np.testing.assert_allclose(R[1, :4], [25, 25, 25, 25])
        np.testing.assert_allclose(R[:, 4:], X[:, 4:])
        np.testing.assert_allclose(X[0, :4], [10, 5, 3, 2])  # input untouched

    def test_rates_on_project_data(self):
        p = an.load_performance(DATA / "performance_matrix.json")
        R = an.delivery_rates(p["matrix"])
        np.testing.assert_allclose(R[:, :4].sum(axis=1), 100.0)
        self.assertEqual(p["benefit_mask"].tolist(), [True, False, False, False, True, False, True, False])


class TestRanking(unittest.TestCase):
    def test_topsis_dominance(self):
        X = np.array([[2.0, 1.0], [1.0, 2.0]])
        s = an.topsis(X, [0.5, 0.5], [True, False])
        np.testing.assert_allclose(s, [1.0, 0.0])

    def test_topsis_hand_value(self):
        # single benefit criterion 3,4 -> normalised 0.6,0.8; best 0.8 worst 0.6
        np.testing.assert_allclose(an.topsis(np.array([[3.0], [4.0]]), [1.0], [True]), [0.0, 1.0])

    def test_aras_ideal_in_denominator(self):
        X = np.array([[2.0], [1.0]])
        np.testing.assert_allclose(an.aras(X, [1.0], [True]), [1.0, 0.5])   # S=[.4,.2], S0=.4
        np.testing.assert_allclose(an.aras(X, [1.0], [False]), [0.5, 1.0])  # b=[.5,1], opt=1
        # two criteria: benefit [4,2], cost [1,2] -> b2=[1,.5]; denominators 10 and 2.5
        X = np.array([[4.0, 1.0], [2.0, 2.0]])
        k = an.aras(X, [0.5, 0.5], [True, False])
        s = np.array([0.5 * 0.4 + 0.5 * 0.4, 0.5 * 0.2 + 0.5 * 0.2])
        np.testing.assert_allclose(k, s / 0.4)

    def test_edas_hand_value(self):
        X = np.array([[3.0], [1.0]])
        np.testing.assert_allclose(an.edas(X, [1.0], [True]), [1.0, 0.0])
        np.testing.assert_allclose(an.edas(X, [1.0], [False]), [0.0, 1.0])
        X = np.array([[3.0], [2.0], [1.0]])  # avg 2: PDA .5,0,0  NDA 0,0,.5
        np.testing.assert_allclose(an.edas(X, [1.0], [True]), [1.0, 0.5, 0.0])

    def test_ranks(self):
        np.testing.assert_allclose(an.rank_desc([0.2, 0.9, 0.5]), [3, 1, 2])
        np.testing.assert_allclose(an.average_ranks([1, 2, 2, 3]), [1, 2.5, 2.5, 4])


class TestSimilarity(unittest.TestCase):
    def test_identical_and_reversed(self):
        w = [0.4, 0.3, 0.2, 0.1]
        s = an.similarity(w, w)
        self.assertEqual((s["spearman"], s["kendall"], s["rmse_pp"], s["mad_pp"]), (1.0, 1.0, 0.0, 0.0))
        s = an.similarity(w[::-1], w)
        self.assertAlmostEqual(s["spearman"], -1.0)
        self.assertAlmostEqual(s["kendall"], -1.0)

    def test_pp_metrics(self):
        s = an.similarity([0.5, 0.3, 0.2], [0.4, 0.4, 0.2])
        self.assertAlmostEqual(s["mad_pp"], 20 / 3)
        self.assertAlmostEqual(s["rmse_pp"], np.sqrt(200 / 3))
        # one discordant pair out of three, no ties -> tau = 1/3
        self.assertAlmostEqual(an.kendall_tau([1, 3, 2], [1, 2, 3]), 1 / 3)
        self.assertAlmostEqual(an.spearman([1, 3, 2], [1, 2, 3]), 0.5)


class TestSelection(unittest.TestCase):
    def test_duplicates_and_failures(self):
        m1, m2 = consistent([0.5, 0.2, 0.1, 0.1, 0.1]), consistent([0.2, 0.2, 0.2, 0.2, 0.2])
        records = [rec("a", "lean", "main", 1, None, ok=False),
                   rec("a", "lean", "main", 1, m1),
                   rec("a", "lean", "main", 1, m2),
                   rec("a", "lean", "main", 2, np.ones((4, 4))),  # wrong size -> unusable
                   rec("a", "lean", "main", 1, m2, tag="other")]
        sel, fails, tagged = an.select_records(records, "main")
        self.assertEqual(len(tagged), 4)
        self.assertEqual(list(sel), [("a", "lean", "main", 1)])
        np.testing.assert_allclose(sel[("a", "lean", "main", 1)]["matrix"], m1)
        self.assertEqual(fails[("a", "lean", "main")], 2)

    def test_partial_last_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calls.jsonl"
            p.write_text(json.dumps(rec("a", "lean", "main", 1, np.ones((5, 5))))
                         + "\n" + '{"run_tag": "main", "mod', encoding="utf-8")
            records, bad = an.load_records(p)
            self.assertEqual((len(records), bad), (1, 1))


class TestReportEndToEnd(unittest.TestCase):
    def test_report_runs(self):
        from ahp_bench import report
        records = []
        main_w = [0.3, 0.25, 0.2, 0.15, 0.1]
        del_w = [0.4, 0.3, 0.2, 0.1]
        seed = 0
        for model in ["claude-sonnet-4.6", "gpt-4o", "gpt-5.2"]:
            for cond in ["lean", "cot"]:  # guided missing on purpose
                for rep in (1, 2):
                    for task, w in (("main", main_w), ("delivery", del_w)):
                        seed += 1
                        extra = {"reasoning": "because", "ranking": prompts.CRITERIA[task]} if cond == "cot" else {}
                        if model == "gpt-4o" and rep == 1 and task == "main":
                            records.append(rec(model, cond, task, rep, None, ok=False))
                        records.append(rec(model, cond, task, rep, consistent(w, 0.3, seed), **extra))
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "calls.jsonl"
            inp.write_text("\n".join(json.dumps(r) for r in records) + '\n{"partial', encoding="utf-8")
            out = Path(d) / "out"
            info = report.run_report(inp, out, tag="main", data_dir=DATA, latex=True)
            self.assertEqual(info["skipped_lines"], 1)
            self.assertEqual(info["n_failed"], 2)
            ids = [f"T{k:02d}" for k in range(1, 14)]
            for tid in ids:
                self.assertEqual(len(list((out / "tables").glob(f"{tid}_*.csv"))), 1, tid)
                self.assertEqual(len(list((out / "tables").glob(f"{tid}_*.tex"))), 1, tid)
            for f in ["F1_global_weight_diff_heatmap", "F2_cr_main_dotplot"]:
                self.assertTrue((out / "figures" / f"{f}.pdf").exists())
                self.assertTrue((out / "figures" / f"{f}.png").exists())
            for f in ["matrices_llm.csv", "matrices_expert.csv", "cot_reasoning.csv"]:
                self.assertTrue((out / "appendix" / f).exists())
            md = (out / "report.md").read_text(encoding="utf-8")
            positions = [md.index(f"## {tid} ") for tid in ids]
            self.assertEqual(positions, sorted(positions))
            tex = (out / "tables" / "T06_global_weights.tex").read_text(encoding="utf-8")
            self.assertIn("\\toprule", tex)
            self.assertNotIn("documentclass", tex)
            # guided columns empty, lean filled
            import csv
            rows = list(csv.DictReader(io.StringIO((out / "tables" / "T06_global_weights.csv").read_text(encoding="utf-8"))))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(r["LLM guided"] == "" for r in rows))
            self.assertAlmostEqual(sum(float(r["LLM unguided_direct"]) for r in rows), 1.0, places=3)
            n_llm = len((out / "appendix" / "matrices_llm.csv").read_text(encoding="utf-8").splitlines()) - 1
            self.assertEqual(n_llm, 3 * 2 * 2 * (25 + 16))
            # deterministic output
            first = {p.name: p.read_bytes() for p in (out / "tables").iterdir()}
            report.run_report(inp, out, tag="main", data_dir=DATA, figures=False, latex=True)
            self.assertEqual(first, {p.name: p.read_bytes() for p in (out / "tables").iterdir()})
            # without latex=True no TeX file is written, and the stored key "lean" is shown as "unguided"
            out2 = Path(d) / "out2"
            report.run_report(inp, out2, tag="main", data_dir=DATA, figures=False)
            self.assertEqual(list(out2.rglob("*.tex")), [])
            self.assertEqual({p.name for p in (out2 / "tables").iterdir()}, {n for n in first if n.endswith(".csv")})
            md2 = (out2 / "report.md").read_text(encoding="utf-8")
            self.assertIn("Unguided, zero-shot (direct)", md2)
            self.assertIn("| Chain-of-thought | not run |", md2)
            t03 = list(csv.DictReader(io.StringIO((out2 / "tables" / "T03_llm_consistency.csv").read_text(encoding="utf-8"))))
            self.assertEqual({(r["Guidance"], r["Technique"], r["Condition"]) for r in t03},
                             {("unguided", "zero-shot", "direct"), ("unguided", "CoT", "CoT"), ("guided", "zero-shot", "guided")})
            a01 = (out2 / "appendix" / "matrices_llm.csv").read_text(encoding="utf-8").splitlines()
            self.assertTrue(a01[0].startswith("model,guidance,technique,condition,"))
            self.assertFalse(any(",lean," in line for line in a01))

    def test_t22_guided_compliance(self):
        import csv
        from ahp_bench import report
        prompted = prompts.guided_prompted_order("main")
        self.assertEqual([prompts.MAIN_CRITERIA[i] for i in prompted["order"]],
                         ["Quality Performance", "Compliance and Reliability", "Delivery and Operational Performance",
                          "Financial Performance", "Volume and Logistics"])
        self.assertEqual(prompts.DELIVERY_SUBCRITERIA[prompts.guided_prompted_order("delivery")["top"]], "Delivered Orders")
        follow = np.zeros(5)
        follow[prompted["order"]] = [0.4, 0.25, 0.2, 0.1, 0.05]      # weights in exactly the prompted order
        swap = follow.copy()
        swap[prompted["order"][:2]] = swap[prompted["order"][1::-1]]   # first two swapped
        reverse = np.zeros(5)
        reverse[prompted["order"]] = [0.05, 0.1, 0.2, 0.25, 0.4]   # the prompted order reversed
        deliv_first, deliv_last = [0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]
        records = []
        for model, wm, wd in (("gpt-4o", follow, deliv_first), ("glm-5", swap, deliv_last)):
            for rep in (1, 2):
                records.append(rec(model, "guided", "main", rep, consistent(wm)))
                records.append(rec(model, "guided", "delivery", rep, consistent(wd)))
                records.append(rec(model, "lean", "main", rep, consistent(reverse)))
                records.append(rec(model, "lean", "delivery", rep, consistent(deliv_last)))
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "calls.jsonl"
            inp.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            out = Path(d) / "out"
            report.run_report(inp, out, tag="main", data_dir=DATA, figures=False)
            rows = list(csv.DictReader(io.StringIO((out / "tables" / "T22_guided_compliance.csv").read_text(encoding="utf-8"))))
        t = {(r["Respondents"], r["Condition"], r["Task"]): r for r in rows}
        cols = ("Replies", "Ranking equals prompted order", "Prompted first criterion ranked first")
        self.assertEqual(tuple(t[("gpt-4o", "guided", "main")][c] for c in cols), ("2", "2", "2"))
        self.assertEqual(t[("gpt-4o", "guided", "main")]["Mean Spearman rho with prompted order"], "1.000")
        self.assertEqual(tuple(t[("glm-5", "guided", "main")][c] for c in cols), ("2", "0", "0"))
        self.assertEqual(t[("glm-5", "guided", "main")]["Mean Spearman rho with prompted order"], "0.900")
        self.assertEqual(tuple(t[("Panel", "guided", "main")][c] for c in cols), ("4", "2", "2"))
        p = t[("Panel", "direct", "main")]
        self.assertEqual((p["Comparison"], p["Guidance"], p["Technique"]), ("reference (order not shown)", "unguided", "zero-shot"))
        self.assertEqual(p["Mean Spearman rho with prompted order"], "-1.000")
        self.assertEqual(t[("Panel", "CoT", "main")]["Replies"], "")             # no CoT records
        dl = t[("Panel", "guided", "delivery")]
        self.assertEqual((dl["Replies"], dl["Ranking equals prompted order"], dl["Prompted first criterion ranked first"],
                          dl["Mean Spearman rho with prompted order"]), ("4", "", "2", ""))   # only the first is stated
        self.assertEqual(t[("Experts (individual matrices)", "", "main")]["Replies"], "7")

    def test_t21_ranking_agreement(self):
        import csv
        from ahp_bench import report
        records = []
        seed = 0
        for model in ["claude-sonnet-4.6", "gpt-4o"]:
            for cond in ["lean", "cot"]:
                for rep in (1, 2):
                    for task, w in (("main", [0.3, 0.25, 0.2, 0.15, 0.1]), ("delivery", [0.4, 0.3, 0.2, 0.1])):
                        seed += 1
                        records.append(rec(model, cond, task, rep, consistent(w, 0.3, seed)))
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "calls.jsonl"
            inp.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            out = Path(d) / "out"
            report.run_report(inp, out, tag="main", data_dir=DATA, figures=False)
            rows = list(csv.DictReader(io.StringIO((out / "tables" / "T21_ranking_agreement.csv").read_text(encoding="utf-8"))))
            self.assertEqual([(r["Method"], r["Weights"]) for r in rows],
                             [(m, w) for m in an.METHODS for w in ("LLM panel Unguided, zero-shot (direct)",
                                                                    "LLM panel Unguided, CoT",
                                                                    "LLM panel Guided (zero-shot)", "Equal (1/8)")])
            t11 = list(csv.DictReader(io.StringIO((out / "tables" / "T11_supplier_rankings.csv").read_text(encoding="utf-8"))))
            t13 = list(csv.DictReader(io.StringIO((out / "tables" / "T13_equal_weight_baseline.csv").read_text(encoding="utf-8"))))
            for r in rows:
                if r["Weights"] == "LLM panel Guided (zero-shot)":   # no guided records
                    self.assertEqual((r["Spearman rho"], r["Kendall tau"], r["Suppliers with different rank"]), ("", "", ""))
                    continue
                m = r["Method"]
                ref = [float(x[f"{m} rank"]) for x in t11 if x["Weights"] == "Experts"]
                src = t13 if r["Weights"].startswith("Equal") else [x for x in t11 if x["Weights"] == r["Weights"]]
                ranks = [float(x[f"{m} rank"]) for x in src]
                self.assertEqual(int(r["Suppliers with different rank"]), sum(a != b for a, b in zip(ranks, ref)))
                self.assertEqual(r["Spearman rho"], f"{an.spearman(ranks, ref):.3f}")
                self.assertEqual(r["Kendall tau"], f"{an.kendall_tau(ranks, ref):.3f}")
            self.assertIn("## T21 Ranking agreement with the expert weights", (out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
