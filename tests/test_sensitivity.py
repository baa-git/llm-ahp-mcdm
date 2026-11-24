"""Offline tests for the main-criterion weight sensitivity (T14, T15), the consistency rule and the
revision / temperature report tables (T16-T18)."""
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ahp_bench import analysis as an  # noqa: E402
from ahp_bench import prompts, report, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_data import DATA, STUDY_DIR  # noqa: E402
PERF = an.load_performance(DATA / "performance_matrix.json")
EXPERT = an.load_expert_data(DATA / "expert_matrices.json")
X = an.delivery_rates(PERF["matrix"])
B = PERF["benefit_mask"]
ALTS = PERF["alternatives"]


class TestOneAtATime(unittest.TestCase):
    def test_weights(self):
        w = an.one_at_a_time_weights(8, 2, 0.4)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertAlmostEqual(w[2], 0.4)
        np.testing.assert_allclose(np.delete(w, 2), 0.6 / 7)
        np.testing.assert_allclose(an.one_at_a_time_weights(8, 0, 0.125), 1 / 8)

    def test_deterministic_and_consistent(self):
        a = an.one_at_a_time_sensitivity(X, B, ALTS)
        b = an.one_at_a_time_sensitivity(X, B, ALTS)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 8 * 3)
        for row in a:
            self.assertEqual(len(row["rankings"]), len(an.SENSITIVITY_STEPS))
            self.assertEqual(row["rankings"][0], row["base_ranking"])  # 0.125 = equal weights
            ch = row["top_change_at"]
            if ch is None:
                self.assertTrue(all(r.split(">")[0] == row["base_ranking"].split(">")[0] for r in row["rankings"]))
            else:
                k = list(an.SENSITIVITY_STEPS).index(ch)
                tops = [r.split(">")[0] for r in row["rankings"]]
                self.assertNotEqual(tops[k], tops[0])
                self.assertTrue(all(t == tops[0] for t in tops[:k]))

    def test_initials_and_ranking_string(self):
        self.assertEqual(an.supplier_initials(ALTS), ["A", "B", "D", "E", "G"])
        self.assertEqual(an.supplier_initials(["Alpha", "Alto", "Beta"]), ["Alp", "Alt", "Bet"])
        self.assertEqual(an.ranking_string([0.2, 0.9, 0.5], ["x", "y", "z"]), "y>z>x")

    def test_dominant_indicator_decides_top(self):
        # toy: two benefit criteria; supplier 0 best on c0, supplier 1 best on c1
        Xt = np.array([[10.0, 1.0], [1.0, 10.0], [5.0, 5.0]])
        bt = np.array([True, True])
        rows = an.one_at_a_time_sensitivity(Xt, bt, ["P", "Q", "R"], steps=(0.5, 0.9))
        for r in rows:
            if r["indicator"] == 0:
                self.assertTrue(r["rankings"][-1].startswith("P"))
            else:
                self.assertTrue(r["rankings"][-1].startswith("Q"))


class TestMainWeightSensitivity(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(len(an.MAIN_LEVELS), 14)
        self.assertEqual((an.MAIN_LEVELS[0], an.MAIN_LEVELS[-1]), (0.05, 0.70))
        np.testing.assert_allclose(np.diff(an.MAIN_LEVELS), 0.05)

    def test_rows_and_weights(self):
        rows = an.main_weight_sensitivity(X, B, ALTS)
        self.assertEqual(len(rows), 5 * 14 + 1)
        self.assertEqual((rows[0]["criterion"], rows[0]["level"]), (None, None))
        np.testing.assert_allclose(rows[0]["weights"], 1 / 8)
        for r in rows:
            w = r["weights"]
            self.assertAlmostEqual(float(w.sum()), 1.0)
            np.testing.assert_allclose(w[:4], w[0])  # Delivery main weight split equally over D1..D4
            self.assertEqual(set(r["methods"]), set(an.METHODS))
            for m in r["methods"].values():
                self.assertEqual(m["ranking"].split(">")[0], an.supplier_initials(ALTS)[ALTS.index(m["top"])])
            if r["criterion"] is None:
                continue
            k = an.MAIN_SHORT.index(r["criterion"])
            main = np.r_[w[:4].sum(), w[4:]]
            self.assertAlmostEqual(main[k], r["level"])
            others = np.delete(main, k)
            np.testing.assert_allclose(others, (1 - r["level"]) / 4)  # non-selected main weights equal
        d70 = an.main_level_weights(0, 0.70)
        np.testing.assert_allclose(d70[:4], 0.175)
        np.testing.assert_allclose(d70[4:], 0.075)

    def test_summary(self):
        rows = an.main_weight_sensitivity(X, B, ALTS)
        summ = an.sensitivity_summary(rows)
        self.assertEqual(len(summ), 5 * 3)
        for s in summ:
            rs = [r for r in rows if r["criterion"] == s["criterion"]]
            tops = [r["methods"][s["method"]]["top"] for r in rs]
            self.assertEqual((s["top_low"], s["top_high"]), (tops[0], tops[-1]))
            self.assertEqual(s["equal_top"], rows[0]["methods"][s["method"]]["top"])
            self.assertEqual(s["levels_same_top"], sum(t == s["equal_top"] for t in tops))
            self.assertTrue(1 <= s["distinct_rankings"] <= 14)
            ch = s["first_change_at"]
            if ch is None:
                self.assertEqual(s["levels_same_top"], 14)
            else:
                k = an.MAIN_LEVELS.index(ch)
                self.assertNotEqual(tops[k], s["equal_top"])
                self.assertTrue(all(t == s["equal_top"] for t in tops[:k]))

    def test_toy_dominance(self):
        # supplier P is best on Financial only; at a high Financial weight P must be on top
        Xt = np.array([[1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0],
                       [2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0]])
        rows = an.main_weight_sensitivity(Xt, np.ones(8, bool), ["P", "Q"], levels=(0.70,))
        fin = next(r for r in rows if r["criterion"] == "Financial")
        self.assertTrue(all(m["top"] == "P" for m in fin["methods"].values()))


class TestConsistencyRule(unittest.TestCase):
    def test_rounding(self):
        self.assertTrue(an.is_consistent(0.10004))
        self.assertTrue(an.is_consistent(0.1))
        self.assertFalse(an.is_consistent(0.1001))
        self.assertFalse(an.is_consistent(0.10005000001))


def lean(values):
    return json.dumps({"answers": [{"question": q, "value": v} for q, v in enumerate(values, 1)]})


def src(model, cond, task, rep, values, tag="main", temp=0.6):
    reply = lean(values) if cond == "lean" else json.dumps(
        {"reasoning": "r", "ranking": [], "answers": json.loads(lean(values))["answers"]})
    r = {"run_tag": tag, "model_key": model, "model_id": f"x/{model}", "developer": "Dev", "condition": cond,
         "task": task, "rep": rep, "messages": prompts.build_messages(cond, task), "raw_reply": reply,
         "parse_ok": True, "cost": 0.0, "usage": {"reasoning_tokens": 0}, "provider": "P",
         "requested_params": {"temperature": temp}, "applied_params": {"reasoning": None}}
    r.update(run.analyse(cond, task, reply))
    return r


BAD_MAIN = ["9", "1/9", "9", "1/9", "9", "1/9", "9", "1/9", "9", "1/9"]
OK_MAIN = ["3", "3", "5", "5", "1", "3", "3", "3", "3", "1"]
BAD_DEL = ["9", "1/9", "9", "1/9", "9", "1/9"]
OK_DEL = ["3", "5", "3", "3", "1", "1/3"]


class TestReportRevisionTables(unittest.TestCase):
    def test_tables_t14_to_t18(self):
        records, revs = [], []
        for model in ("gpt-4o", "glm-5"):
            for cond in ("lean", "cot", "guided"):
                if cond == "guided":
                    continue
                for rep in (1, 2):
                    records.append(src(model, cond, "main", rep, BAD_MAIN if rep == 1 else OK_MAIN))
                    records.append(src(model, cond, "delivery", rep, OK_DEL if rep == 1 else BAD_DEL))
        for t in ("temp_0.0", "temp_1.0"):
            for model in ("gpt-4o", "glm-5"):
                for rep in (1, 2):
                    records.append(src(model, "lean", "main", rep, OK_MAIN, tag=t, temp=float(t[5:])))
                    records.append(src(model, "lean", "delivery", rep, OK_DEL, tag=t, temp=float(t[5:])))
        # revisions: gpt-4o lean main rep1 reached at revision 2; glm-5 lean main rep1 never reached (max 2)
        for model, seq in (("gpt-4o", [BAD_MAIN, OK_MAIN]), ("glm-5", [BAD_MAIN, BAD_MAIN])):
            for k, vals in enumerate(seq, 1):
                r = src(model, "lean", "main", 1, vals, tag="revision")
                r.update(source_tag="main", revision_number=k)
                revs.append(r)
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "calls.jsonl"
            inp.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            (Path(d) / "revisions.jsonl").write_text("\n".join(json.dumps(r) for r in revs) + "\n", encoding="utf-8")
            out = Path(d) / "out"
            report.run_report(inp, out, tag="main", data_dir=DATA, figures=False)
            names = {p.stem for p in (out / "tables").glob("*.csv")}
            for tid in ("T14_weight_sensitivity", "T15_sensitivity_summary", "T16_revision_summary",
                        "T17_weights_before_after_revision", "T18_temperature_comparison"):
                self.assertIn(tid, names)
            self.assertFalse([n for n in names if n.startswith("T19")])
            md = (out / "report.md").read_text(encoding="utf-8")
            for gone in ("T19", "consistent_expert", "Consistent-expert", "Monte Carlo", "monte_carlo",
                         "one_at_a_time"):
                self.assertNotIn(gone, md)

            def rows(stem):
                return list(csv.DictReader(io.StringIO((out / "tables" / f"{stem}.csv").read_text(encoding="utf-8"))))

            t16 = {(r["Model"], r["Condition"], r["Task"]): r for r in rows("T16_revision_summary")}
            g = t16[("gpt-4o", "direct", "main")]
            self.assertEqual((g["Matrices"], g["Needing revision"], g["Reached after ≤1"], g["Reached after ≤2"],
                              g["Reached after ≤5"], g["Not reached"]), ("2", "1", "0", "1", "1", "0"))
            z = t16[("glm-5", "direct", "main")]
            self.assertEqual((z["Reached after ≤5"], z["Not reached"], z["No revision record"]), ("0", "1", "0"))
            p = t16[("Panel", "direct", "delivery")]
            self.assertEqual((p["Needing revision"], p["No revision record"]), ("2", "2"))
            self.assertNotIn(("Panel", "CoT", "main"), t16)  # no cot revision records

            t17 = rows("T17_weights_before_after_revision")
            self.assertEqual([(r["Condition"], r["Answers"]) for r in t17],
                             [("Experts", ""), ("direct", "first"), ("direct", "final")])
            self.assertNotEqual(t17[1]["Delivery"], t17[2]["Delivery"])
            self.assertEqual(t17[1]["D1 Delivered"], t17[2]["D1 Delivered"])  # delivery untouched

            t18 = rows("T18_temperature_comparison")
            self.assertEqual([(r["Run tag"], r["Temperature"]) for r in t18],
                             [("temp_0.0", "0.0"), ("main", "0.6"), ("temp_1.0", "1.0")])
            self.assertEqual(t18[0]["CR ≤ 0.10 main"], "4/4")

            t14 = rows("T14_weight_sensitivity")
            self.assertEqual(len(t14), 5 * 14 + 1)
            self.assertEqual((t14[0]["Varied criterion"], t14[0]["Level"]), ("Equal", ""))
            self.assertEqual([r["Level"] for r in t14[1:15]], [f"{0.05 * k:.2f}" for k in range(1, 15)])
            for r in t14:
                self.assertAlmostEqual(sum(float(r[c]) for c in an.GLOBAL_CODES), 1.0, places=3)
            d70 = next(r for r in t14 if r["Varied criterion"] == "Delivery" and r["Level"] == "0.70")
            self.assertEqual([d70[c] for c in ("D1", "D2", "D3", "D4", "F")], ["0.1750"] * 4 + ["0.0750"])
            self.assertEqual(list(t14[0])[-6:], ["TOPSIS ranking", "TOPSIS top supplier", "ARAS ranking",
                                                 "ARAS top supplier", "EDAS ranking", "EDAS top supplier"])
            t15 = rows("T15_sensitivity_summary")
            self.assertEqual(len(t15), 15)
            self.assertEqual([r["Method"] for r in t15[:3]], ["TOPSIS", "ARAS", "EDAS"])
            first = {p.name: p.read_bytes() for p in (out / "tables").iterdir()}
            report.run_report(inp, out, tag="main", data_dir=DATA, figures=False)
            self.assertEqual(first, {p.name: p.read_bytes() for p in (out / "tables").iterdir()})
            # without revision records / temp tags the optional tables are not written
            out2 = Path(d) / "out2"
            inp2 = Path(d) / "only_main" / "calls.jsonl"
            inp2.parent.mkdir()
            inp2.write_text("\n".join(json.dumps(r) for r in records if r["run_tag"] == "main") + "\n",
                            encoding="utf-8")
            report.run_report(inp2, out2, tag="main", data_dir=DATA, figures=False)
            names2 = {p.stem[:3] for p in (out2 / "tables").glob("*.csv")}
            self.assertFalse({"T16", "T17", "T18"} & names2)
            self.assertTrue({"T14", "T15"} <= names2)
            self.assertNotIn("T19", names2)


if __name__ == "__main__":
    unittest.main()
