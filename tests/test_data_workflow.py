"""Stages 0-2 and the walk-through from a CSV to a ranking: reading awkward CSVs, EDA, criteria specs and
every aggregation, problem files with provenance, criteria proposals (host and API mode, offline), the
hierarchy outputs and the bundled synthetic example (offline, temporary folders)."""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ahp_bench import __main__ as entry  # noqa: E402
from ahp_bench import audit, criteria, eda, example, hierarchy, host, report, run, store, tabular  # noqa: E402
from ahp_bench import problem as pb  # noqa: E402

EXAMPLES = ROOT / "examples"
QUESTION = re.compile(r"^\d+\.\s+Left: .+ \| Right: .+$", re.M)


def cli(fn, argv, **kw):
    """Run a command main(); return (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = fn(argv, **kw)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 2
            err.write(str(e.code))
    return rc, out.getvalue(), err.getvalue()


def canned_answers(prompt_text: str, values=("2", "3", "1/2", "1", "3", "2")) -> str:
    n = len(QUESTION.findall(prompt_text))
    return json.dumps({"answers": [{"question": q, "value": values[(q - 1) % len(values)]} for q in range(1, n + 1)]})


SMALL_CSV = """supplier,price,status,defects,qty,ordered,delivered,check
S1,10,ok,1,100,2026-01-01,2026-01-05,Pass
S1,14,late,3,50,2026-01-02,2026-01-10,Fail
S1,,ok,0,50,2026-01-03,2026-01-04,Pass
S2,20,ok,2,200,2026-01-01,2026-01-03,Pass
S2,22,ok,4,100,2026-01-02,2026-01-04,Pass
S3,5,late,5,50,2026-01-01,2026-01-11,Fail
S3,7,ok,1,50,2026-01-02,2026-01-06,Pass
S3,9,late,2,100,2026-01-03,2026-01-05,Pass
"""


def small_spec(**over):
    spec = {
        "spec_version": 1, "proposed_by": "human",
        "problem": {"name": "small", "title": "Small", "goal": "choosing a supplier", "role": "You are a buyer."},
        "alternative_column": "supplier",
        "indicators": [
            {"id": "P", "label": "Mean price", "aggregation": "mean", "column": "price", "direction": "cost", "unit": "USD"},
            {"id": "OK", "label": "On-time share", "aggregation": "share", "column": "status", "value": "ok",
             "direction": "benefit", "unit": "share"},
            {"id": "D", "label": "Defect rate", "aggregation": "ratio", "numerator": "defects", "denominator": "qty",
             "direction": "cost", "scale": 100, "unit": "%"},
            {"id": "LT", "label": "Lead time", "aggregation": "mean_days_between", "start": "ordered", "end": "delivered",
             "direction": "cost", "unit": "days"},
        ],
        "hierarchy": {"criteria": [{"name": "Cost", "indicator": "P"}, {"name": "Quality", "indicator": "D"},
                                   {"name": "Delivery", "sub_criteria": [{"name": "Punctuality", "indicator": "OK"},
                                                                         {"name": "Speed", "indicator": "LT"},
                                                                         {"name": "Orders", "indicator": "N"}]}]},
    }
    spec["indicators"].append({"id": "N", "label": "Orders", "aggregation": "count", "direction": "benefit"})
    spec.update(over)
    return spec


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text, encoding="utf-8"):
        p = self.d / name
        p.write_bytes(text.encode(encoding) if isinstance(text, str) else text)
        return p


class TabularTests(TmpCase):
    def test_semicolon_decimal_comma_cp1254_and_bom(self):
        text = "Tedarikçi;Fiyat;Not\nAğaç A.Ş.;1.234,5;iyi\nÇam Ltd;12,75;\nAğaç A.Ş.;NA;kötü\n"
        for enc, prefix in (("cp1254", b""), ("utf-8", b"\xef\xbb\xbf"), ("utf-8", b"")):
            p = self.write("t.csv", prefix + text.encode(enc))
            t = tabular.read_table(p)
            self.assertEqual(t.delimiter, ";")
            self.assertEqual(t.header, ["Tedarikçi", "Fiyat", "Not"])
            self.assertEqual(t.encoding, {"cp1254": "cp1254", "utf-8": "utf-8-sig" if prefix else "utf-8"}[enc])
            self.assertEqual(t.decimal("Fiyat"), ",")
            self.assertEqual(t.numbers("Fiyat"), [1234.5, 12.75, None])
            self.assertEqual(t.column("Not"), ["iyi", "", "kötü"])

    def test_tab_pipe_comments_malformed_and_numbers(self):
        p = self.write("t.tsv", "# comment\n\na\tb\tc\n1\t2\t3\n4\t5\n6\t7\t8\t9\n")
        t = tabular.read_table(p)
        self.assertEqual((t.delimiter, t.comment_lines, t.malformed_rows), ("\t", 2, 2))
        self.assertEqual(t.rows[1], ["4", "5", ""])
        p = self.write("p.txt", "x|y\n1,000.5|a\n2,500|b\n")
        t = tabular.read_table(p)
        self.assertEqual(t.delimiter, "|")
        self.assertEqual(t.numbers("x"), [1000.5, 2500.0])
        self.assertIsNone(tabular.number("abc"))
        self.assertIsNone(tabular.number(" - "))
        self.assertEqual(tabular.number("12 %"), 12.0)
        self.assertEqual(tabular.number("1.234.567,8", ","), 1234567.8)
        self.assertEqual(tabular.date_ordinal("05.01.2026") - tabular.date_ordinal("2026-01-01"), 4)
        self.assertIsNone(tabular.date_ordinal("2026-02-30"))
        p = self.write("dup.csv", "a,a,\n1,2,3\n")
        self.assertEqual(tabular.read_table(p).header, ["a", "a_2", "column_3"])
        with self.assertRaises(tabular.TableError):
            tabular.read_table(self.write("empty.csv", "# only a comment\n"))


class EdaTests(TmpCase):
    def test_profile_awkward_csv(self):
        text = ("# exported\nid;vendor;price;late_days;status;region;note\n"
                + "".join(f"{k};V{k % 4};{10 + k},5;{k % 3};{'Late' if k % 5 == 0 else 'On time'};R{k % 2};x{k}\n"
                          for k in range(1, 41))
                + "41;V1;;;;R1;\n")
        p = self.write("awk.csv", text, "cp1254")
        prof = eda.profile(p)
        cols = {c["name"]: c for c in prof["columns"]}
        self.assertEqual(prof["rows"], 41)
        self.assertEqual(prof["delimiter"], ";")
        self.assertEqual(cols["id"]["type"], "identifier")
        self.assertEqual(cols["price"]["type"], "numeric")
        self.assertEqual(cols["price"]["missing"], 1)
        self.assertEqual(cols["price"]["numeric"]["min"], 11.5)
        self.assertEqual(cols["price"]["numeric"]["decimal_mark"], ",")
        self.assertEqual(cols["status"]["type"], "categorical")
        self.assertEqual(dict(map(tuple, cols["status"]["values"])), {"On time": 32, "Late": 8})
        self.assertEqual(prof["alternative_column"], "vendor")
        ids = {c["id"]: c for c in prof["candidate_indicators"]}
        self.assertEqual(ids["PRICE_MEAN"]["suggested_direction"], "cost")
        self.assertEqual(ids["LATE_DAYS_MEAN"]["suggested_direction"], "cost")
        self.assertEqual(ids["STATUS_LATE_SHARE"]["suggested_direction"], "cost")  # two values: the rarer one
        region = next(v for k, v in ids.items() if k.startswith("REGION_"))
        self.assertIsNone(region["suggested_direction"])
        self.assertIn("cannot be inferred from the data", region["note"])
        self.assertNotIn("ID_MEAN", ids)
        j, m = eda.write(prof, self.d / "out")
        self.assertEqual(json.loads(j.read_text(encoding="utf-8"))["data_file"], "awk.csv")
        md = m.read_text(encoding="utf-8")
        self.assertIn("## Candidate indicators", md)
        self.assertIn("nothing here is a weight", md)
        rc, out, _ = cli(eda.main, ["--data", str(p), "--alternative-column", "region", "--out", str(self.d / "o2")])
        self.assertEqual(rc, 0)
        self.assertIn("alternative column: region", out)
        rc, _, err = cli(eda.main, ["--data", str(p), "--alternative-column", "nope", "--out", str(self.d / "o3")])
        self.assertEqual(rc, 2)
        self.assertIn("not found", err)

    def test_notes_without_alternatives_or_criteria(self):
        p = self.write("u.csv", "a,b\n" + "".join(f"{k},{k * 2}\n" for k in range(1, 12)))
        prof = eda.profile(p)
        self.assertIsNone(prof["alternative_column"])
        notes = " ".join(prof["notes"])
        self.assertIn("no column looks like an alternative", notes)
        self.assertIn("at least 3 criteria", notes)


class AggregationTests(TmpCase):
    def setUp(self):
        super().setUp()
        self.table = tabular.read_table(self.write("s.csv", SMALL_CSV))

    def agg(self, ind, filt=None, alt="S1"):
        rows = [r for r in self.table.rows if r[0] == alt]
        if filt:
            f = criteria._row_filter(criteria._conditions(filt, "")[0], self.table)
            rows = [r for r in rows if f(r)]
        return criteria.aggregate(ind, rows, self.table)

    def test_every_aggregation(self):
        self.assertEqual(self.agg({"aggregation": "mean", "column": "price"}), 12.0)
        self.assertEqual(self.agg({"aggregation": "sum", "column": "qty"}), 200.0)
        self.assertEqual(self.agg({"aggregation": "median", "column": "price", }, alt="S3"), 7.0)
        self.assertEqual(self.agg({"aggregation": "min", "column": "price"}, alt="S3"), 5.0)
        self.assertEqual(self.agg({"aggregation": "max", "column": "price"}, alt="S3"), 9.0)
        self.assertEqual(self.agg({"aggregation": "count"}), 3.0)
        self.assertEqual(self.agg({"aggregation": "count", "column": "price"}), 2.0)
        self.assertAlmostEqual(self.agg({"aggregation": "share", "column": "status", "value": "ok"}), 2 / 3)
        self.assertEqual(self.agg({"aggregation": "share", "column": "status", "value": ["ok", "late"]}), 1.0)
        self.assertAlmostEqual(self.agg({"aggregation": "ratio", "numerator": "defects", "denominator": "qty"}), 4 / 200)
        self.assertAlmostEqual(self.agg({"aggregation": "mean_days_between", "start": "ordered", "end": "delivered"}),
                               (4 + 8 + 1) / 3)

    def test_filters(self):
        mean = {"aggregation": "mean", "column": "price"}
        self.assertEqual(self.agg(mean, {"column": "status", "equals": "late"}, "S3"), 7.0)
        self.assertEqual(self.agg(mean, {"column": "status", "not_equals": "late"}, "S3"), 7.0)
        self.assertEqual(self.agg(mean, {"column": "check", "in": ["Pass"]}, "S3"), 8.0)
        self.assertEqual(self.agg(mean, {"column": "check", "not_in": ["Pass"]}, "S3"), 5.0)
        self.assertEqual(self.agg(mean, {"column": "qty", "gt": 50}, "S3"), 9.0)
        self.assertEqual(self.agg(mean, {"column": "qty", "ge": 50}, "S3"), 7.0)
        self.assertEqual(self.agg(mean, {"column": "qty", "lt": 100}, "S3"), 6.0)
        self.assertEqual(self.agg(mean, {"column": "qty", "le": 100}, "S3"), 7.0)
        self.assertEqual(self.agg(mean, [{"column": "qty", "le": 100}, {"column": "status", "equals": "late"}], "S3"), 7.0)
        self.assertIsNone(self.agg(mean, {"column": "status", "equals": "none"}, "S3"))
        self.assertEqual(self.agg({"aggregation": "count"}, {"column": "qty", "equals": 50}), 2.0)

    def test_compute_matrix_scale_offset(self):
        res = criteria.compute(small_spec(), self.table)
        self.assertEqual(res["alternatives"], ["S1", "S2", "S3"])
        self.assertEqual(res["matrix"][0], [12.0, float(f"{2 / 3:.6g}"), 2.0, float(f"{13 / 3:.6g}"), 3.0])
        spec = small_spec(alternatives=["S3", "S1"])
        spec["indicators"][1]["offset"] = 1
        res = criteria.compute(spec, self.table)
        self.assertEqual(res["alternatives"], ["S3", "S1"])
        self.assertAlmostEqual(res["matrix"][0][1], 1 + 1 / 3, places=5)


class SpecValidationTests(TmpCase):
    def setUp(self):
        super().setUp()
        self.data = self.write("s.csv", SMALL_CSV)
        self.table = tabular.read_table(self.data)

    def errors(self, spec):
        return " | ".join(criteria.validate_against(spec, self.data))

    def test_valid(self):
        self.assertEqual(criteria.validate_against(small_spec(), self.data), [])

    def test_structure_errors(self):
        s = small_spec()
        s["indicators"][0]["direction"] = "REVIEW: benefit or cost"
        self.assertIn("indicator P: direction must be 'benefit' or 'cost' (still a REVIEW placeholder)", self.errors(s))
        s = small_spec(problem={"name": "Bad Name", "goal": "REVIEW: x", "role": ""})
        e = self.errors(s)
        self.assertIn("problem.goal: still a REVIEW placeholder", e)
        self.assertIn("problem.role: required", e)
        self.assertIn("problem.name: lower-case", e)
        s = small_spec()
        s["indicators"][0]["aggregation"] = "average"
        self.assertIn("aggregation must be one of", self.errors(s))
        s = small_spec()
        del s["indicators"][2]["denominator"]
        self.assertIn("aggregation 'ratio' needs 'denominator'", self.errors(s))
        s = small_spec()
        s["indicators"][0]["weight"] = 0.4
        self.assertIn("weights or importance are not part of a criteria spec", self.errors(s))
        s = small_spec()
        s["hierarchy"]["criteria"][0]["indicator"] = "ZZ"
        self.assertIn("maps to unknown indicator 'ZZ'", self.errors(s))
        self.assertIn("indicators not used by any leaf criterion: P", self.errors(s))
        s = small_spec()
        s["hierarchy"]["criteria"][1]["indicator"] = "P"
        self.assertIn("used by more than one leaf: P", self.errors(s))
        s = small_spec()
        s["hierarchy"]["criteria"] = s["hierarchy"]["criteria"][:2]
        self.assertIn("2 main criteria; an AHP matrix needs 3-9", self.errors(s))
        s = small_spec()
        s["hierarchy"]["criteria"][2]["sub_criteria"].pop()
        self.assertIn("2 sub-criteria; a group needs 3-9", self.errors(s))
        s = small_spec()
        s["indicators"][0]["filter"] = {"column": "qty", "gt": "x"}
        self.assertIn("'gt' needs a number", self.errors(s))
        s = small_spec()
        s["indicators"][1]["id"] = "P"
        self.assertIn("duplicate ids", self.errors(s))

    def test_data_errors(self):
        s = small_spec(alternative_column="vendor")
        self.assertIn("column 'vendor' not in the data", self.errors(s))
        s = small_spec()
        s["indicators"][0]["column"] = "status"
        self.assertIn("column 'status' is not numeric", self.errors(s))
        s = small_spec()
        s["indicators"][1]["value"] = "early"
        self.assertIn("value 'early' never occurs", self.errors(s))
        s = small_spec()
        s["indicators"][3]["start"] = "price"
        self.assertIn("does not hold dates", self.errors(s))
        s = small_spec()
        s["indicators"][1]["value"] = "late"   # S2 has no late orders -> share 0
        self.assertIn("value 0 for alternative 'S2' is not > 0", self.errors(s))
        s = small_spec(alternatives=["S1", "S9"])
        self.assertIn("alternatives not found", self.errors(s))
        s = small_spec(filter={"column": "supplier", "equals": "S1"})
        self.assertIn("1 alternatives after filtering", self.errors(s))
        s = small_spec()
        s["indicators"][0]["filter"] = {"column": "status", "equals": "late"}
        self.assertIn("no usable rows for alternative 'S2'", self.errors(s))


class BuildAndTemplateTests(TmpCase):
    def setUp(self):
        super().setUp()
        self.data = self.write("s.csv", SMALL_CSV)
        self.spec = self.d / "spec.json"
        self.spec.write_text(json.dumps(small_spec()), encoding="utf-8")

    def test_build_problem_with_provenance(self):
        out = self.d / "problems" / "small.json"
        rc, text, err = cli(criteria.main, ["build", "--spec", str(self.spec), "--data", str(self.data), "--out", str(out)])
        self.assertEqual(rc, 0, err)
        self.assertIn("Goal: choosing a supplier", text)
        self.assertIn("criteria_confirmed_by_user is false", text)
        p = pb.load_problem(out)
        self.assertEqual(p.task_names, ("main", "delivery"))
        self.assertEqual(p.ranking["criterion_indicator"]["Speed"], "LT")
        prov = p.raw["provenance"]
        self.assertEqual(prov["data_file"], "s.csv")
        self.assertEqual(prov["data_rows"], 8)
        self.assertEqual(prov["criteria_proposed_by"], "human")
        self.assertFalse(prov["criteria_confirmed_by_user"])
        self.assertEqual(len(prov["data_sha256"]), 64)
        self.assertEqual(len(prov["spec_sha256"]), 64)
        self.assertEqual(prov["data_summary"]["missing_cells_in_used_columns"]["price"], 1)
        ind = {i["code"]: i for i in p.ranking["indicators"]}
        self.assertEqual(ind["D"]["source"], "defects / qty")
        self.assertEqual(ind["D"]["scale"], 100)
        before = out.read_bytes()
        self.assertEqual(cli(criteria.main, ["build", "--spec", str(self.spec), "--data", str(self.data),
                                             "--out", str(out)])[0], 0)   # same content: rebuild is allowed
        self.assertEqual(out.read_bytes(), before)
        rc, _, err = cli(criteria.main, ["build", "--spec", str(self.spec), "--data", str(self.data), "--out",
                                         str(out), "--confirmed"])
        self.assertEqual(rc, 2)
        self.assertIn("use --force", err)
        self.assertEqual(cli(criteria.main, ["build", "--spec", str(self.spec), "--data", str(self.data), "--out",
                                             str(out), "--confirmed", "--force"])[0], 0)
        self.assertTrue(pb.load_problem(out).raw["provenance"]["criteria_confirmed_by_user"])

    def test_weighting_warns_when_not_confirmed(self):
        out = self.d / "small.json"
        cli(criteria.main, ["build", "--spec", str(self.spec), "--data", str(self.data), "--out", str(out)])
        rc, _, err = cli(run.main, ["--problem", str(out), "--results", str(self.d / "res"), "--budget", "1",
                                    "--dry-run"], root=self.d)
        self.assertEqual(rc, 0)
        self.assertIn("not confirmed by the user", err)
        rc, _, err = cli(host.main, ["prompts", "--problem", str(out), "--panelists", "1", "--out", str(self.d / "w")])
        self.assertEqual(rc, 0)
        self.assertIn("not confirmed by the user", err)
        from ahp_bench import guidance
        self.assertEqual(cli(guidance.main, ["confirm", "--problem", str(out), "--criteria"])[0], 0)
        rc, _, err = cli(run.main, ["--problem", str(out), "--results", str(self.d / "res"), "--budget", "1",
                                    "--dry-run"], root=self.d)
        self.assertNotIn("not confirmed", err)

    def test_validate_reports_errors(self):
        bad = self.d / "bad.json"
        s = small_spec()
        s["indicators"][0]["direction"] = "up"
        bad.write_text(json.dumps(s), encoding="utf-8")
        rc, _, err = cli(criteria.main, ["validate", "--spec", str(bad), "--data", str(self.data)])
        self.assertEqual(rc, 2)
        self.assertIn("direction must be 'benefit' or 'cost'", err)
        bad.write_text("{not json", encoding="utf-8")
        rc, _, err = cli(criteria.main, ["validate", "--spec", str(bad), "--data", str(self.data)])
        self.assertIn("invalid JSON", err)

    def test_template_is_a_draft(self):
        prof = eda.profile(self.data)
        eda.write(prof, self.d / "eda")
        out = self.d / "draft.json"
        rc, text, _ = cli(criteria.main, ["template", "--eda", str(self.d / "eda" / "eda.json"), "--out", str(out)])
        self.assertEqual(rc, 0)
        self.assertIn("DRAFT", text)
        spec = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("A person must review", spec["_DRAFT"])
        errs = criteria.validate_against(spec, self.data)
        self.assertTrue(any("REVIEW placeholder" in e for e in errs), errs)
        self.assertEqual(cli(criteria.main, ["template", "--eda", str(self.d / "eda" / "eda.json"),
                                             "--out", str(out)])[0], 2)


class ProposalTests(TmpCase):
    """Criteria proposal: host mode (prompt file + canned reply) and API mode (dry run, fake transport)."""

    def setUp(self):
        super().setUp()
        self.data = self.write("s.csv", SMALL_CSV)
        eda.write(eda.profile(self.data), self.d / "eda")
        self.eda = self.d / "eda" / "eda.json"
        s = small_spec()
        self.reply = {"alternative_column": "supplier", "goal": "choosing a supplier",
                      "indicators": s["indicators"], "hierarchy": s["hierarchy"], "notes": "none"}
        self.reply["hierarchy"]["criteria"][0]["weight"] = 0.5   # must be removed

    def test_prompt_rules(self):
        msgs = criteria.criteria_messages(json.loads(self.eda.read_text(encoding="utf-8")), "choosing a supplier")
        user = msgs[1]["content"]
        self.assertIn("Do not assign weights, importance levels or a priority order", user)
        self.assertIn("state its direction", user)
        self.assertIn("rationale", user)
        self.assertIn("only criteria that the columns above can measure", user)
        self.assertIn("- price (integer; 1 missing", user)
        self.assertNotIn("S1 (", user.split("Columns:")[0])

    def test_host_criteria_prompt_and_ingest(self):
        w = self.d / "crit"
        rc, out, _ = cli(host.main, ["criteria-prompt", "--eda", str(self.eda), "--out", str(w)])
        self.assertEqual(rc, 0)
        prompt = (w / "criteria_prompt.txt").read_text(encoding="utf-8")
        self.assertTrue(prompt.startswith("=== SYSTEM ===\n"))
        (w / "criteria_reply.json").write_text("```json\n" + json.dumps(self.reply) + "\n```", encoding="utf-8")
        spec = self.d / "spec.json"
        rc, out, err = cli(host.main, ["criteria-ingest", "--dir", str(w), "--data", str(self.data), "--out", str(spec),
                                       "--name", "small"])
        self.assertEqual(rc, 0, err)
        self.assertIn("removed weight / importance fields", out)
        s = json.loads(spec.read_text(encoding="utf-8"))
        self.assertEqual(s["proposed_by"], "host agent")
        self.assertNotIn("weight", json.dumps(s["hierarchy"]))
        self.assertEqual(criteria.validate_against(s, self.data), [])
        log = store.read_jsonl(w / "criteria_calls.jsonl")
        self.assertEqual(len(log), 1)
        self.assertTrue(log[0]["parse_ok"])
        self.assertEqual(log[0]["removed_weight_fields"], ["hierarchy.criteria.0.weight"])
        self.assertIn("raw_reply", log[0])
        self.assertEqual(log[0]["messages"][1]["content"], prompt.split("=== USER ===\n", 1)[1].rstrip("\n"))
        prob = self.d / "small.json"
        self.assertEqual(cli(criteria.main, ["build", "--spec", str(spec), "--data", str(self.data), "--out",
                                             str(prob)])[0], 0)
        self.assertEqual(pb.load_problem(prob).raw["provenance"]["criteria_proposed_by"], "host agent")
        # a reply that is not JSON is logged and rejected
        (w / "criteria_reply.json").write_text("I cannot do that.", encoding="utf-8")
        rc, _, err = cli(host.main, ["criteria-ingest", "--dir", str(w), "--data", str(self.data), "--out",
                                     str(self.d / "x.json")])
        self.assertEqual(rc, 2)
        self.assertFalse(store.read_jsonl(w / "criteria_calls.jsonl")[-1]["parse_ok"])
        # a different data file is refused
        other = self.write("o.csv", SMALL_CSV.replace("S3", "S4"))
        (w / "criteria_reply.json").write_text(json.dumps(self.reply), encoding="utf-8")
        rc, _, err = cli(host.main, ["criteria-ingest", "--dir", str(w), "--data", str(other), "--out",
                                     str(self.d / "y.json")])
        self.assertEqual(rc, 2)
        self.assertIn("differs from the data file", err)

    def test_api_propose_dry_run_and_fake_call(self):
        calls = []

        def fake(url, headers, body):
            calls.append(body)
            return 200, {"id": "gen-1", "provider": "Fake", "choices": [{"message": {"content": json.dumps(self.reply)},
                                                                          "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 900, "completion_tokens": 300, "total_tokens": 1200, "cost": 0.002}}

        spec = self.d / "spec.json"
        base = ["propose", "--eda", str(self.eda), "--data", str(self.data), "--out", str(spec), "--model",
                "test/model", "--budget", "0.01", "--name", "small"]
        rc, out, _ = cli(criteria.main, base + ["--dry-run"], transport=fake)
        self.assertEqual((rc, calls), (0, []))
        self.assertIn("dry run: 1 planned call to test/model", out)
        self.assertIn("Do not assign weights", out)
        env = {k: v for k, v in os.environ.items() if not k.startswith("OPENROUTER_")}
        env["OPENROUTER_API_KEY"] = "test-SECRET-key"
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("ahp_bench.config.load_env", lambda *a, **k: []):
            rc, out, err = cli(criteria.main, base, transport=fake)
            self.assertEqual(rc, 0, err)
            self.assertEqual(len(calls), 1)
            s = json.loads(spec.read_text(encoding="utf-8"))
            self.assertEqual(s["proposed_by"], "test/model")
            log_path = self.d / "eda" / "criteria_calls.jsonl"
            self.assertNotIn("SECRET", log_path.read_text(encoding="utf-8"))
            log = store.read_jsonl(log_path)
            self.assertEqual(log[0]["cost"], 0.002)
            self.assertEqual(log[0]["source"], "api")
            rc, _, err = cli(criteria.main, base[:-4] + ["--budget", "0.001"], transport=fake)
            self.assertEqual(rc, 2)
            self.assertIn("budget reached", err)
            self.assertEqual(len(calls), 1)


class HierarchyTests(TmpCase):
    def test_outputs_for_example_problem(self):
        p = pb.load_problem(pb.EXAMPLE_PROBLEM_PATH)
        tree = hierarchy.text_tree(p)
        self.assertTrue(tree.startswith("Goal: choosing a laptop"))
        self.assertIn("|   +-- Processor Speed  [P1: Processor benchmark score (points, benefit)", tree)
        self.assertIn("`-- Portability  [W: Weight (kg, cost)", tree)
        rc, out, _ = cli(hierarchy.main, ["--problem", str(pb.EXAMPLE_PROBLEM_PATH), "--out", str(self.d / "h")])
        self.assertEqual(rc, 0)
        md = (self.d / "h" / "hierarchy.md").read_text(encoding="utf-8")
        self.assertIn("```mermaid\ngraph TD", md)
        self.assertIn('M1S2["Memory Capacity"]', md)
        self.assertIn("| Main criterion | Criterion | Indicator | Unit | Direction | Source column(s) | Aggregation |", md)
        try:
            import matplotlib  # noqa: F401
            self.assertEqual((self.d / "h" / "hierarchy.png").read_bytes()[:4], b"\x89PNG")
        except ImportError:
            self.assertIn("no hierarchy.png", out)

    def test_without_matplotlib_and_without_ranking(self):
        d = json.loads(pb.EXAMPLE_PROBLEM_PATH.read_text(encoding="utf-8"))
        d.pop("ranking")
        p = pb.problem_from_dict(d)
        self.assertNotIn("[", hierarchy.text_tree(p))
        with mock.patch.dict(sys.modules, {"matplotlib": None, "matplotlib.figure": None,
                                           "matplotlib.backends.backend_agg": None, "matplotlib.patches": None}):
            res = hierarchy.write(p, self.d / "h2")
        self.assertIsNone(res["png"])
        self.assertIn("graph TD", res["md"].read_text(encoding="utf-8"))


class RankGivenWeightsTests(TmpCase):
    def test_rank_with_leaf_and_local_weights(self):
        from ahp_bench import rank
        p = str(pb.EXAMPLE_PROBLEM_PATH)
        wl = self.write("w.json", json.dumps({"Processor Speed": 2, "Memory Capacity": 1, "Storage Speed": 1,
                                              "Price": 4, "Portability": 2}))
        rc, out, err = cli(rank.main, ["--problem", p, "--weights", str(wl), "--out", str(self.d / "r1")])
        self.assertEqual(rc, 0, err)
        md = (self.d / "r1" / "rank.md").read_text(encoding="utf-8")
        self.assertIn("supplied by the user", md)
        self.assertIn("| Price | 0.4000 |", md)
        for stem in ("R01_rankings_given_weights", "R02_equal_weight_baseline", "R03_weight_sensitivity"):
            self.assertTrue((self.d / "r1" / "tables" / f"{stem}.csv").exists())
        loc = self.write("l.json", json.dumps({"main": {"Performance": 0.4, "Price": 0.4, "Portability": 0.2},
                                               "performance": {"Processor Speed": 0.5, "Memory Capacity": 0.25,
                                                               "Storage Speed": 0.25}}))
        self.assertEqual(cli(rank.main, ["--problem", p, "--weights", str(loc), "--out", str(self.d / "r2")])[0], 0)
        self.assertEqual((self.d / "r1" / "tables" / "R01_rankings_given_weights.csv").read_bytes(),
                         (self.d / "r2" / "tables" / "R01_rankings_given_weights.csv").read_bytes())
        bad = self.write("b.json", json.dumps({"Price": 1}))
        rc, _, err = cli(rank.main, ["--problem", p, "--weights", str(bad)])
        self.assertEqual(rc, 2)
        self.assertIn("one value per leaf criterion", err)


class ExampleWalkthroughTests(TmpCase):
    def test_generator_reproduces_csv(self):
        sys.path.insert(0, str(EXAMPLES))
        try:
            import make_orders_synthetic as gen
        finally:
            sys.path.remove(str(EXAMPLES))
        self.assertEqual(gen.generate().encode("utf-8"), (EXAMPLES / "orders_synthetic.csv").read_bytes())
        self.assertTrue((EXAMPLES / "orders_synthetic.csv").read_text(encoding="utf-8").startswith("# SYNTHETIC DATA"))

    def test_csv_to_ranking_host_mode(self):
        w = self.d / "ex"
        rc, out, _ = cli(entry.main, ["example", "--out", str(w)])
        self.assertEqual(rc, 0)
        self.assertIn("invented data", out)
        self.assertEqual(cli(example.main, ["--out", str(w)])[0], 2)    # no silent overwrite
        csv, spec = w / "orders_synthetic.csv", w / "orders_spec.json"
        self.assertEqual(cli(entry.main, ["eda", "--data", str(csv), "--out", str(w / "eda")])[0], 0)
        prof = json.loads((w / "eda" / "eda.json").read_text(encoding="utf-8"))
        self.assertEqual((prof["rows"], prof["alternative_column"]), (400, "supplier"))
        prob = w / "problems" / "orders_synthetic.json"
        rc, out, err = cli(entry.main, ["criteria", "build", "--spec", str(spec), "--data", str(csv), "--out",
                                        str(prob), "--confirmed"])
        self.assertEqual(rc, 0, err)
        p = pb.load_problem(prob)
        self.assertEqual(len(p.ranking["alternatives"]), 5)
        self.assertEqual(p.leaf_short, ["Cost", "Quality", "On-time Delivery", "Lead Time", "Cancellations", "Compliance"])
        self.assertEqual(cli(entry.main, ["hierarchy", "--problem", str(prob), "--out", str(w / "hier")])[0], 0)
        work, res = w / "work", w / "results"
        rc, _, err = cli(entry.main, ["host", "prompts", "--problem", str(prob), "--conditions", "direct,cot",
                                      "--panelists", "3", "--out", str(work)])
        self.assertEqual(rc, 0)
        self.assertEqual(err, "")    # confirmed criteria: no warning
        for f in sorted((work / "prompts").glob("*_p*.txt")):
            text = f.read_text(encoding="utf-8")
            vals = ("9", "1/9", "5") if "_p02_" in f.name else ("2", "3", "1/2", "1", "3", "2")
            reply = json.loads(canned_answers(text, vals))
            if "cot_" in f.name:
                reply.update(reasoning="canned", ranking=["x"])
            (work / "replies" / f"{f.stem}.json").write_text(json.dumps(reply), encoding="utf-8")
        rc, out, _ = cli(entry.main, ["host", "ingest", "--problem", str(prob), "--dir", str(work), "--results",
                                      str(res), "--tag", "main", "--model-name", "canned"])
        self.assertIn("12 records written (12 parsed)", out)
        self.assertEqual(cli(entry.main, ["report", "--problem", str(prob), "--results", str(res)])[0], 0)
        md = (res / "report" / "report.md").read_text(encoding="utf-8")
        for part in ("## Decision hierarchy", "```mermaid", "## Data set", "orders_synthetic.csv", "## H01 Criteria and",
                     "## H02 Performance matrix", "## G04 Global weights", "## G07 Rankings"):
            self.assertIn(part, md)
        self.assertLess(md.index("## H02"), md.index("## G03a"))
        self.assertLess(md.index("## Data set"), md.index("## G07"))
        self.assertTrue((res / "report" / "hierarchy.md").exists())
        self.assertTrue((res / "report" / "tables" / "H02_performance_matrix.csv").exists())
        self.assertEqual(cli(entry.main, ["audit", "--problem", str(prob), "--results", str(res)])[0], 0)
        import csv as _csv
        with (res / "report" / "audit" / "expectations.csv").open(encoding="utf-8") as f:
            exp = list(_csv.DictReader(f))
        self.assertTrue(exp)
        self.assertEqual([e for e in exp if e["status"] != "PASS"], [])

    def test_api_mode_dry_run_from_csv(self):
        w = self.d / "ex"
        example.copy_example(w)
        prob = w / "p.json"
        cli(criteria.main, ["build", "--spec", str(w / "orders_spec.json"), "--data", str(w / "orders_synthetic.csv"),
                            "--out", str(prob)])
        eda.write(eda.profile(w / "orders_synthetic.csv"), w / "eda")
        rc, out, _ = cli(criteria.main, ["propose", "--eda", str(w / "eda" / "eda.json"), "--data",
                                         str(w / "orders_synthetic.csv"), "--out", str(w / "s.json"), "--model",
                                         "claude-sonnet-4.6", "--budget", "0.05", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertIn("anthropic/claude-sonnet-4.6", out)
        rc, out, err = cli(run.main, ["--problem", str(prob), "--results", str(w / "res"), "--conditions", "direct",
                                      "--reps", "1", "--budget", "1", "--dry-run"], root=w)
        self.assertEqual(rc, 0)
        self.assertIn("18 planned calls", out)
        self.assertIn("not confirmed by the user", err)
        self.assertFalse((w / "res" / "raw").exists())


if __name__ == "__main__":
    unittest.main()
