"""End-to-end run + revise + report + audit on problems other than the study (fake transport, temp dirs)."""
import csv
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

from ahp_bench import audit, cli, report, revise, run, store  # noqa: E402
from ahp_bench import problem as pb  # noqa: E402

QUESTION = re.compile(r"^\d+\.\s+Left: .+ \| Right: .+$", re.M)
PREVIOUS = re.compile(r"^Q\d+: Left .+ \| Right .+ = \S+$", re.M)
JUDGE_ID = "test/judge-model"

MODELS_FILE = {
    "defaults": {"temperature": 0.6, "max_tokens": 1000, "reasoning": {"effort": "none"}},
    "models": {
        "alpha": {"id": "test/alpha", "env": "OPENROUTER_KEY_TEST_ALPHA", "developer": "Lab A"},
        "beta": {"id": "test/beta", "developer": "Lab B", "reasoning": None},
    },
    "judge": {"key": "judge", "id": JUDGE_ID, "developer": "Lab J", "env": "OPENROUTER_KEY_TEST_JUDGE",
              "temperature": 0.0, "reasoning": None},
}


def answers_json(n, value_fn):
    return json.dumps({"reasoning": "r", "ranking": ["x"],
                       "answers": [{"question": q, "value": value_fn(q)} for q in range(1, n + 1)]})


class Fake:
    """alpha: first answers circular (inconsistent), revisions all equal; beta: always consistent."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, headers, body):
        self.calls.append(body)
        msgs = body["messages"]
        if body["model"] == JUDGE_ID:
            if "coordinate an AHP elicitation" in msgs[0]["content"]:
                content = json.dumps({"feedback": "Some of your answers contradict each other; please review them."})
            else:
                content = json.dumps({"logic_preserved": "partly", "changed_priorities": [], "comment": "ok"})
        else:
            last = msgs[-1]["content"]
            revision = len(msgs) > 2
            n = len(PREVIOUS.findall(last)) if revision else len(QUESTION.findall(last))
            if body["model"] == "test/alpha" and not revision:
                content = answers_json(n, lambda q: "9" if q % 2 else "1/9")
            else:
                content = answers_json(n, lambda q: "1")
        return 200, {"id": f"gen-{len(self.calls)}", "provider": "FakeProvider",
                     "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60, "cost": 0.0001}}


class GenericEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        env = {k: v for k, v in os.environ.items() if not k.startswith("OPENROUTER_")}
        env.update(OPENROUTER_API_KEY="shared-test-SECRET", OPENROUTER_KEY_TEST_ALPHA="alpha-test-SECRET")
        self.env = mock.patch.dict(os.environ, env, clear=True)
        self.env.start()
        self.stdout = io.StringIO()
        self.out = mock.patch("sys.stdout", self.stdout)
        self.out.start()
        self.models = self.root / "models.json"
        self.models.write_text(json.dumps(MODELS_FILE), encoding="utf-8")
        self.problem = str(pb.EXAMPLE_PROBLEM_PATH)
        self.results = self.root / "res_laptop"

    def tearDown(self):
        self.out.stop()
        self.env.stop()
        self.tmp.cleanup()

    def common(self, problem=None, results=None):
        return ["--problem", problem or self.problem, "--results", str(results or self.results),
                "--models-file", str(self.models)]

    def expectations(self, out_dir):
        with (out_dir / "audit" / "expectations.csv").open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_run_revise_report_audit(self):
        fake = Fake()
        # dry run first: messages written, no calls
        self.assertEqual(run.main(self.common() + ["--reps", "3", "--budget", "1", "--dry-run"],
                                  transport=fake, root=self.root), 0)
        self.assertEqual(fake.calls, [])
        self.assertEqual(sorted(p.name for p in (self.results / "dry_run").glob("*.json")),
                         ["cot_main.json", "cot_performance.json", "lean_main.json", "lean_performance.json"])
        self.assertIn("24 planned calls", self.stdout.getvalue())
        self.assertIn("estimated cost", self.stdout.getvalue())

        self.assertEqual(run.main(self.common() + ["--reps", "3", "--budget", "1", "--tag", "main"],
                                  transport=fake, root=self.root), 0)
        calls = store.read_jsonl(self.results / "raw" / "calls.jsonl")
        self.assertEqual(len(calls), 2 * 2 * 2 * 3)
        self.assertTrue(all(r["parse_ok"] for r in calls))
        self.assertEqual({r["problem"] for r in calls}, {"example_laptop_choice"})
        alpha = [r for r in calls if r["model_key"] == "alpha"]
        self.assertTrue(all(r["CR"] > 0.1 for r in alpha))
        headers = {json.dumps(b["model"]) for b in fake.calls}
        self.assertEqual(headers, {'"test/alpha"', '"test/beta"'})
        self.assertNotIn("SECRET", (self.results / "raw" / "calls.jsonl").read_text(encoding="utf-8"))

        n_before = len(fake.calls)
        self.assertEqual(revise.main(self.common() + ["--source-tag", "main", "--tag", "revision", "--budget", "1",
                                                      "--judge", "--max-revisions", "3"],
                                     transport=fake, root=self.root), 0)
        revs = store.read_jsonl(self.results / "raw" / "revisions.jsonl")
        self.assertEqual(len(revs), 12)  # 12 inconsistent alpha matrices, each consistent after 1 revision
        self.assertTrue(all(r["parse_ok"] and r["CR"] < 1e-9 for r in revs))
        self.assertTrue(all(r["prompt_id"].startswith("example_laptop_choice:revision_judge_") for r in revs))
        judge = store.read_jsonl(self.results / "raw" / "judge_calls.jsonl")
        self.assertEqual(len(judge), 24)
        self.assertEqual(len(fake.calls) - n_before, 36)

        rc = report.main(self.common() + ["--revision-tag", "revision"])
        self.assertEqual(rc, 0)
        tables = sorted(p.name for p in (self.results / "report" / "tables").glob("*.csv"))
        for stem in ("G01_models_settings", "G02_consistency", "G03a_local_weights_main",
                     "G03b_local_weights_performance", "G04_global_weights", "G07_rankings", "G08_equal_weight_baseline",
                     "G09_weight_sensitivity", "G10_revision_summary", "G11_weights_before_after_revision",
                     "G12_logic_preservation"):
            self.assertIn(f"{stem}.csv", tables)
        self.assertFalse([t for t in tables if t.startswith("G06")])  # no experts given
        md = (self.results / "report" / "report.md").read_text(encoding="utf-8")
        self.assertIn("there is no human reference", md)
        logic = (self.results / "report" / "tables" / "G12_logic_preservation.csv").read_text(encoding="utf-8")
        self.assertIn("Panel,unguided,zero-shot,direct,main,3,3,1.000", logic)

        # with an expert file (two made-up experts)
        experts = self.root / "experts.json"
        eye3 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        experts.write_text(json.dumps({"experts": ["X1", "X2"], "main": [eye3, [[1, 3, 5], [1 / 3, 1, 3], [0.2, 1 / 3, 1]]],
                                       "performance": [eye3, eye3]}), encoding="utf-8")
        self.assertEqual(report.main(self.common() + ["--experts", str(experts), "--out", str(self.root / "rep2")]), 0)
        self.assertTrue((self.root / "rep2" / "tables" / "G06b_similarity_to_experts.csv").exists())
        g07 = (self.root / "rep2" / "tables" / "G07_rankings.csv").read_text(encoding="utf-8")
        self.assertIn("Experts,Laptop_A", g07)

        self.assertEqual(audit.main(["--problem", self.problem, "--results", str(self.results)]), 0)
        exp = self.expectations(self.results / "report")
        self.assertTrue(exp)
        self.assertEqual([e for e in exp if e["status"] != "PASS"], [])
        names = " ".join(e["check"] for e in exp)
        self.assertIn("stored messages = current code", names)
        self.assertIn("j revision: protocol rule violations", names)
        res = audit.run_problem_audit(pb.load_problem(self.problem), self.results, "main", "revision")
        self.assertEqual(res["revision"]["chains_expected"], 12)
        self.assertEqual(res["revision"]["chains_present"], 12)

    def test_audit_detects_tampered_prompt(self):
        fake = Fake()
        run.main(self.common() + ["--reps", "1", "--budget", "1", "--tag", "main", "--models", "beta"],
                 transport=fake, root=self.root)
        path = self.results / "raw" / "calls.jsonl"
        recs = store.read_jsonl(path)
        recs[0]["messages"][0]["content"] += " Quality matters most."
        path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        res = audit.run_problem_audit(pb.load_problem(self.problem), self.results)
        failed = [e["check"] for e in audit.expectations(res) if e["status"] == "FAIL"]
        self.assertTrue(any("stored messages = current code" in f for f in failed))

    def test_flat_problem_without_ranking(self):
        d = json.loads(pb.EXAMPLE_PROBLEM_PATH.read_text(encoding="utf-8"))
        for k in ("groups", "ranking", "short"):
            d.pop(k)
        d["name"] = "flat_six"
        d["criteria"] = [f"Criterion {c}" for c in "ABCDEF"]
        path = self.root / "flat.json"
        path.write_text(json.dumps(d), encoding="utf-8")
        fake = Fake()
        res_dir = self.root / "res_flat"
        self.assertEqual(run.main(self.common(str(path), res_dir) + ["--reps", "2", "--budget", "1", "--tag", "main"],
                                  transport=fake, root=self.root), 0)
        calls = store.read_jsonl(res_dir / "raw" / "calls.jsonl")
        self.assertEqual(len(calls), 2 * 2 * 1 * 2)
        self.assertTrue(all(len(r["answers"]) == 15 for r in calls))
        self.assertEqual(revise.main(self.common(str(path), res_dir) + ["--budget", "1"], transport=fake,
                                     root=self.root), 0)
        self.assertEqual(report.main(self.common(str(path), res_dir) + ["--revision-tag", "revision"]), 0)
        tables = sorted(p.stem for p in (res_dir / "report" / "tables").glob("*.csv"))
        self.assertFalse([t for t in tables if t[:3] in ("G07", "G08", "G09", "G06")])
        self.assertIn("G10_revision_summary", tables)
        self.assertEqual(audit.main(["--problem", str(path), "--results", str(res_dir)]), 0)
        self.assertEqual([e for e in self.expectations(res_dir / "report") if e["status"] != "PASS"], [])

    def test_refuses_mixing_problems_and_default_folder(self):
        fake = Fake()
        run.main(self.common() + ["--reps", "1", "--budget", "1", "--models", "beta"], transport=fake, root=self.root)
        flat = json.loads(pb.EXAMPLE_PROBLEM_PATH.read_text(encoding="utf-8"))
        flat["criteria"] = ["Speed", "Cost", "Weight"]
        for k in ("groups", "ranking", "short"):
            flat.pop(k)
        other = self.root / "other.json"
        other.write_text(json.dumps(flat), encoding="utf-8")
        with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            rc = run.main(self.common(str(other)) + ["--reps", "1", "--budget", "1"], transport=fake, root=self.root)
        self.assertEqual(rc, 2)
        self.assertIn("another --results folder", err.getvalue())
        ns = type("A", (), {"problem": self.problem, "results": None, "models_file": None})()
        self.assertEqual(cli.context(ns, self.root).results, self.root / "results" / "example_laptop_choice")
        ns.problem = None
        self.assertEqual(cli.context(ns, self.root).results, self.root / "results")


if __name__ == "__main__":
    unittest.main()
