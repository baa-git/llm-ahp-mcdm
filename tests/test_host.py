"""Host mode (answers from the coding agent's subagents, no API): prompt files, ingest, revision loop, and a
full offline round trip on the bundled example problem with canned replies. Also the condition aliases
and the study-data messages of the command line."""
import contextlib
import csv
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ahp_bench import audit, host, prompts, report, run  # noqa: E402
from ahp_bench import problem as pb  # noqa: E402

EXAMPLE = pb.EXAMPLE_PROBLEM_PATH
CYCLE = ["9", "1/9", "9"]          # 3 criteria, circular: CR far above 0.10
CONSISTENT = ["2", "4", "2"]


def reply_for(prompt_text: str, values) -> str:
    """Canned subagent reply in the format the prompt asks for."""
    n = len(re.findall(r"^\d+\.\s+Left:", prompt_text, re.M))
    answers = [{"question": q, "value": v} for q, v in enumerate(values[:n], 1)]
    if '"reasoning"' in prompt_text:
        return json.dumps({"reasoning": "step by step", "ranking": [], "answers": answers})
    return json.dumps({"answers": answers})


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rc = fn(*args)
    return rc, out.getvalue()


class HostRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.work, self.results = self.d / "work", self.d / "results"
        self.problem = pb.load_problem(EXAMPLE)

    def tearDown(self):
        self.tmp.cleanup()

    def common(self, *more):
        return ["--problem", str(EXAMPLE), "--dir", str(self.work), "--results", str(self.results), *more]

    def answer_all(self, sub, reply_dir, values_for):
        for p in sorted((self.work / sub).glob("*.txt")):
            if p.name == "README.txt":
                continue
            text = p.read_text(encoding="utf-8")
            (self.work / reply_dir).mkdir(parents=True, exist_ok=True)
            (self.work / reply_dir / f"{p.stem}.json").write_text(reply_for(text, values_for(p.stem)), encoding="utf-8")

    def test_prompt_files_equal_api_messages(self):
        rc, _ = quiet(host.main, ["prompts", "--problem", str(EXAMPLE), "--conditions", "unguided-direct,CoT",
                                  "--panelists", "2", "--reps", "2", "--out", str(self.work)])
        self.assertEqual(rc, 0)
        m = json.loads((self.work / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(m["entries"]), 2 * 2 * 2 * 2)           # conditions x tasks x panelists x reps
        self.assertEqual(sorted({e["rep"] for e in m["entries"]}), [1, 2, 3, 4])
        self.assertEqual({e["condition"] for e in m["entries"]}, {"lean", "cot"})   # stored keys
        for e in m["entries"]:
            msgs = prompts.build_messages(e["condition"], e["task"], self.problem)
            text = (self.work / e["prompt_file"]).read_text(encoding="utf-8")
            self.assertEqual(text, host.render(msgs))
            self.assertEqual(e["prompt_sha256"], prompts.prompt_sha256(msgs))
            self.assertTrue(e["id"].startswith(("direct_", "cot_")))

    def test_full_round_trip_with_revision_report_and_audit(self):
        quiet(host.main, ["prompts", "--problem", str(EXAMPLE), "--panelists", "3", "--out", str(self.work)])
        # panelist 1 answers the main task circularly; everything else is consistent
        self.answer_all("prompts", "replies", lambda s: CYCLE if "_main_p01" in s else CONSISTENT)
        (self.work / "replies" / "cot_performance_p03_r01.json").unlink()   # one reply still missing
        rc, out = quiet(host.main, ["ingest", *self.common("--tag", "main", "--model-name", "host-model")])
        self.assertEqual(rc, 0)
        self.assertIn("11 records written (11 parsed)", out)
        self.assertIn("1 replies missing", out)
        self.answer_all("prompts", "replies", lambda s: CONSISTENT)
        rc, out = quiet(host.main, ["ingest", *self.common("--tag", "main", "--model-name", "host-model")])
        self.assertIn("1 records written", out)
        self.assertIn("11 already ingested", out)
        recs = [json.loads(x) for x in (self.results / "raw" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(recs), 12)
        r0 = recs[0]
        self.assertEqual((r0["source"], r0["provider"], r0["cost"], r0["temperature_requested"]), ("host", "host", 0.0, None))
        self.assertEqual(r0["problem"], "example_laptop_choice")
        self.assertTrue(all(r["parse_ok"] for r in recs))

        # deterministic revision loop: two inconsistent chains (unguided and CoT main, panelist 1)
        rc, out = quiet(host.main, ["revise-prompts", *self.common("--source-tag", "main", "--tag", "revision")])
        self.assertIn("2 matrices with CR > 0.10; 2 revision prompts written", out)
        man = json.loads((self.work / "revise_manifest.json").read_text(encoding="utf-8"))
        e = man["entries"][0]
        self.assertEqual(e["messages"][-2]["role"], "assistant")
        self.assertIn("CR", e["messages"][-1]["content"])
        self.answer_all("revise", "revise_replies", lambda s: CYCLE)        # still inconsistent
        rc, out = quiet(host.main, ["revise-ingest", *self.common("--tag", "revision", "--model-name", "host-model")])
        self.assertIn("2 revision records written (2 parsed, 0 now CR <= 0.10)", out)
        rc, out = quiet(host.main, ["revise-ingest", *self.common("--tag", "revision", "--model-name", "host-model")])
        self.assertIn("2 stale entries skipped", out)                       # no double ingest
        quiet(host.main, ["revise-prompts", *self.common("--tag", "revision")])
        for p in (self.work / "revise_replies").glob("*.json"):
            p.unlink()
        self.answer_all("revise", "revise_replies", lambda s: CONSISTENT)
        rc, out = quiet(host.main, ["revise-ingest", *self.common("--tag", "revision", "--model-name", "host-model")])
        self.assertIn("2 now CR <= 0.10", out)
        rc, out = quiet(host.main, ["revise-prompts", *self.common("--tag", "revision")])
        self.assertIn("0 revision prompts written", out)

        # report and audit work unchanged on host records
        rc, _ = quiet(report.main, ["--problem", str(EXAMPLE), "--results", str(self.results), "--revision-tag", "revision"])
        self.assertEqual(rc, 0)
        md = (self.results / "report" / "report.md").read_text(encoding="utf-8")
        self.assertIn("direct", md)
        g10 = (self.results / "report" / "tables" / "G10_revision_summary.csv").read_text(encoding="utf-8")
        self.assertIn("Panel,unguided,zero-shot,direct,main,3,1", g10)
        rc, out = quiet(audit.main, ["--problem", str(EXAMPLE), "--results", str(self.results)])
        self.assertIn("0 failed", out)
        exp = list(csv.DictReader(io.StringIO((self.results / "report" / "audit" / "expectations.csv")
                                              .read_text(encoding="utf-8"))))
        self.assertTrue(exp and all(e["status"] == "PASS" for e in exp), [e for e in exp if e["status"] != "PASS"])

    def test_judge_feedback_by_host(self):
        quiet(host.main, ["prompts", "--problem", str(EXAMPLE), "--conditions", "direct", "--tasks", "main",
                          "--panelists", "1", "--out", str(self.work)])
        self.answer_all("prompts", "replies", lambda s: CYCLE)
        quiet(host.main, ["ingest", *self.common("--model-name", "m")])
        rc, out = quiet(host.main, ["revise-prompts", *self.common("--judge")])
        self.assertIn("0 revision prompts written", out)
        self.assertIn("1 judge prompts pending", out)
        jp = next((self.work / "judge").glob("direct_*.txt"))
        self.assertIn("Most contradictory comparisons", jp.read_text(encoding="utf-8"))
        (self.work / "judge_replies" / f"{jp.stem}.json").write_text(
            json.dumps({"feedback": "Questions 1 to 3 contradict each other."}), encoding="utf-8")
        rc, out = quiet(host.main, ["revise-prompts", *self.common("--judge")])
        self.assertIn("1 revision prompts written", out)
        e = json.loads((self.work / "revise_manifest.json").read_text(encoding="utf-8"))["entries"][0]
        self.assertTrue(e["messages"][-1]["content"].startswith("Questions 1 to 3 contradict each other."))
        self.answer_all("revise", "revise_replies", lambda s: CONSISTENT)
        quiet(host.main, ["revise-ingest", *self.common("--model-name", "m")])
        rev = json.loads((self.results / "raw" / "revisions.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual((rev["judge_source"], rev["judge_feedback_ok"]), ("host", True))
        self.assertNotIn("judge_model_id", rev)
        res = audit.run_problem_audit(self.problem, self.results, "main", "revision")
        self.assertEqual(res["revision"]["violations"], [])

    def test_ingest_refuses_other_problem(self):
        quiet(host.main, ["prompts", "--problem", str(EXAMPLE), "--panelists", "1", "--out", str(self.work)])
        with self.assertRaises(SystemExit):
            host.main(["ingest", "--problem", str(pb.STUDY_PROBLEM_PATH), "--dir", str(self.work),
                       "--results", str(self.results), "--model-name", "m"])


class ConditionNames(unittest.TestCase):
    def test_labels_and_aliases(self):
        self.assertEqual([prompts.condition_label(c) for c in prompts.CONDITIONS], ["direct", "CoT", "guided"])
        self.assertEqual([prompts.condition_full(c) for c in prompts.CONDITIONS],
                         ["Unguided, zero-shot (direct)", "Unguided, CoT", "Guided (zero-shot)"])
        self.assertEqual([(prompts.condition_guidance(c), prompts.condition_technique(c)) for c in prompts.CONDITIONS],
                         [("unguided", "zero-shot"), ("unguided", "CoT"), ("guided", "zero-shot")])
        self.assertEqual(prompts.condition_guidance("direct"), "unguided")   # labels map too
        for name in ("direct", "Direct", "unguided-direct", "unguided_direct", "lean"):
            self.assertEqual(prompts.condition_key(name), "lean")
        for name in ("CoT", "unguided-cot", "cot"):
            self.assertEqual(prompts.condition_key(name), "cot")
        self.assertEqual(run._conditions("unguided-direct,CoT", ("lean", "cot")), ["lean", "cot"])
        self.assertEqual(run._conditions("all", ("lean", "cot")), ["lean", "cot"])
        with self.assertRaises(SystemExit):
            run._conditions("hinted", ("lean", "cot"))

    def test_dry_run_accepts_alias(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out = quiet(run.main, ["--problem", str(EXAMPLE), "--conditions", "unguided-direct", "--budget", "1",
                                       "--dry-run", "--results", d])
            self.assertEqual(rc, 0)
            self.assertIn(" direct ", out)
            self.assertTrue((Path(d) / "dry_run" / "lean_main.json").exists())   # stored key in file names


class StudyDataMessages(unittest.TestCase):
    def test_commands_name_the_study_repository(self):
        from ahp_bench import tech_report
        with tempfile.TemporaryDirectory() as d:
            for fn, args in ((report.main, ["--results", d]), (audit.main, ["--results", d]),
                             (audit.main, []), (tech_report.main, ["--results", d])):
                with self.assertRaises(SystemExit) as cm:
                    fn(args)
                self.assertIn("llm-ahp-mcdm-study", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
