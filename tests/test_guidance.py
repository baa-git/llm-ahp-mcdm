"""Guidance levels for problems other than the study: importance-cue validator, prompt construction per level,
problem-file blocks, host and API (dry run and fake transport) paths, records and the compliance table G13."""
import contextlib
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

from ahp_bench import guidance, host, prompts, report, revise, run, store  # noqa: E402
from ahp_bench import problem as pb  # noqa: E402

QUESTION = re.compile(r"^\d+\.\s+Left: (.+) \| Right: (.+)$", re.M)
PREVIOUS = re.compile(r"^Q\d+: Left .+ \| Right .+ = \S+$", re.M)


def cli(fn, argv, **kw):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = fn(argv, **kw)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 2
            err.write(str(e.code))
    return rc, out.getvalue(), err.getvalue()


def example_dict():
    return json.loads(pb.EXAMPLE_PROBLEM_PATH.read_text(encoding="utf-8"))


NEUTRAL = {"Performance": "Groups processor, memory and storage speed.",
           "Price": "Purchase price in USD; lower values are better.",
           "Portability": "Weight in kg; lower values are better.",
           "Processor Speed": "Processor benchmark score in points; higher values are better.",
           "Memory Capacity": "Memory in GB; higher values are better.",
           "Storage Speed": "SSD read speed in MB/s; higher values are better."}


def with_blocks(descriptions=True, guidance_block=True, source="user", confirmed=True, texts=None):
    d = example_dict()
    d["name"] = "laptop_guided"
    if descriptions:
        d["descriptions"] = {"drafted_by": "user", "confirmed_by_user": True, "criteria": dict(texts or NEUTRAL)}
    if guidance_block:
        d["guidance"] = {"source": source, "confirmed_by_user": confirmed, "user_statement": "Price first, then speed.",
                         "order": {"main": ["Price", "Performance", "Portability"]},
                         "hints": {"performance": ["Memory matters for my statistics courses."]}}
    return d


def preference_reply(user_text, order, cot=False):
    """Answers that follow a given order exactly (consistent): 3 per rank step."""
    rank = {c: k for k, c in enumerate(order)}
    ans = []
    for q, (left, right) in enumerate(QUESTION.findall(user_text), 1):
        diff = rank[right.strip()] - rank[left.strip()]
        ans.append({"question": q, "value": str(1 + 2 * diff) if diff > 0 else f"1/{1 - 2 * diff}" if diff < 0 else "1"})
    obj = {"answers": ans}
    if cot:
        obj.update(reasoning="r", ranking=list(order))
    return json.dumps(obj)


class CueValidatorTests(unittest.TestCase):
    def test_cues(self):
        self.assertEqual(guidance.importance_cues("Purchase price in USD; lower values are better."), [])
        self.assertEqual(guidance.importance_cues("A critical and KEY factor, the most important one."),
                         ["critical", "key", "most important"])
        for text in ("Price is more important than weight", "top priority", "essential for buyers", "vital",
                     "ranked first", "a major driver", "çok kritik", "orta önemde"):
            self.assertTrue(guidance.importance_cues(text), text)
        self.assertEqual(guidance.importance_cues("keyboard layout; topology; monkey"), [])

    def test_code_drafts_are_neutral(self):
        p = pb.load_problem(pb.EXAMPLE_PROBLEM_PATH)
        drafts = guidance.code_drafts(p)
        self.assertEqual(list(drafts), p.all_criteria)
        self.assertEqual(drafts["Price"], "Measured by Price (USD); lower values are better (indicator PR).")
        self.assertTrue(all(not guidance.importance_cues(t) for t in drafts.values()))


class ProblemBlockTests(unittest.TestCase):
    def test_conditions_per_level(self):
        self.assertEqual(pb.problem_from_dict(example_dict()).conditions, ("lean", "cot"))
        p = pb.problem_from_dict(with_blocks(guidance_block=False))
        self.assertEqual(p.conditions, ("lean", "cot", "described", "described_cot"))
        p = pb.problem_from_dict(with_blocks())
        self.assertEqual(p.conditions[-2:], ("user_guided", "user_guided_cot"))
        self.assertEqual(p.default_conditions, ("lean", "cot"))
        study = pb.load_problem(pb.STUDY_PROBLEM_PATH)
        self.assertEqual(study.conditions, ("lean", "cot", "guided"))
        self.assertEqual(study.default_conditions, ("lean", "cot", "guided"))

    def test_validation_errors(self):
        d = with_blocks(guidance_block=False)
        del d["descriptions"]["criteria"]["Price"]
        with self.assertRaisesRegex(pb.ProblemError, "missing \\['Price'\\]"):
            pb.problem_from_dict(d)
        with self.assertRaisesRegex(pb.ProblemError, "guidance needs neutral criterion descriptions"):
            pb.problem_from_dict(with_blocks(descriptions=False))
        d = with_blocks()
        d["guidance"]["order"]["main"] = ["Price", "Performance"]
        with self.assertRaisesRegex(pb.ProblemError, "every criterion of the task exactly once"):
            pb.problem_from_dict(d)
        d = with_blocks(source="agent")
        with self.assertRaisesRegex(pb.ProblemError, "guidance.source"):
            pb.problem_from_dict(d)
        d = json.loads(pb.STUDY_PROBLEM_PATH.read_text(encoding="utf-8"))
        d["descriptions"] = {"drafted_by": "x", "criteria": {}}
        with self.assertRaisesRegex(pb.ProblemError, "verbatim guided prompts"):
            pb.problem_from_dict(d)


class PromptTests(unittest.TestCase):
    def test_levels(self):
        plain = pb.problem_from_dict(example_dict())
        p = pb.problem_from_dict(with_blocks())
        lean = prompts.build_messages("lean", "main", p)
        self.assertEqual(lean, prompts.build_messages("lean", "main", plain))   # descriptions never leak into direct
        self.assertEqual(prompts.build_messages("cot", "main", p), prompts.build_messages("cot", "main", plain))
        desc = prompts.build_messages("described", "main", p)[1]["content"]
        self.assertIn("The criteria:\n- Performance: Groups processor", desc)
        self.assertNotIn("Priorities stated", desc)
        self.assertIn('{"answers": [{"question": 1', desc)
        self.assertEqual(desc.replace(prompts.criteria_block("main", p) + "\n\n", ""), lean[1]["content"])
        dcot = prompts.build_messages("described_cot", "performance", p)[1]["content"]
        self.assertIn("The sub-criteria:\n- Processor Speed", dcot)
        self.assertIn("think step by step", dcot)
        g = prompts.build_messages("user_guided", "main", p)[1]["content"]
        self.assertIn("Priorities stated by the decision maker:\nOrder of importance, most important first: "
                      "1. Price, 2. Performance, 3. Portability", g)
        self.assertLess(g.index("The criteria:"), g.index("Priorities stated"))
        gp = prompts.build_messages("user_guided_cot", "performance", p)[1]["content"]
        self.assertIn("- Memory matters for my statistics courses.", gp)
        self.assertNotIn("Order of importance", gp)
        with self.assertRaises(ValueError):
            prompts.build_messages("described", "main", plain)
        with self.assertRaises(ValueError):
            prompts.build_messages("described", "main", None)  # the study
        self.assertEqual(prompts.prompt_id("user_guided_cot", "main", p), "laptop_guided:user_guided_cot_main_v2")
        f = prompts.record_fields("user_guided_cot", p)
        self.assertEqual((f["guidance_level"], f["technique"], f["guidance_source"]), ("guided", "CoT", "user"))
        self.assertEqual(prompts.record_fields("lean", None), {})
        self.assertEqual(prompts.record_fields("lean", p), {"guidance_level": "unguided", "technique": "zero-shot"})
        self.assertEqual(prompts.condition_guidance("guided CoT"), "guided")
        self.assertEqual(prompts.condition_guidance("guided"), "guided")
        self.assertEqual(prompts.condition_technique("described CoT"), "CoT")
        self.assertEqual(run._conditions("direct,guided,guided-cot,described-cot", p.conditions),
                         ["lean", "user_guided", "user_guided_cot", "described_cot"])
        study = pb.load_problem(pb.STUDY_PROBLEM_PATH)
        self.assertEqual(run._conditions("guided", study.conditions), ["guided"])


class TmpProblem(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def problem_file(self, data):
        path = self.d / "p.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path


class CommandTests(TmpProblem):
    def test_guidance_commands(self):
        d = example_dict()
        d["name"] = "laptop_cmd"
        path = self.problem_file(d)
        rc, out, _ = cli(guidance.main, ["set-order", "--problem", str(path), "--order", "Price | Performance | Portability",
                                         "--source", "user"])
        self.assertEqual(rc, 2)     # descriptions first
        self.assertEqual(cli(guidance.main, ["draft-descriptions", "--problem", str(path)])[0], 0)
        p = pb.load_problem(path)
        self.assertEqual(p.descriptions["drafted_by"], "ahp-bench (from the indicator table)")
        self.assertFalse(p.descriptions["confirmed_by_user"])
        self.assertEqual(cli(guidance.main, ["draft-descriptions", "--problem", str(path)])[0], 2)
        rc, out, err = cli(guidance.main, ["set-order", "--problem", str(path), "--order", "2,1,3", "--source",
                                           "orchestrator-draft"])
        self.assertEqual(rc, 0)
        self.assertEqual(pb.load_problem(path).guidance["order"]["main"], ["Price", "Performance", "Portability"])
        self.assertIn("orchestrator-draft", err)
        rc, _, err = cli(guidance.main, ["set-order", "--problem", str(path), "--order", "Price | Weight | Performance",
                                         "--source", "user"])
        self.assertEqual(rc, 2)
        self.assertIn("unknown criterion 'Weight'", err)
        rc, _, err = cli(guidance.main, ["check", "--problem", str(path)])
        self.assertEqual(rc, 0)
        self.assertIn("GUIDANCE NOT CONFIRMED", err)
        self.assertEqual(cli(guidance.main, ["set-order", "--problem", str(path), "--task", "performance", "--hint",
                                             "Memory first for me.", "--source", "user",
                                             "--user-statement", "I need memory."])[0], 0)
        self.assertEqual(cli(guidance.main, ["confirm", "--problem", str(path), "--guidance"])[0], 0)
        self.assertEqual(cli(guidance.main, ["confirm", "--problem", str(path), "--descriptions"])[0], 0)
        g = pb.load_problem(path).guidance
        self.assertEqual((g["source"], g["confirmed_by_user"], g["user_statement"]), ("user", True, "I need memory."))
        self.assertEqual(cli(guidance.main, ["check", "--problem", str(path)])[2], "")
        self.assertEqual(cli(guidance.main, ["confirm", "--problem", str(path), "--criteria"])[0], 2)  # no provenance
        self.assertEqual(cli(guidance.main, ["check", "--problem", str(pb.STUDY_PROBLEM_PATH)])[0], 2)

    def test_host_prompts_refuse_cues_unless_allowed(self):
        texts = dict(NEUTRAL, Price="The most important cost item; critical for students.")
        path = self.problem_file(with_blocks(texts=texts))
        w = self.d / "w"
        rc, _, err = cli(host.main, ["prompts", "--problem", str(path), "--conditions", "described", "--panelists", "1",
                                     "--out", str(w)])
        self.assertEqual(rc, 2)
        self.assertIn("importance cues", err)
        self.assertIn("Price: critical, most important", err)
        self.assertFalse(w.exists())
        rc, _, err = cli(host.main, ["prompts", "--problem", str(path), "--conditions", "guided", "--panelists", "1",
                                     "--out", str(w)])
        self.assertEqual(rc, 0)     # guided level: not checked
        rc, out, err = cli(host.main, ["prompts", "--problem", str(path), "--conditions", "described,described-cot",
                                       "--panelists", "1", "--out", str(w), "--allow-cues"])
        self.assertEqual(rc, 0, err)
        p = pb.load_problem(path)
        for f in (w / "prompts").glob("described*_p01_r01.txt"):
            text = f.read_text(encoding="utf-8").split("=== USER ===\n", 1)[1]
            (w / "replies" / f"{f.stem}.json").write_text(
                preference_reply(text, p.criteria("main" if "_main_" in f.name else "performance"), "_cot_" in f.name),
                encoding="utf-8")
        self.assertEqual(sorted(x.name for x in (w / "replies").glob("*.json")),
                         ["described_cot_main_p01_r01.json", "described_cot_performance_p01_r01.json",
                          "described_main_p01_r01.json", "described_performance_p01_r01.json"])
        rc, out, _ = cli(host.main, ["ingest", "--problem", str(path), "--dir", str(w), "--results", str(self.d / "r"),
                                     "--model-name", "canned"])
        self.assertIn("4 records written (4 parsed)", out)
        recs = store.read_jsonl(self.d / "r" / "raw" / "calls.jsonl")
        self.assertTrue(all(r["cues_allowed"] is True and r["guidance_level"] == "described" for r in recs))
        self.assertEqual({r["technique"] for r in recs}, {"zero-shot", "CoT"})
        self.assertTrue(all(r["reasoning"] == "r" for r in recs if r["condition"] == "described_cot"))

    def test_descriptions_host_and_api(self):
        d = example_dict()
        d["name"] = "laptop_desc"
        path = self.problem_file(d)
        w = self.d / "dw"
        self.assertEqual(cli(guidance.main, ["descriptions-prompt", "--problem", str(path), "--out", str(w)])[0], 0)
        prompt = (w / "descriptions_prompt.txt").read_text(encoding="utf-8")
        self.assertIn("Do not signal importance or rank", prompt)
        self.assertIn("- Price: indicator PR: Price; unit USD; lower values are better", prompt)
        reply = dict(NEUTRAL, Portability="A key factor: weight in kg.")
        (w / "descriptions_reply.json").write_text(json.dumps({"descriptions": reply}), encoding="utf-8")
        rc, out, err = cli(guidance.main, ["descriptions-ingest", "--problem", str(path), "--dir", str(w)])
        self.assertEqual(rc, 2)      # stored, but the cue is flagged
        self.assertIn("Portability: key", err)
        self.assertEqual(pb.load_problem(path).descriptions["drafted_by"], "host agent")
        log = store.read_jsonl(w / "descriptions_calls.jsonl")
        self.assertEqual(log[0]["importance_cues"], {"Portability": ["key"]})
        calls = []

        def fake(url, headers, body):
            calls.append(body)
            return 200, {"id": "g", "choices": [{"message": {"content": json.dumps({"descriptions": NEUTRAL})}}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost": 0.0003}}

        base = ["descriptions-propose", "--problem", str(path), "--model", "test/m", "--budget", "0.01", "--force"]
        rc, out, _ = cli(guidance.main, base + ["--dry-run"], transport=fake)
        self.assertEqual((rc, calls), (0, []))
        env = {k: v for k, v in os.environ.items() if not k.startswith("OPENROUTER_")}
        env["OPENROUTER_API_KEY"] = "k-SECRET"
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("ahp_bench.config.load_env", lambda *a, **k: []):
            rc, out, err = cli(guidance.main, base, transport=fake)
        self.assertEqual(rc, 0, err)
        self.assertEqual(pb.load_problem(path).descriptions["drafted_by"], "test/m")
        self.assertNotIn("SECRET", (self.d / "descriptions_calls.jsonl").read_text(encoding="utf-8"))


class Fake:
    """Follows the prompted order under guided conditions, the reverse order otherwise; revisions all 1."""

    def __init__(self, problem):
        self.p, self.calls = problem, []

    def __call__(self, url, headers, body):
        self.calls.append(body)
        msgs = body["messages"]
        last = msgs[-1]["content"]
        if len(msgs) > 2:
            n = len(PREVIOUS.findall(last))
            content = json.dumps({"reasoning": "r", "ranking": [], "answers": [{"question": q, "value": "1"}
                                                                               for q in range(1, n + 1)]})
        else:
            names = [x for pair in QUESTION.findall(last) for x in pair]
            task = "main" if "Price" in names else "performance"
            order = self.p.criteria(task)
            if "Order of importance" in last:
                order = self.p.guidance["order"]["main"]
            elif body["model"] == "test/beta":
                order = list(reversed(order))
            content = preference_reply(last, order, cot="think step by step" in last)
            if body["model"] == "test/alpha" and "think step by step" in last and "Order of importance" not in last:
                obj = json.loads(content)  # alpha: an inconsistent described-CoT matrix to revise
                for a in obj["answers"]:
                    a["value"] = "9" if a["question"] % 2 else "1/9"
                content = json.dumps(obj)
        return 200, {"id": f"g{len(self.calls)}", "provider": "Fake", "choices": [{"message": {"content": content},
                                                                                    "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "cost": 0.0}}


class ApiAndReportTests(TmpProblem):
    def setUp(self):
        super().setUp()
        env = {k: v for k, v in os.environ.items() if not k.startswith("OPENROUTER_")}
        env["OPENROUTER_API_KEY"] = "shared-SECRET"
        self.env = mock.patch.dict(os.environ, env, clear=True)
        self.env.start()
        self.models = self.d / "models.json"
        self.models.write_text(json.dumps({"models": {"alpha": {"id": "test/alpha"}, "beta": {"id": "test/beta"}},
                                           "judge": {"id": "test/judge"}}), encoding="utf-8")

    def tearDown(self):
        self.env.stop()
        super().tearDown()

    def common(self, path, res):
        return ["--problem", str(path), "--results", str(res), "--models-file", str(self.models)]

    def test_dry_run_warns_for_unconfirmed_guidance(self):
        path = self.problem_file(with_blocks(source="orchestrator-draft", confirmed=False))
        fake = Fake(pb.load_problem(path))
        rc, out, err = cli(run.main, self.common(path, self.d / "r") + ["--conditions", "direct,guided", "--budget", "1",
                                                                         "--dry-run", "--print-prompts"],
                           transport=fake, root=self.d)
        self.assertEqual((rc, fake.calls), (0, []))
        self.assertIn("!" * 78 + "\nWARNING: GUIDANCE NOT CONFIRMED", err)
        self.assertIn("===== guided / main =====", out)
        self.assertIn("Order of importance, most important first: 1. Price", out)
        self.assertTrue((self.d / "r" / "dry_run" / "user_guided_main.json").exists())

    def test_run_revise_report_with_compliance(self):
        path = self.problem_file(with_blocks(source="orchestrator-draft", confirmed=False))
        p = pb.load_problem(path)
        fake, res = Fake(p), self.d / "res"
        rc, _, err = cli(run.main, self.common(path, res) + ["--conditions", "direct,described-cot,guided,guided-cot",
                                                             "--reps", "2", "--budget", "1", "--tag", "main"],
                         transport=fake, root=self.d)
        self.assertEqual(rc, 0, err)
        recs = store.read_jsonl(res / "raw" / "calls.jsonl")
        self.assertEqual(len(recs), 2 * 4 * 2 * 2)
        self.assertTrue(all(r["parse_ok"] for r in recs))
        g = [r for r in recs if r["condition"] == "user_guided"]
        self.assertTrue(all(r["guidance_source"] == "orchestrator-draft" and r["guidance_confirmed_by_user"] is False
                            and r["guidance_level"] == "guided" for r in g))
        self.assertTrue(all(len(r["guidance_sha256"]) == 64 for r in g))
        self.assertTrue(all(r["guidance_level"] == "unguided" for r in recs if r["condition"] == "lean"))
        rc, _, err = cli(revise.main, self.common(path, res) + ["--conditions", "described-cot", "--budget", "1"],
                         transport=fake, root=self.d)
        self.assertEqual(rc, 0, err)
        revs = store.read_jsonl(res / "raw" / "revisions.jsonl")
        self.assertEqual(len(revs), 4)    # alpha's described-CoT matrices (2 tasks x 2 reps)
        self.assertTrue(all(r["condition"] == "described_cot" for r in revs))
        rc, _, err = cli(report.main, self.common(path, res) + ["--revision-tag", "revision"])
        self.assertEqual(rc, 0, err)
        tables = res / "report" / "tables"
        with (tables / "G13_guidance_compliance.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        panel = [r for r in rows if r["Respondents"] == "Panel" and r["Comparison"] == "prompted"]
        self.assertEqual({r["Condition"] for r in panel}, {"guided", "guided CoT"})
        for r in panel:
            self.assertEqual((r["Guidance"], r["Replies"], r["Share ranking equals prompted order"],
                              r["Share prompted first criterion ranked first"], r["Mean Spearman rho with prompted order"]),
                             ("guided", "4", "1.0000", "1.0000", "1.000"))
        ref = [r for r in rows if r["Comparison"].startswith("reference")]
        self.assertEqual({r["Condition"] for r in ref}, {"direct", "described CoT"})
        direct = next(r for r in ref if r["Condition"] == "direct")
        self.assertEqual(direct["Share ranking equals prompted order"], "0.0000")
        md = (res / "report" / "report.md").read_text(encoding="utf-8")
        self.assertIn(guidance.GUIDED_NOTE, md)
        self.assertIn("**WARNING: GUIDANCE NOT CONFIRMED", md)
        self.assertIn("## Criterion descriptions shown in the described and guided prompts", md)
        self.assertIn("## G13 Replies and the priority order stated in the guided prompt", md)
        g02 = (tables / "G02_consistency.csv").read_text(encoding="utf-8")
        self.assertIn("Panel,described,CoT,described CoT,main", g02)
        from ahp_bench import audit
        rc, _, _ = cli(audit.main, ["--problem", str(path), "--results", str(res)])
        with (res / "report" / "audit" / "expectations.csv").open(encoding="utf-8") as f:
            exp = list(csv.DictReader(f))
        self.assertEqual([e["check"] for e in exp if e["status"] != "PASS"], [])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
