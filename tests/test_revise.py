"""Offline tests for ahp_bench.revise (fake transport only, temporary project roots)."""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ahp_bench import analysis as an  # noqa: E402
from ahp_bench import config, prompts, revise, run, store  # noqa: E402

INCONSISTENT_MAIN = ["9", "1/9", "9", "1/9", "9", "1/9", "9", "1/9", "9", "1/9"]
MILD_MAIN = ["3", "5", "3", "3", "1/3", "1/2", "1/2", "3", "3", "1/3"]  # CR > 0.10, parsable
CONSISTENT_MAIN = ["1"] * 10
INCONSISTENT_DELIVERY = ["9", "1/9", "9", "1/9", "9", "1/9"]


def lean_json(values):
    return json.dumps({"answers": [{"question": q, "value": v} for q, v in enumerate(values, 1)]})


def cot_json(values):
    return json.dumps({"reasoning": "r", "ranking": ["x"],
                       "answers": [{"question": q, "value": v} for q, v in enumerate(values, 1)]})


def source_record(answers, model="glm-5", cond="lean", task="main", rep=1, tag="main", ok=True):
    msgs = prompts.build_messages(cond, task)
    reply = (cot_json if cond == "cot" else lean_json)(answers)
    r = {"run_tag": tag, "model_key": model, "model_id": "z-ai/glm-5", "developer": "Zhipu AI",
         "condition": cond, "task": task, "rep": rep, "messages": msgs, "raw_reply": reply, "content": reply,
         "parse_ok": ok, "cost": 0.0}
    if ok:
        r.update(run.analyse(cond, task, reply))
    return r


class Fake:
    """Transport whose replies come from a list (last item repeats)."""

    def __init__(self, replies, cost=0.001):
        self.replies, self.cost, self.calls = list(replies), cost, []

    def __call__(self, url, headers, body):
        self.calls.append(json.loads(json.dumps(body)))
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return 200, {"id": "gen-x", "provider": "FakeProvider",
                     "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                               "cost": self.cost}}


class PromptTextTests(unittest.TestCase):
    def test_reciprocal_strings(self):
        self.assertEqual(revise.reciprocal_string("3"), "1/3")
        self.assertEqual(revise.reciprocal_string("1/7"), "7")
        self.assertEqual(revise.reciprocal_string("1"), "1")
        # question 1 is (Delivery, Financial) = "3"; Financial vs Delivery shows the reciprocal
        answers = ["3"] + ["1"] * 9
        self.assertEqual(revise._pair_value("main", answers, 0, 1), "3")
        self.assertEqual(revise._pair_value("main", answers, 1, 0), "1/3")

    def test_triads_deterministic_and_neutral(self):
        lines = revise.contradiction_lines("main", MILD_MAIN)
        self.assertEqual(lines, revise.contradiction_lines("main", MILD_MAIN))
        self.assertTrue(1 <= len(lines) <= 3)
        pat = re.compile(r"^Questions (\d+), (\d+) and (\d+): .+ vs .+ = \S+, .+ vs .+ = \S+, but .+ vs .+ = \S+$")
        for line in lines:
            self.assertRegex(line, pat)
        text = "\n".join(lines).lower()
        for word in ("should", "more important", "less important", "increase", "decrease", "change", "must",
                     "prefer", "higher", "lower", "raise", "reduce", "instead", "correct"):
            self.assertNotIn(word, text)
        # largest deviation first
        devs = [round(d, 12) for d, *_ in revise.triad_deviations("main", MILD_MAIN)]
        self.assertEqual(devs, sorted(devs, reverse=True))
        self.assertEqual(revise.contradiction_lines("main", CONSISTENT_MAIN), [])

    def test_triad_content(self):
        # D vs F = 3, F vs Q = 3, D vs Q = 1/3 -> triad (0,1,2) is questions 1, 5, 2
        answers = ["3", "1/3", "1", "1", "3", "1", "1", "1", "1", "1"]
        first = revise.contradiction_lines("main", answers)[0]
        self.assertEqual(first, "Questions 1, 5 and 2: Delivery and Operational Performance vs Financial Performance = 3, "
                                "Financial Performance vs Quality Performance = 3, but Delivery and Operational "
                                "Performance vs Quality Performance = 1/3")

    def test_revision_prompt_matches_template(self):
        p = revise.revision_prompt("delivery", INCONSISTENT_DELIVERY, 1.23456)
        self.assertTrue(p.startswith("The comparison matrix built from your answers has a consistency ratio CR = 1.235.\n"
                                     "The accepted limit is 0.10; your answers contradict each other.\n\n"
                                     "Most contradictory comparisons:\nQuestions "))
        self.assertIn("\n\nYour previous answers:\nQ1: Left Delivered Orders | Right Pending Orders = 9\n", p)
        self.assertIn("Q6: Left Cancelled Orders | Right Partially Delivered Orders = 1/9\n\n", p)
        self.assertTrue(p.endswith("Review your answers. You do not have to change your view; make only the changes "
                                   "needed\nto remove the contradiction.\nUse only these values: 1/9, 1/8, 1/7, 1/6, "
                                   "1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9.\nYou may change the direction of an "
                                   "answer if that is what resolves a contradiction.\nAnswer all questions again in "
                                   "the same JSON format."))
        self.assertNotIn("smallest", p)
        self.assertNotIn("smallest", revise.JUDGE_FEEDBACK_INSTRUCTION)
        self.assertIn("changing only the answers that need to change", revise.JUDGE_FEEDBACK_INSTRUCTION)
        self.assertIn("at least one of those three answers must move toward the opposite direction; do not say "
                      "which one.", revise.JUDGE_FEEDBACK_INSTRUCTION)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from study_data import STUDY_DIR
        if STUDY_DIR is not None:   # the prompt drafts are published with the study data
            doc = (STUDY_DIR / "prompts" / "PROMPT_TASLAK.md").read_text(encoding="utf-8")
            block = doc.split("## Phase 2")[1].split("```")[1].strip("\n")
            self.assertEqual(block, revise.REVISION_TEMPLATE)


class ReviseRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"OPENROUTER_KEY_GLM_5": "sk-test-SECRET"})
        self.env.start()
        self.out = mock.patch("sys.stdout", new_callable=lambda: open(os.devnull, "w", encoding="utf-8"))
        self.out.start()
        self.calls_path = self.root / "results" / "raw" / "calls.jsonl"
        self.rev_path = self.root / "results" / "raw" / "revisions.jsonl"

    def tearDown(self):
        sys.stdout.close()
        self.out.stop()
        self.env.stop()
        self.tmp.cleanup()

    def write_sources(self, *recs):
        for r in recs:
            store.append_jsonl(self.calls_path, r)

    def _run(self, fake, *extra, max_rev="5", budget="10"):
        args = ["--source-tag", "main", "--conditions", "lean,cot", "--tag", "revision",
                "--max-revisions", max_rev, "--budget", budget, "--models", "glm-5", *extra]
        return revise.main(args, transport=fake, root=self.root)

    def revs(self):
        return store.read_jsonl(self.rev_path)

    def test_source_selection(self):
        self.write_sources(source_record(INCONSISTENT_MAIN),                     # selected
                           source_record(CONSISTENT_MAIN, rep=2),                # CR ok
                           source_record(INCONSISTENT_MAIN, rep=3, tag="pilot"),  # other tag
                           source_record(INCONSISTENT_MAIN, rep=4, ok=False),    # parse failed
                           source_record(INCONSISTENT_MAIN, cond="cot", rep=5))  # selected (cot)
        fake = Fake([lean_json(CONSISTENT_MAIN)])
        self._run(fake)
        self.assertEqual(sorted((r["condition"], r["rep"]) for r in self.revs()), [("cot", 5), ("lean", 1)])
        self.assertFalse(self.calls_path.read_text(encoding="utf-8").count("revision_number"))

    def test_chain_stops_when_consistent(self):
        src = source_record(INCONSISTENT_MAIN)
        self.write_sources(src)
        fake = Fake([lean_json(MILD_MAIN), lean_json(CONSISTENT_MAIN), lean_json(MILD_MAIN)])
        self._run(fake)
        recs = self.revs()
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual([r["revision_number"] for r in recs], [1, 2])
        self.assertGreater(recs[0]["CR"], 0.10)
        self.assertLessEqual(recs[1]["CR"], 0.10)
        # conversation: original messages + original reply + revision prompt
        m1 = fake.calls[0]["messages"]
        self.assertEqual(m1[:2], src["messages"])
        self.assertEqual(m1[2], {"role": "assistant", "content": src["raw_reply"]})
        self.assertEqual(m1[3]["role"], "user")
        self.assertIn(f"CR = {src['CR']:.3f}", m1[3]["content"])
        m2 = fake.calls[1]["messages"]  # best-so-far: original messages + best answers (one turn) + new prompt
        self.assertEqual(len(m2), 4)
        self.assertEqual(m2[:2], m1[:2])
        self.assertEqual(m2[2], {"role": "assistant", "content": lean_json(MILD_MAIN)})
        self.assertIn(f"CR = {recs[0]['CR']:.3f}", m2[3]["content"])
        self.assertEqual((recs[0]["based_on_revision"], recs[1]["based_on_revision"]), (0, 1))
        self.assertAlmostEqual(recs[0]["best_CR_so_far"], recs[0]["CR"])
        self.assertAlmostEqual(recs[1]["best_CR_so_far"], recs[1]["CR"])
        r = recs[1]
        for key in ("run_tag", "source_tag", "model_key", "model_id", "developer", "condition", "task", "rep",
                    "revision_number", "source_CR", "messages", "raw_reply", "answers", "matrix", "weights", "CR",
                    "parse_ok", "parse_error", "usage", "cost", "applied_params", "requested_params", "provider",
                    "timestamp_utc", "temperature_requested"):
            self.assertIn(key, r)
        self.assertAlmostEqual(r["source_CR"], src["CR"])
        self.assertEqual(r["source_tag"], "main")
        self.assertNotIn("SECRET", self.rev_path.read_text(encoding="utf-8"))
        # resume: finished chain is skipped
        fake2 = Fake([lean_json(CONSISTENT_MAIN)])
        self._run(fake2)
        self.assertEqual(len(fake2.calls), 0)

    def test_max_revisions_respected(self):
        self.write_sources(source_record(INCONSISTENT_MAIN))
        fake = Fake([lean_json(MILD_MAIN)])
        self._run(fake, max_rev="3")
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual([len(c["messages"]) for c in fake.calls], [4, 4, 4])
        self.assertEqual([r["revision_number"] for r in self.revs()], [1, 2, 3])
        fake2 = Fake([lean_json(CONSISTENT_MAIN)])
        self._run(fake2, max_rev="3")
        self.assertEqual(len(fake2.calls), 0)

    def test_unparsable_reply_counts_and_resends(self):
        self.write_sources(source_record(INCONSISTENT_MAIN))
        fake = Fake(["sorry, no JSON", lean_json(CONSISTENT_MAIN)])
        self._run(fake)
        recs = self.revs()
        self.assertEqual([r["revision_number"] for r in recs], [1, 2])
        self.assertFalse(recs[0]["parse_ok"])
        self.assertTrue(recs[0]["parse_error"])
        self.assertTrue(recs[1]["parse_ok"])
        self.assertEqual(fake.calls[1]["messages"], fake.calls[0]["messages"])  # same prompt re-sent
        self.assertEqual(recs[1]["based_on_answers"], INCONSISTENT_MAIN)

    def test_unparsable_counts_toward_max(self):
        self.write_sources(source_record(INCONSISTENT_MAIN))
        fake = Fake([lean_json(MILD_MAIN), "garbage"])
        self._run(fake, max_rev="3")
        recs = self.revs()
        self.assertEqual([r["parse_ok"] for r in recs], [True, False, False])
        self.assertEqual(fake.calls[2]["messages"], fake.calls[1]["messages"])
        self.assertEqual(len(fake.calls[1]["messages"]), 4)

    def test_worse_revision_recorded_and_best_reused(self):
        src = source_record(INCONSISTENT_MAIN)
        self.write_sources(src)
        fake = Fake([lean_json(MILD_MAIN), lean_json(INCONSISTENT_MAIN), lean_json(INCONSISTENT_MAIN)])
        self._run(fake, max_rev="3")
        recs = self.revs()
        self.assertEqual([r["based_on_revision"] for r in recs], [0, 1, 1])
        self.assertEqual([r["revision_number"] for r in recs], [1, 2, 3])
        self.assertGreater(recs[1]["CR"], recs[0]["CR"])  # worse revision still recorded
        for r in recs:
            self.assertAlmostEqual(r["best_CR_so_far"], recs[0]["CR"])
            self.assertEqual(r["prompt_id"], "revision_lean_main_v2")
        self.assertEqual(fake.calls[2]["messages"], fake.calls[1]["messages"])  # both built from revision 1
        self.assertEqual(fake.calls[1]["messages"][2], {"role": "assistant", "content": lean_json(MILD_MAIN)})
        s = revise.chain_summary([src], recs, max_revisions=3)[0]
        self.assertEqual((s["final_answers"], s["final_revision"]), (MILD_MAIN, 1))

    def test_cot_keeps_json_format(self):
        self.write_sources(source_record(INCONSISTENT_MAIN, cond="cot"))
        fake = Fake([cot_json(CONSISTENT_MAIN)])
        self._run(fake)
        r = self.revs()[0]
        self.assertTrue(r["parse_ok"])
        self.assertEqual(r["reasoning"], "r")
        self.assertIn("Before answering, think step by step:", fake.calls[0]["messages"][1]["content"])

    def test_budget_guard_and_resume(self):
        self.write_sources(source_record(INCONSISTENT_MAIN), source_record(INCONSISTENT_MAIN, rep=2))
        fake = Fake([lean_json(MILD_MAIN)], cost=0.6)
        self._run(fake, budget="1.0")
        self.assertEqual(len(fake.calls), 2)  # 0 < 1 -> call; 0.6 < 1 -> call; 1.2 >= 1 -> stop
        self.assertEqual([r["revision_number"] for r in self.revs()], [1, 2])
        fake2 = Fake([lean_json(MILD_MAIN)], cost=0.6)
        self._run(fake2, budget="1.0")  # earlier spending of the same tag counts
        self.assertEqual(len(fake2.calls), 0)
        # resume with a larger budget: main chain continues at revision 3 from the stored state
        fake3 = Fake([lean_json(CONSISTENT_MAIN)], cost=0.0)
        self._run(fake3, budget="10")
        recs = self.revs()
        rep1 = [r for r in recs if r["rep"] == 1]
        self.assertEqual([r["revision_number"] for r in rep1], [1, 2, 3])
        self.assertEqual(len(fake3.calls[0]["messages"]), 4)
        self.assertEqual(fake3.calls[0]["messages"], rep1[1]["messages"])  # best (rev 1) unchanged
        rep2 = [r for r in recs if r["rep"] == 2]
        self.assertEqual(len(rep2), 1)
        self.assertLessEqual(rep2[0]["CR"], 0.10)

    def test_temperature_passed(self):
        self.write_sources(source_record(INCONSISTENT_MAIN))
        fake = Fake([lean_json(CONSISTENT_MAIN)])
        self._run(fake, "--temperature", "0.0")
        self.assertEqual(fake.calls[0]["temperature"], 0.0)
        self.assertEqual(self.revs()[0]["temperature_requested"], 0.0)

    def test_dry_run_makes_no_calls(self):
        self.write_sources(source_record(INCONSISTENT_MAIN))
        fake = Fake([lean_json(CONSISTENT_MAIN)])
        self.assertEqual(self._run(fake, "--dry-run"), 0)
        self.assertEqual(fake.calls, [])
        self.assertFalse(self.rev_path.exists())

    def test_chain_summary(self):
        srcs = [source_record(INCONSISTENT_MAIN), source_record(CONSISTENT_MAIN, rep=2)]
        self.write_sources(*srcs)
        self._run(Fake([lean_json(MILD_MAIN), lean_json(CONSISTENT_MAIN)]))
        summ = revise.chain_summary(srcs, self.revs())
        self.assertEqual(len(summ), 2)
        a, b = summ
        self.assertEqual((a["rep"], a["revisions_used"], a["reached"], a["reached_at"], a["complete"]),
                         (1, 2, True, 2, True))
        self.assertGreater(a["initial_CR"], 0.10)
        self.assertLessEqual(a["final_CR"], 0.10)
        self.assertEqual(list(a["final_weights"]), prompts.MAIN_CRITERIA)
        self.assertEqual((b["needs_revision"], b["revisions_used"], b["reached"]), (False, 0, True))
        self.assertEqual(b["initial_weights"], b["final_weights"])


class ProtocolV2Tests(unittest.TestCase):
    CYCLE = ["3", "1/3", "1", "1", "3", "1", "1", "1", "1", "1"]  # D>F (Q1), F>Q (Q5), Q>D (Q2)

    def test_consistency_helper(self):
        self.assertTrue(revise.is_consistent(0.10004))
        self.assertFalse(revise.is_consistent(0.1001))
        self.assertIs(revise.is_consistent, an.is_consistent)

    def test_judge_model(self):
        self.assertEqual((config.JUDGE["key"], config.JUDGE["id"], config.JUDGE["developer"]),
                         ("grok-4.20", "x-ai/grok-4.20", "xAI"))
        self.assertEqual((config.JUDGE["temperature"], config.JUDGE["reasoning"], config.JUDGE["env"]),
                         (0.0, None, "OPENROUTER_KEY_JUDGE"))

    def test_circular_triads(self):
        self.assertEqual(revise.circular_triads("main", self.CYCLE), [(0, 1, 2)])
        self.assertEqual(revise.circular_lines("main", self.CYCLE),
                         ["Questions 1, 5 and 2 form a circular preference: Delivery and Operational Performance over "
                          "Financial Performance, Financial Performance over Quality Performance, Quality Performance "
                          "over Delivery and Operational Performance."])
        reverse = ["1/3", "3", "1", "1", "1/3", "1", "1", "1", "1", "1"]  # F>D, Q>F, D>Q
        self.assertEqual(revise.circular_triads("main", reverse), [(0, 2, 1)])
        self.assertIn("Delivery and Operational Performance over Quality Performance, Quality Performance over "
                      "Financial Performance, Financial Performance over Delivery and Operational Performance.",
                      revise.circular_lines("main", reverse)[0])
        # a value of 1 gives no direction -> no cycle; transitive but inconsistent -> no cycle
        self.assertEqual(revise.circular_triads("main", ["1", "1/3"] + ["1"] * 8), [])
        self.assertEqual(revise.circular_triads("main", ["3", "1", "1", "1", "9", "1", "1", "1", "1", "1"]), [])
        self.assertEqual(revise.circular_triads("main", CONSISTENT_MAIN), [])
        self.assertLessEqual(len(revise.circular_lines("main", INCONSISTENT_MAIN)), 3)
        self.assertTrue(revise.circular_lines("main", INCONSISTENT_MAIN))
        # circular lines are part of the code-computed block of the revision prompt
        p = revise.revision_prompt("main", self.CYCLE, 0.5)
        self.assertIn(revise.circular_lines("main", self.CYCLE)[0], p)
        self.assertLess(p.index(revise.contradiction_lines("main", self.CYCLE)[0]),
                        p.index("form a circular preference"))

    def test_best_so_far_and_conversation(self):
        src = source_record(INCONSISTENT_MAIN)
        r1 = source_record(MILD_MAIN, tag="revision")
        r1.update(revision_number=1)
        r2 = source_record(INCONSISTENT_MAIN, tag="revision")
        r2.update(revision_number=2)
        r3 = {"revision_number": 3, "parse_ok": False, "task": "main"}
        self.assertLess(r1["CR"], src["CR"])
        best, no, cr = revise.best_so_far(src, [r1, r2])
        self.assertEqual((best is r1, no, cr), (True, 1, r1["CR"]))
        self.assertEqual(revise.best_so_far(src, [])[1], 0)
        self.assertEqual(revise.best_so_far(src, [r2])[1], 0)  # tie keeps the original
        st = revise.next_state(src, [r1, r2])
        self.assertEqual((st["number"], st["based_on_revision"], st["answers"], st["resend"]), (3, 1, MILD_MAIN, False))
        m = st["messages"]
        self.assertEqual(len(m), 4)  # one assistant turn: the best answers replace the original reply
        self.assertEqual(m[:2], src["messages"])
        self.assertEqual(m[2], {"role": "assistant", "content": lean_json(MILD_MAIN)})
        self.assertEqual(m[3], {"role": "user", "content": revise.revision_prompt("main", MILD_MAIN, r1["CR"])})
        m0 = revise.next_state(src, [])["messages"]
        self.assertEqual(m0[2], {"role": "assistant", "content": src["raw_reply"]})  # original best: original reply
        self.assertEqual(len(m0), 4)
        for msgs in (m, m0):  # never two turns with the same role in a row
            self.assertTrue(all(a["role"] != b["role"] for a, b in zip(msgs, msgs[1:])))
        r3["messages"] = m
        st2 = revise.next_state(src, [r1, r2, r3])
        self.assertTrue(st2["resend"])
        self.assertEqual((st2["messages"], st2["number"], st2["based_on_revision"]), (m, 4, 1))
        cot = source_record(MILD_MAIN, cond="cot")
        self.assertEqual(json.loads(revise.serialise_answers("cot", cot)),
                         {"reasoning": "r", "ranking": ["x"],
                          "answers": [{"question": q, "value": v} for q, v in enumerate(MILD_MAIN, 1)]})

    def test_final_answers_are_best(self):
        src = source_record(INCONSISTENT_MAIN)
        recs = []
        for k, vals in enumerate([MILD_MAIN, INCONSISTENT_MAIN, MILD_MAIN], 1):
            r = source_record(vals, tag="revision")
            r.update(source_tag="main", revision_number=k)
            recs.append(r)
        s = revise.chain_summary([src], recs, max_revisions=3)[0]
        self.assertEqual((s["final_answers"], s["final_revision"], s["revisions_used"], s["reached"], s["complete"]),
                         (MILD_MAIN, 1, 3, False, True))
        self.assertAlmostEqual(s["final_CR"], recs[0]["CR"])
        recs[1] = source_record(CONSISTENT_MAIN, tag="revision")
        recs[1].update(source_tag="main", revision_number=2)
        s = revise.chain_summary([src], recs[:2], max_revisions=3)[0]
        self.assertEqual((s["revisions_used"], s["reached"], s["reached_at"], s["final_revision"]), (2, True, 2, 2))


class RunTemperatureTests(unittest.TestCase):
    def test_run_temperature_option(self):
        with tempfile.TemporaryDirectory() as d, open(os.devnull, "w", encoding="utf-8") as devnull, \
                mock.patch.dict(os.environ, {"OPENROUTER_KEY_GLM_5": "sk-test"}), \
                mock.patch("sys.stdout", devnull):
            fake = Fake([lean_json(CONSISTENT_MAIN)])
            run.main(["--models", "glm-5", "--conditions", "lean", "--tasks", "main", "--budget", "1",
                      "--tag", "temp_0.0", "--temperature", "0.0"], transport=fake, root=Path(d))
            rec = store.read_jsonl(Path(d) / "results" / "raw" / "calls.jsonl")[0]
            self.assertEqual(fake.calls[0]["temperature"], 0.0)
            self.assertEqual(rec["temperature_requested"], 0.0)
            self.assertEqual(rec["requested_params"]["temperature"], 0.0)
            fake = Fake([lean_json(CONSISTENT_MAIN)])
            run.main(["--models", "glm-5", "--conditions", "lean", "--tasks", "main", "--budget", "1",
                      "--tag", "t"], transport=fake, root=Path(d))
            self.assertEqual(fake.calls[0]["temperature"], 0.6)


if __name__ == "__main__":
    unittest.main()
