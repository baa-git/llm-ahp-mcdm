"""Offline tests for the revision judge (revise --judge), client model override and report T20.
Fake transports only; no network."""
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_data import DATA, STUDY_DIR  # noqa: E402
sys.path.insert(0, str(ROOT))

from ahp_bench import client, config, prompts, report, revise, run, store  # noqa: E402

BAD = ["9", "1/9", "9", "1/9", "9", "1/9", "9", "1/9", "9", "1/9"]
MILD = ["3", "5", "3", "3", "1/3", "1/2", "1/2", "3", "3", "1/3"]  # CR > 0.10
OK = ["1"] * 10
OK2 = ["3", "3", "5", "5", "1", "3", "3", "3", "3", "1"]
OK_DEL = ["3", "5", "3", "3", "1", "1/3"]

FEEDBACK = "Your answers to questions 1, 2 and 5 contradict each other. Please reconsider them."
FB_JSON = json.dumps({"feedback": FEEDBACK})


def verdict_json(v="yes"):
    return json.dumps({"logic_preserved": v, "changed_priorities": ["Financial Performance"], "comment": "ok"})


def lean_json(values):
    return json.dumps({"answers": [{"question": q, "value": v} for q, v in enumerate(values, 1)]})


def cot_json(values, ranking=("Quality Performance", "Financial Performance")):
    return json.dumps({"reasoning": "quality first", "ranking": list(ranking),
                       "answers": [{"question": q, "value": v} for q, v in enumerate(values, 1)]})


def source_record(answers, model="glm-5", cond="lean", task="main", rep=1, tag="main"):
    reply = (cot_json if cond == "cot" else lean_json)(answers)
    r = {"run_tag": tag, "model_key": model, "model_id": f"x/{model}", "developer": "Dev", "condition": cond,
         "task": task, "rep": rep, "messages": prompts.build_messages(cond, task), "raw_reply": reply,
         "content": reply, "parse_ok": True, "cost": 0.0}
    r.update(run.analyse(cond, task, reply))
    return r


def _resp(content, cost):
    return 200, {"id": "gen-x", "provider": "FakeProvider",
                 "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": cost}}


class Fake:
    """Routes by model id: judge calls (feedback / verdict, told apart by the system prompt) and panel
    calls each take replies from their own list (last item repeats)."""

    def __init__(self, panel, feedback=(FB_JSON,), verdict=(verdict_json(),), panel_cost=0.001, judge_cost=0.002):
        self.lists = {"panel": list(panel), "feedback": list(feedback), "verdict": list(verdict)}
        self.costs = {"panel": panel_cost, "feedback": judge_cost, "verdict": judge_cost}
        self.calls = {"panel": [], "feedback": [], "verdict": []}
        self.order, self.headers = [], []

    def __call__(self, url, headers, body):
        body = json.loads(json.dumps(body))
        if body["model"] == config.JUDGE["id"]:
            kind = "feedback" if body["messages"][0]["content"] == revise.JUDGE_FEEDBACK_SYSTEM else "verdict"
        else:
            kind = "panel"
        self.calls[kind].append(body)
        self.order.append(kind)
        self.headers.append((kind, headers["Authorization"]))
        replies = self.lists[kind]
        return _resp(replies[min(len(self.calls[kind]) - 1, len(replies) - 1)], self.costs[kind])


class JudgeReviseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"OPENROUTER_KEY_GLM_5": "sk-test-SECRET"})
        self.env.start()
        os.environ.pop("OPENROUTER_KEY_JUDGE", None)
        os.environ.pop("OPENROUTER_API_KEY", None)  # the shared key would take precedence over the panel fallback
        self.stdout = io.StringIO()
        self.out = mock.patch("sys.stdout", self.stdout)
        self.out.start()
        raw = self.root / "results" / "raw"
        self.calls_path, self.rev_path, self.judge_path = raw / "calls.jsonl", raw / "revisions.jsonl", raw / "judge_calls.jsonl"

    def tearDown(self):
        self.out.stop()
        self.env.stop()
        self.tmp.cleanup()

    def write_sources(self, *recs):
        for r in recs:
            store.append_jsonl(self.calls_path, r)

    def _run(self, fake, *extra, max_rev="5", budget="10", judge=True):
        args = ["--source-tag", "main", "--conditions", "lean,cot", "--tag", "rj", "--max-revisions", max_rev,
                "--budget", budget, "--models", "glm-5", *(["--judge"] if judge else []), *extra]
        return revise.main(args, transport=fake, root=self.root)

    def revs(self):
        return store.read_jsonl(self.rev_path)

    def test_feedback_inserted_and_format_tail_by_code(self):
        src = source_record(BAD)
        self.write_sources(src)
        fake = Fake([lean_json(OK)])
        self._run(fake)
        self.assertEqual(fake.order, ["feedback", "panel", "verdict"])
        jmsg = fake.calls["feedback"][0]
        self.assertEqual(jmsg["temperature"], 0.0)
        self.assertNotIn("reasoning", jmsg)
        user = jmsg["messages"][1]["content"]
        self.assertIn(f"CR = {src['CR']:.3f}", user)
        for line in revise.contradiction_lines("main", BAD) + revise.previous_answer_lines("main", BAD):
            self.assertIn(line, user)
        self.assertTrue(user.endswith(revise.JUDGE_FEEDBACK_INSTRUCTION))
        for line in revise.circular_lines("main", BAD):  # circular triads are in the code-computed block
            self.assertIn(line, user)
        self.assertTrue(revise.circular_lines("main", BAD))
        self.assertNotIn("smallest", panel_text := "\n".join(m["content"] for m in fake.calls["panel"][0]["messages"][2:]))
        self.assertIn("1/9, 1/8, 1/7", panel_text)
        self.assertNotIn("stated ranking", user)  # lean: no ranking
        panel = fake.calls["panel"][0]["messages"]
        self.assertEqual(panel[:2], src["messages"])
        self.assertEqual(panel[2], {"role": "assistant", "content": src["raw_reply"]})
        expected = (FEEDBACK + "\n\nYour previous answers:\n" + "\n".join(revise.previous_answer_lines("main", BAD))
                    + "\n\nUse only these values: 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9.\n"
                    "You may change the direction of an answer if that is what resolves a contradiction.\n"
                    "Answer all questions again in the same JSON format.")
        self.assertEqual(panel[3], {"role": "user", "content": expected})
        r = self.revs()[0]
        self.assertEqual((r["judge_model_id"], r["judge_feedback"], r["judge_feedback_ok"]),
                         ("x-ai/grok-4.20", FEEDBACK, True))
        self.assertEqual(r["messages"], panel)
        self.assertEqual(r["prompt_id"], "revision_judge_lean_main_v2")

    def test_cot_ranking_and_reasoning_passed(self):
        src = source_record(BAD, cond="cot")
        self.write_sources(src)
        fake = Fake([cot_json(OK)])
        self._run(fake)
        self.assertIn("Respondent's own stated ranking (most to least important): Quality Performance > "
                      "Financial Performance", fake.calls["feedback"][0]["messages"][1]["content"])
        v = fake.calls["verdict"][0]["messages"][1]["content"]
        self.assertIn("Respondent's original stated reasoning:\nquality first", v)
        self.assertIn("ORIGINAL answers (first answers):\n" + revise.previous_answer_lines("main", BAD)[0], v)
        self.assertIn("Implied ranking (most to least important, weights): " + revise.ranking_line("main", OK), v)
        self.assertTrue(v.endswith(revise.JUDGE_VERDICT_INSTRUCTION))

    def test_fallback_when_judge_reply_unusable(self):
        src = source_record(BAD)
        self.write_sources(src)
        for bad_reply in ("no json here", json.dumps({"feedback": ""}), json.dumps({"message": "x"})):
            with self.subTest(reply=bad_reply):
                self.rev_path.unlink(missing_ok=True)
                fake = Fake([lean_json(OK)], feedback=[bad_reply])
                self._run(fake)
                panel = fake.calls["panel"][0]["messages"]
                self.assertEqual(panel[3]["content"], revise.revision_prompt("main", BAD, src["CR"]))
                r = self.revs()[0]
                self.assertFalse(r["judge_feedback_ok"])
                self.assertIsNone(r["judge_feedback"])

    def test_fallback_when_judge_call_fails(self):
        src = source_record(BAD)
        self.write_sources(src)

        def transport(url, headers, body):
            if body["model"] == config.JUDGE["id"]:
                return 400, {"error": {"message": "bad request"}}
            return _resp(lean_json(OK), 0.0)
        self._run(transport)
        r = self.revs()[0]
        self.assertFalse(r["judge_feedback_ok"])
        self.assertEqual(r["messages"][3]["content"], revise.revision_prompt("main", BAD, src["CR"]))
        self.assertFalse(r["judge_verdict_ok"])
        self.assertIsNone(r["judge_verdict"])
        self.assertTrue(r["parse_ok"])

    def test_verdict_stored_and_logic_metrics(self):
        self.write_sources(source_record(BAD))
        self._run(Fake([lean_json(OK2)], verdict=[verdict_json("PARTLY")]))
        r = self.revs()[0]
        self.assertTrue(r["judge_verdict_ok"])
        self.assertEqual(r["judge_verdict"], {"logic_preserved": "partly",
                                              "changed_priorities": ["Financial Performance"], "comment": "ok"})
        lm = r["logic_metrics"]
        self.assertEqual(lm, revise.logic_metrics("main", BAD, OK2))
        self.assertEqual(lm["n_answers_changed"], 10)
        for k in ("n_answers_changed", "spearman", "top_criterion_unchanged", "bottom_criterion_unchanged",
                  "max_abs_weight_change_pp"):
            self.assertIn(k, lm)
        same = revise.logic_metrics("main", BAD, BAD)
        self.assertEqual((same["n_answers_changed"], same["top_criterion_unchanged"],
                          same["bottom_criterion_unchanged"], same["max_abs_weight_change_pp"]), (0, True, True, 0.0))
        self.assertAlmostEqual(same["spearman"], 1.0)

    def test_invalid_verdict_value(self):
        self.write_sources(source_record(BAD))
        self._run(Fake([lean_json(OK)], verdict=[json.dumps({"logic_preserved": "maybe"})]))
        r = self.revs()[0]
        self.assertFalse(r["judge_verdict_ok"])
        self.assertIsNone(r["judge_verdict"])

    def test_stopping_unaffected_by_verdict(self):
        self.write_sources(source_record(BAD), source_record(BAD, rep=2))
        # consistent reply + verdict "no" -> still stops after 1; inconsistent + "yes" -> runs to max
        fake = Fake([lean_json(OK), lean_json(MILD), lean_json(MILD)], verdict=[verdict_json("no"), verdict_json("yes")])
        self._run(fake, max_rev="3")
        recs = self.revs()
        rep1 = [r for r in recs if r["rep"] == 1]
        rep2 = [r for r in recs if r["rep"] == 2]
        self.assertEqual(len(rep1), 1)
        self.assertEqual(rep1[0]["judge_verdict"]["logic_preserved"], "no")
        self.assertEqual([r["revision_number"] for r in rep2], [1, 2, 3])
        self.assertTrue(all(r["judge_verdict"]["logic_preserved"] == "yes" for r in rep2))

    def test_unparsable_reply_reuses_feedback(self):
        self.write_sources(source_record(BAD))
        fake = Fake(["garbage", lean_json(OK)])
        self._run(fake)
        self.assertEqual(fake.order, ["feedback", "panel", "panel", "verdict"])
        r1, r2 = self.revs()
        self.assertIsNone(r1["judge_verdict"])
        self.assertIsNone(r1["judge_verdict_ok"])
        self.assertIsNone(r1["logic_metrics"])
        self.assertTrue(r2["judge_feedback_reused"])
        self.assertEqual(r2["judge_feedback"], FEEDBACK)
        self.assertEqual(fake.calls["panel"][1]["messages"], fake.calls["panel"][0]["messages"])

    def test_second_revision_feedback_uses_current_answers(self):
        self.write_sources(source_record(BAD))
        fake = Fake([lean_json(MILD), lean_json(OK)])
        self._run(fake)
        self.assertEqual(len(fake.calls["feedback"]), 2)
        self.assertIn(revise.previous_answer_lines("main", MILD)[0], fake.calls["feedback"][1]["messages"][1]["content"])
        m2 = fake.calls["panel"][1]["messages"]
        self.assertEqual(len(m2), 4)  # best-so-far: one assistant turn, no accumulated turns
        self.assertEqual(m2[2], {"role": "assistant", "content": lean_json(MILD)})
        self.assertTrue(m2[3]["content"].startswith(FEEDBACK + "\n\nYour previous answers:\nQ1: "))
        self.assertEqual(self.revs()[1]["based_on_revision"], 1)
        # verdict 2 compares against the ORIGINAL (source) answers
        v2 = fake.calls["verdict"][1]["messages"][1]["content"]
        self.assertIn("ORIGINAL answers (first answers):\n" + revise.previous_answer_lines("main", BAD)[0], v2)
        self.assertEqual(self.revs()[1]["logic_metrics"], revise.logic_metrics("main", BAD, OK))

    def test_judge_cost_in_budget_and_resume(self):
        self.write_sources(source_record(BAD), source_record(BAD, rep=2))
        fake = Fake([lean_json(MILD)], panel_cost=0.0, judge_cost=0.3)
        self._run(fake, budget="1.0")
        # step 1: 0 < 1 -> 0.6 spent; step 2: 0.6 < 1 -> 1.2; stop
        self.assertEqual(len(fake.calls["panel"]), 2)
        recs = self.revs()
        self.assertEqual([r["judge_cost"] for r in recs], [0.6, 0.6])
        self.assertIn("judge=$0.6000 total=$1.2000", self.stdout.getvalue())
        fake2 = Fake([lean_json(MILD)], panel_cost=0.0, judge_cost=0.3)
        self._run(fake2, budget="1.0")  # judge spending of the tag counts on resume
        self.assertEqual(fake2.order, [])
        fake3 = Fake([lean_json(OK)], panel_cost=0.0, judge_cost=0.0)
        self._run(fake3, budget="10")
        rep1 = [r for r in self.revs() if r["rep"] == 1]
        self.assertEqual([r["revision_number"] for r in rep1], [1, 2, 3])
        self.assertEqual(fake3.calls["panel"][0]["messages"], rep1[1]["messages"])  # best (rev 1) unchanged

    def test_judge_calls_jsonl(self):
        self.write_sources(source_record(BAD))
        self._run(Fake([lean_json(MILD), lean_json(OK)]))
        jc = store.read_jsonl(self.judge_path)
        self.assertEqual([(j["purpose"], j["revision_number"]) for j in jc],
                         [("feedback", 1), ("verdict", 1), ("feedback", 2), ("verdict", 2)])
        for j in jc:
            self.assertEqual(j["chain_key"], ["glm-5", "lean", "main", 1])
            self.assertEqual((j["run_tag"], j["source_tag"], j["model_id"]), ("rj", "main", "x-ai/grok-4.20"))
            for k in ("messages", "raw_reply", "usage", "cost", "applied_params", "judge_parsed"):
                self.assertIn(k, j)
            self.assertEqual(j["applied_params"]["temperature"], 0.0)
        self.assertEqual(jc[0]["raw_reply"], FB_JSON)
        self.assertAlmostEqual(sum(j["cost"] for j in jc), sum(r["judge_cost"] for r in self.revs()))
        self.assertFalse(self.calls_path.read_text(encoding="utf-8").count("judge"))

    def test_no_key_material_and_key_fallback(self):
        self.write_sources(source_record(BAD))
        fake = Fake([lean_json(OK)])
        self._run(fake)
        self.assertTrue(all(h == "Bearer sk-test-SECRET" for _, h in fake.headers))  # fallback to panel key
        for path in (self.rev_path, self.judge_path):
            self.assertNotIn("SECRET", path.read_text(encoding="utf-8"))
        self.assertNotIn("SECRET", self.stdout.getvalue())
        self.rev_path.unlink()
        with mock.patch.dict(os.environ, {"OPENROUTER_KEY_JUDGE": "sk-judge-HIDDEN"}):
            fake = Fake([lean_json(OK)])
            self._run(fake)
        self.assertEqual({h for k, h in fake.headers if k != "panel"}, {"Bearer sk-judge-HIDDEN"})
        self.assertEqual({h for k, h in fake.headers if k == "panel"}, {"Bearer sk-test-SECRET"})
        for path in (self.rev_path, self.judge_path):
            self.assertNotIn("HIDDEN", path.read_text(encoding="utf-8"))

    def test_judge_off_reproduces_previous_behaviour(self):
        src = source_record(BAD)
        self.write_sources(src)
        fake = Fake([lean_json(MILD), lean_json(OK)])
        self._run(fake, judge=False)
        self.assertEqual(fake.order, ["panel", "panel"])
        self.assertFalse(self.judge_path.exists())
        r1, r2 = self.revs()
        self.assertEqual(fake.calls["panel"][0]["messages"][3]["content"], revise.revision_prompt("main", BAD, src["CR"]))
        self.assertEqual(fake.calls["panel"][1]["messages"][3]["content"],
                         revise.revision_prompt("main", MILD, r1["CR"]))
        for r in (r1, r2):
            self.assertFalse([k for k in r if k.startswith("judge") or k == "logic_metrics"])
            self.assertEqual(r["prompt_id"], "revision_lean_main_v2")
        self.assertNotIn("judge=", self.stdout.getvalue())

    def test_dry_run_reports_judge_calls(self):
        self.write_sources(source_record(BAD), source_record(BAD, rep=2))
        fake = Fake([lean_json(OK)])
        self.assertEqual(self._run(fake, "--dry-run", max_rev="3"), 0)
        out = self.stdout.getvalue()
        self.assertIn("at most 6 calls", out)
        self.assertIn("judge: x-ai/grok-4.20 (T=0.0), at most 12 judge calls", out)
        self.assertEqual(fake.order, [])
        self.assertFalse(self.judge_path.exists())
        self._run(fake, "--dry-run", max_rev="3", judge=False)
        self.assertEqual(self.stdout.getvalue().count("judge:"), 1)


class ClientModelOverrideTests(unittest.TestCase):
    def test_model_dict_and_key_lookup(self):
        seen = []

        def transport(url, headers, body):
            seen.append((headers["Authorization"], body))
            return _resp("{}", 0.0)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("OPENROUTER_KEY_") and k != "OPENROUTER_API_KEY"}  # shared key: test_problem
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(client.api_key_for(config.JUDGE))
            with self.assertRaises(RuntimeError):
                client.call_model("grok-4.20", [], temperature=0.0, reasoning=None, max_tokens=10,
                                  transport=transport, model=config.JUDGE)
            os.environ["OPENROUTER_KEY_GPT_4O"] = "sk-panel"
            # panel models never borrow another model's key
            self.assertIsNone(client.api_key_for(config.MODELS["glm-5"]))
            rec = client.call_model("grok-4.20", [{"role": "user", "content": "x"}], temperature=0.0, reasoning=None,
                                    max_tokens=10, transport=transport, model=config.JUDGE)
            self.assertEqual(seen[-1][0], "Bearer sk-panel")
            self.assertEqual(seen[-1][1]["model"], "x-ai/grok-4.20")
            self.assertEqual(seen[-1][1]["temperature"], 0.0)
            self.assertNotIn("reasoning", seen[-1][1])
            self.assertEqual((rec["model_key"], rec["model_id"]), ("grok-4.20", "x-ai/grok-4.20"))
            self.assertNotIn("sk-panel", json.dumps(rec))
            os.environ["OPENROUTER_KEY_JUDGE"] = "sk-judge"
            self.assertEqual(client.api_key_for(config.JUDGE), "sk-judge")
            os.environ["OPENROUTER_KEY_GLM_5"] = "sk-glm"
            client.call_model("glm-5", [], temperature=0.6, reasoning=None, max_tokens=10, transport=transport)
            self.assertEqual(seen[-1], ("Bearer sk-glm", seen[-1][1]))
            self.assertEqual(seen[-1][1]["model"], "z-ai/glm-5")


class ReportT20Tests(unittest.TestCase):
    def _records(self):
        records = []
        for model in ("gpt-4o", "glm-5"):
            records.append(source_record(BAD, model=model, rep=1))
            records.append(source_record(OK2, model=model, rep=2))
            records.append(source_record(OK_DEL, model=model, task="delivery", rep=1))
        return records

    def _revs(self, tag, judge):
        revs = []
        # gpt-4o rep1: rev1 MILD (unreached), rev2 OK2 (reached); glm-5 rep1: rev1 BAD (not reached)
        for model, seq, verdicts in (("gpt-4o", [MILD, OK2], ["no", "yes"]), ("glm-5", [BAD], ["no"])):
            for k, (vals, v) in enumerate(zip(seq, verdicts), 1):
                r = source_record(vals, model=model, rep=1, tag=tag)
                r.update(source_tag="main", revision_number=k)
                if judge:
                    r.update(judge_model_id="x-ai/grok-4.20", judge_verdict={"logic_preserved": v},
                             judge_verdict_ok=True)
                revs.append(r)
        return revs

    def _table(self, d, revs, revision_tag):
        inp = Path(d) / "calls.jsonl"
        inp.write_text("\n".join(json.dumps(r) for r in self._records()) + "\n", encoding="utf-8")
        rp = Path(d) / f"{revision_tag}.jsonl"
        rp.write_text("\n".join(json.dumps(r) for r in revs) + "\n", encoding="utf-8")
        out = Path(d) / f"out_{revision_tag}"
        report.run_report(inp, out, tag="main", data_dir=DATA, figures=False, revision_tag=revision_tag,
                          revisions_path=rp)
        names = {p.stem for p in (out / "tables").glob("*.csv")}
        self.assertTrue({"T16_revision_summary", "T17_weights_before_after_revision"} <= names)
        text = (out / "tables" / "T20_logic_preservation.csv").read_text(encoding="utf-8")
        return {(r["Model"], r["Condition"], r["Task"]): r for r in csv.DictReader(io.StringIO(text))}

    def test_t20_with_judge(self):
        with tempfile.TemporaryDirectory() as d:
            t = self._table(d, self._revs("revision_judge", True), "revision_judge")
            g = t[("gpt-4o", "direct", "main")]
            lm = revise.logic_metrics("main", BAD, OK2)
            self.assertEqual((g["Chains revised"], g["Reached CR ≤ 0.10"], g["Mean revisions used"],
                              g["Mean answers changed"]), ("1", "1", "2.000", "10.000"))
            self.assertEqual(g["Mean Spearman rho"], f"{lm['spearman']:.3f}")
            self.assertEqual(g["Mean max abs weight change (pp)"], f"{lm['max_abs_weight_change_pp']:.2f}")
            self.assertEqual((g["Judge yes"], g["Judge partly"], g["Judge no"]), ("1", "0", "0"))  # final step
            z = t[("glm-5", "direct", "main")]
            self.assertEqual((z["Chains revised"], z["Reached CR ≤ 0.10"], z["Mean answers changed"],
                              z["Mean Spearman rho"], z["Top criterion unchanged"], z["Ranking unchanged"],
                              z["Mean max abs weight change (pp)"]),
                             ("1", "0", "0.000", "1.000", "1.0000", "1.0000", "0.00"))
            p = t[("Panel", "direct", "main")]
            self.assertEqual((p["Chains revised"], p["Reached CR ≤ 0.10"], p["Mean revisions used"],
                              p["Mean answers changed"], p["Judge yes"], p["Judge partly"], p["Judge no"]),
                             ("2", "1", "1.500", "5.000", "1", "0", "0"))  # glm-5: original stays best
            self.assertNotIn(("Panel", "direct", "delivery"), t)  # no revised delivery chains

    def test_t20_without_judge_data(self):
        with tempfile.TemporaryDirectory() as d:
            t = self._table(d, self._revs("revision", False), "revision")
            p = t[("Panel", "direct", "main")]
            self.assertEqual((p["Chains revised"], p["Judge yes"], p["Judge partly"], p["Judge no"]), ("2", "", "", ""))


if __name__ == "__main__":
    unittest.main()
