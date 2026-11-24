"""Offline tests: no network. All model calls go through a fake transport."""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ahp_bench import ahp, client, prompts, run, store  # noqa: E402
from ahp_bench.parse import ParseError, extract_json, parse_guided, parse_lean_cot  # noqa: E402

SCALE_LINE = re.compile(r"^(?:[1-9]|1/[2-9])\s+: ", re.M)
QUESTION_LINE = re.compile(r"^\d+\.\s+Left: .+ \| Right: .+$", re.M)


def lean_reply(n_q, value="3"):
    return json.dumps({"answers": [{"question": q, "value": value} for q in range(1, n_q + 1)]})


class FakeTransport:
    """Callable(url, headers, body) -> (status, response). `script` maps the call index to a
    response; otherwise a successful reply produced by `reply_fn(body)` is returned."""

    def __init__(self, reply_fn=None, cost=0.001, reject=None):
        self.calls, self.cost = [], cost
        self.reply_fn = reply_fn or (lambda body: lean_reply(10))
        self.reject = reject or (lambda body: None)

    def __call__(self, url, headers, body):
        self.calls.append(json.loads(json.dumps(body)))
        rej = self.reject(body)
        if rej:
            return rej
        return 200, {
            "id": "gen-123", "provider": "FakeProvider",
            "choices": [{"message": {"content": self.reply_fn(body)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost": self.cost,
                      "completion_tokens_details": {"reasoning_tokens": 0}},
        }


def task_reply(body):
    user = body["messages"][1]["content"]
    return lean_reply(len(QUESTION_LINE.findall(user)) or 6)


class PromptTests(unittest.TestCase):
    def test_lean_cot_messages(self):
        for cond in ("lean", "cot"):
            for task, n_q in (("main", 10), ("delivery", 6)):
                msgs = prompts.build_messages(cond, task)
                self.assertEqual([m["role"] for m in msgs], ["system", "user"])
                self.assertEqual(msgs[0]["content"], prompts.LEAN_SYSTEM)
                user = msgs[1]["content"]
                self.assertEqual(len(SCALE_LINE.findall(user)), 17)
                self.assertEqual(len(QUESTION_LINE.findall(user)), n_q)
                self.assertNotIn("survey", json.dumps(msgs).lower())
                self.assertEqual("Before answering, think step by step:" in user, cond == "cot")
        main = prompts.build_messages("lean", "main")[1]["content"]
        self.assertIn("1.  Left: Delivery and Operational Performance | Right: Financial Performance", main)
        self.assertIn("10. Left: Volume and Logistics | Right: Compliance and Reliability", main)
        self.assertIn("9   : A is extremely more important than B.", main)
        deliv = prompts.build_messages("cot", "delivery")[1]["content"]
        self.assertIn("6. Left: Cancelled Orders | Right: Partially Delivered Orders", deliv)

    def test_guided_messages_match_thesis_serialisation(self):
        for task, fname in (("main", "eda_main_criteria_user_prompt.json"),
                            ("delivery", "eda_delivery_subcriteria_user_prompt.json")):
            msgs = prompts.build_messages("guided", task)
            exported = (prompts.GUIDED_DIR / fname).read_bytes().decode("utf-8")
            self.assertEqual(msgs[1]["content"], exported)
            self.assertNotIn("\r", msgs[0]["content"])
            self.assertTrue(msgs[0]["content"].startswith("You are an expert in Analytic Hierarchy Process"))
        self.assertEqual(len(prompts.guided_labels("main")), 5)
        self.assertEqual(len(prompts.guided_labels("delivery")), 4)


class AhpTests(unittest.TestCase):
    def test_answers_to_reciprocal_matrix(self):
        a = ahp.matrix_from_answers(3, [(0, 1), (0, 2), (1, 2)], ["3", "1/5", 7])
        self.assertAlmostEqual(a[0, 1], 3)
        self.assertAlmostEqual(a[1, 0], 1 / 3)
        self.assertAlmostEqual(a[0, 2], 0.2)
        self.assertAlmostEqual(a[2, 0], 5)
        for i in range(3):
            self.assertEqual(a[i, i], 1)
            for j in range(3):
                self.assertAlmostEqual(a[i, j] * a[j, i], 1)
        with self.assertRaises(ValueError):
            ahp.matrix_from_answers(3, [(0, 1), (0, 2), (1, 2)], ["3", "10", "1"])
        with self.assertRaises(ValueError):
            ahp.scale_value("0.333")

    def test_consistent_matrix_cr_zero(self):
        w = [0.4, 0.3, 0.15, 0.1, 0.05]
        a = [[wi / wj for wj in w] for wi in w]
        c = ahp.consistency(a)
        self.assertAlmostEqual(c["CR"], 0, places=9)
        for x, y in zip(c["weights"], w):
            self.assertAlmostEqual(x, y, places=9)

    def test_known_3x3_cr(self):
        c = ahp.consistency([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]])
        self.assertAlmostEqual(c["lambda_max"], 3.0385, places=3)
        self.assertAlmostEqual(c["CR"], 0.0332, places=3)
        self.assertAlmostEqual(c["weights"][0], 0.637, places=3)

    def test_geometric_mean(self):
        g = ahp.geometric_mean_matrix([[[1, 4], [0.25, 1]], [[1, 1], [1, 1]]])
        self.assertAlmostEqual(g[0, 1], 2)


class ParseTests(unittest.TestCase):
    def test_fenced_json(self):
        reply = "Sure:\n```json\n" + lean_reply(6, "1/3") + "\n```\nDone."
        out = parse_lean_cot(reply, 6)
        self.assertEqual(out["answers"], ["1/3"] * 6)

    def test_cot_fields_and_errors(self):
        obj = {"reasoning": "r {with braces}", "ranking": ["A", "B"],
               "answers": [{"question": q, "value": "2"} for q in range(1, 7)]}
        out = parse_lean_cot(json.dumps(obj), 6, cot=True)
        self.assertEqual(out["ranking"], ["A", "B"])
        self.assertEqual(out["reasoning"], "r {with braces}")
        with self.assertRaises(ParseError):
            parse_lean_cot(lean_reply(5), 6)  # missing question
        with self.assertRaises(ParseError):
            parse_lean_cot(lean_reply(6, "2.5"), 6)  # off scale, not repaired
        with self.assertRaises(ParseError):
            parse_lean_cot('{"answers": [', 6)

    def test_guided_fraction_strings(self):
        reply = json.dumps({"labels": ["A", "B", "C"],
                            "matrix": [[1, 3, "5"], ["1/3", 1, "3"], ["1/5", "1/3", 1]],
                            "rationales": [["", "x", "y"], ["", "", "z"], ["", "", ""]]})
        g = parse_guided(reply, 3)
        self.assertTrue(g["reciprocal_returned"])
        self.assertTrue(g["upper_on_scale"])
        self.assertAlmostEqual(g["raw_matrix"][1][0], 1 / 3)
        self.assertAlmostEqual(ahp.consistency(g["matrix"])["CR"], 0.0332, places=3)

    def test_guided_non_reciprocal_uses_upper_triangle(self):
        reply = '{"labels": ["A","B","C"], "matrix": [[1, 3, 5], [1/2, 1, 3], [1/5, 1/3, 1]]}'
        g = parse_guided(reply, 3)  # bare fractions are quoted, values unchanged
        self.assertTrue(g["bare_fractions_quoted"])
        self.assertFalse(g["reciprocal_returned"])
        self.assertAlmostEqual(g["raw_matrix"][1][0], 0.5)
        self.assertAlmostEqual(g["matrix"][1][0], 1 / 3)
        with self.assertRaises(ParseError):
            parse_guided('{"matrix": [[1, 2], [0.5, 1]]}', 3)

    def test_extract_first_balanced(self):
        obj, _ = extract_json('text {"a": "}"} trailing {"b": 1}')
        self.assertEqual(obj, {"a": "}"})


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"OPENROUTER_KEY_GLM_5": "sk-test-SECRET",
                                                "OPENROUTER_KEY_GPT_4O": "sk-test-SECRET"})
        self.env.start()
        self.sleep = mock.patch.object(client, "_sleep", lambda s: None)
        self.sleep.start()

    def tearDown(self):
        self.env.stop()
        self.sleep.stop()

    def _call(self, transport, model="glm-5", reasoning=None):
        reasoning = {"effort": "none"} if reasoning is None and model != "gpt-4o" else reasoning
        return client.call_model(model, [{"role": "user", "content": "hi"}], temperature=0.6,
                                 reasoning=reasoning, max_tokens=4000, transport=transport)

    def test_reasoning_fallback_then_accept(self):
        def reject(body):
            if body.get("reasoning") in ({"effort": "none"}, {"effort": "minimal"}):
                return 400, {"error": {"code": 400, "message": "Reasoning effort 'none' is not supported"}}
        fake = FakeTransport(reject=reject)
        rec = self._call(fake)
        self.assertTrue(rec["ok"])
        self.assertEqual([c.get("reasoning") for c in fake.calls],
                         [{"effort": "none"}, {"effort": "minimal"}, {"effort": "low"}])
        self.assertEqual(rec["applied_params"]["reasoning"], {"effort": "low"})
        self.assertEqual(rec["requested_params"]["reasoning"], {"effort": "none"})
        self.assertEqual(len(rec["http_attempts"]), 3)
        self.assertEqual(rec["cost"], 0.001)
        self.assertEqual(rec["provider"], "FakeProvider")
        self.assertEqual(rec["generation_id"], "gen-123")
        self.assertEqual(fake.calls[0]["usage"], {"include": True})

    def test_reasoning_omitted_and_temperature_dropped(self):
        def reject(body):
            if "reasoning" in body:
                return 400, {"error": {"message": "Unsupported parameter: reasoning"}}
            if "temperature" in body:
                return 422, {"error": {"message": "temperature is not supported for this model"}}
        fake = FakeTransport(reject=reject)
        rec = self._call(fake)
        self.assertTrue(rec["ok"])
        # none, minimal, low (rejected) -> reasoning omitted, temperature rejected -> omitted -> ok
        self.assertEqual(len(fake.calls), 5)
        self.assertNotIn("reasoning", fake.calls[-1])
        self.assertNotIn("temperature", fake.calls[-1])
        self.assertEqual(sorted(rec["applied_params"]["omitted"]), ["reasoning", "temperature"])

    def test_gpt4o_sends_no_reasoning(self):
        fake = FakeTransport()
        self._call(fake, model="gpt-4o", reasoning=None)
        self.assertNotIn("reasoning", fake.calls[0])

    def test_transient_retry(self):
        n = {"i": 0}

        def reject(body):
            n["i"] += 1
            return (503, {"error": {"message": "overloaded"}}) if n["i"] <= 2 else None
        rec = self._call(FakeTransport(reject=reject))
        self.assertTrue(rec["ok"])
        self.assertEqual([a["status"] for a in rec["http_attempts"]], [503, 503, 200])

    def test_no_key_material_in_record(self):
        def reject(body):  # an error that echoes the key must still not leak
            if body.get("reasoning") == {"effort": "none"}:
                return 400, {"error": {"message": "reasoning unsupported for sk-test-SECRET"}}
        fake = FakeTransport(reject=reject)
        rec = self._call(fake)
        self.assertNotIn("SECRET", json.dumps(rec))
        self.assertNotIn("Authorization", json.dumps(rec))


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"OPENROUTER_KEY_GLM_5": "sk-test-SECRET"})
        self.env.start()
        self.out = mock.patch("sys.stdout", new_callable=lambda: open(os.devnull, "w", encoding="utf-8"))
        self.out.start()

    def tearDown(self):
        sys.stdout.close()
        self.out.stop()
        self.env.stop()
        self.tmp.cleanup()

    def _run(self, fake, *extra):
        args = ["--models", "glm-5", "--conditions", "lean", "--tasks", "main,delivery",
                "--reps", "2", "--tag", "t1", *extra]
        return run.main(args, transport=fake, root=self.root)

    @property
    def calls_path(self):
        return self.root / "results" / "raw" / "calls.jsonl"

    def test_full_record(self):
        self._run(FakeTransport(reply_fn=task_reply), "--budget", "10")
        recs = store.read_jsonl(self.calls_path)
        self.assertEqual(len(recs), 4)
        r = recs[0]
        self.assertTrue(r["parse_ok"])
        for key in ("run_tag", "model_id", "developer", "prompt_id", "prompt_sha256", "messages", "raw_reply",
                    "answers", "matrix", "weights", "lambda_max", "CI", "CR", "applied_params", "usage", "cost"):
            self.assertIn(key, r)
        self.assertEqual(list(r["weights"]), prompts.MAIN_CRITERIA)
        self.assertNotIn("SECRET", self.calls_path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(store.total_cost(self.calls_path, "t1"), 0.004)

    def test_parse_failure_recorded_and_continues(self):
        self._run(FakeTransport(reply_fn=lambda b: "no json here"), "--budget", "10")
        recs = store.read_jsonl(self.calls_path)
        self.assertEqual(len(recs), 4)
        self.assertTrue(all(not r["parse_ok"] and r["parse_error"] for r in recs))

    def test_budget_guard_stops(self):
        fake = FakeTransport(reply_fn=task_reply, cost=0.6)
        self._run(fake, "--budget", "1.0")
        self.assertEqual(len(fake.calls), 2)  # 0.6 < 1.0 -> call; 1.2 >= 1.0 -> stop
        fake2 = FakeTransport(reply_fn=task_reply, cost=0.6)
        self._run(fake2, "--budget", "1.0")  # previous records of the same tag count
        self.assertEqual(len(fake2.calls), 0)

    def test_resume_skips_done(self):
        self._run(FakeTransport(reply_fn=task_reply), "--budget", "10")
        fake = FakeTransport(reply_fn=task_reply)
        self._run(fake, "--budget", "10")
        self.assertEqual(len(fake.calls), 0)
        self.assertEqual(len(store.read_jsonl(self.calls_path)), 4)

    def test_dry_run_writes_messages_without_calls(self):
        fake = FakeTransport()
        rc = run.main(["--dry-run", "--budget", "1", "--models", "all"], transport=fake, root=self.root)
        self.assertEqual(rc, 0)
        self.assertEqual(fake.calls, [])
        files = sorted(p.name for p in (self.root / "results" / "dry_run").glob("*.json"))
        self.assertEqual(len(files), 6)
        self.assertFalse(self.calls_path.exists())


if __name__ == "__main__":
    unittest.main()
