"""Offline tests for ahp_bench.audit and ahp_bench.appendix_extra (synthetic records only)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ahp_bench import analysis as an  # noqa: E402
from ahp_bench import appendix_extra, audit, prompts, revise, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_data import DATA, STUDY_DIR  # noqa: E402
BAD = ["9", "1/9", "9", "1/9", "9", "1/9", "9", "1/9", "9", "1/9"]   # main task, inconsistent
GOOD = ["1"] * 10
TS = "2026-01-01T10:00:00+00:00"


def lean_reply(answers):
    return json.dumps({"answers": [{"question": q, "value": v} for q, v in enumerate(answers, 1)]})


def call(model="gpt-4o", cond="lean", task="main", rep=1, answers=None, tag="main", **extra):
    answers = answers or (GOOD if task == "main" else ["1"] * 6)
    msgs = prompts.build_messages(cond, task)
    reply = lean_reply(answers)
    r = {"run_tag": tag, "model_key": model, "model_id": f"x/{model}", "developer": "Dev", "condition": cond,
         "task": task, "rep": rep, "prompt_id": prompts.prompt_id(cond, task),
         "prompt_sha256": prompts.prompt_sha256(msgs), "messages": msgs, "ok": True, "error": None,
         "content": reply, "raw_reply": reply, "finish_reason": "stop",
         "usage": {"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 0}, "cost": 0.01,
         "provider": "P", "generation_id": f"gen-{model}-{cond}-{task}-{rep}-{tag}", "timestamp_utc": TS,
         "requested_params": {"temperature": 0.6, "reasoning": {"effort": "none"}, "max_tokens": 8000},
         "applied_params": {"temperature": 0.6, "reasoning": {"effort": "none"}, "max_tokens": 8000, "omitted": []},
         "http_attempts": [{"status": 200, "params": {}}]}
    r.update(run.analyse(cond, task, reply), parse_ok=True, parse_error=None)
    r.update(extra)
    return r


def revision_chain(src, replies, tag="revision_final", feedback="Please reconsider."):
    """Revision and judge records built exactly as revise.main does (judge feedback ok)."""
    revs, judges = [], []
    key = revise.chain_key(src)
    for answers in replies:
        st = revise.next_state(src, revs)
        msgs = list(st["messages"])
        msgs[-1] = {"role": "user", "content": revise.judge_revision_message(feedback, src["task"], st["answers"])}
        base = {"run_tag": tag, "source_tag": "main", "chain_key": list(key), "revision_number": st["number"]}
        judges.append({**base, "purpose": "feedback", "model_id": "x-ai/judge", "ok": True, "cost": 0.001,
                       "judge_parse_ok": True})
        reply = lean_reply(answers)
        r = {"run_tag": tag, "source_tag": "main", "model_key": src["model_key"], "condition": src["condition"],
             "task": src["task"], "rep": src["rep"], "revision_number": st["number"], "max_revisions": 5,
             "source_CR": revise._cr(src), "based_on_answers": list(st["answers"]), "based_on_CR": st["CR"],
             "based_on_revision": st["based_on_revision"], "prompt_sha256": prompts.prompt_sha256(msgs),
             "messages": msgs, "ok": True, "raw_reply": reply, "cost": 0.002,
             "requested_params": {"temperature": 0.6}, "judge_model_id": "x-ai/judge", "judge_feedback": feedback,
             "judge_feedback_ok": True, "judge_feedback_reused": False,
             "judge_verdict": {"logic_preserved": "yes", "changed_priorities": [], "comment": "ok"},
             "judge_verdict_ok": True, "judge_cost": 0.002}
        r.update(run.analyse(src["condition"], src["task"], reply), parse_ok=True, parse_error=None)
        judges.append({**base, "purpose": "verdict", "model_id": "x-ai/judge", "ok": True, "cost": 0.001,
                       "judge_parse_ok": True})
        revs.append(r)
    return revs, judges


class CompletenessTests(unittest.TestCase):
    def test_missing_and_duplicate(self):
        recs = [call(rep=1), call(rep=1), call(rep=3)]
        c = audit.completeness(recs, "main", ["gpt-4o"], ("lean",), ("main",), 3)
        self.assertEqual(c["missing"], [("gpt-4o", "lean", "main", 2)])
        self.assertEqual(c["duplicated"], [(("gpt-4o", "lean", "main", 1), 2)])
        self.assertEqual(c["cells_present"], 2)

    def test_complete(self):
        recs = [call(rep=r) for r in (1, 2)]
        c = audit.completeness(recs, "main", ["gpt-4o"], ("lean",), ("main",), 2)
        self.assertEqual((c["missing"], c["duplicated"], c["unexpected"]), ([], [], []))


class FieldTests(unittest.TestCase):
    def test_missing_fields_listed(self):
        good, bad = call(rep=1), call(rep=2, provider=None, generation_id="")
        rows, lacking = audit.field_completeness([("calls.jsonl", "main", [good, bad])])
        self.assertEqual(rows[0]["provider"], 1)
        self.assertEqual(rows[0]["records_lacking_any"], 1)
        self.assertEqual(lacking[0]["missing"], "provider;generation_id")

    def test_judge_calls_skip_prompt_id(self):
        j = {k: v for k, v in call().items() if k not in ("prompt_id", "prompt_sha256")}
        rows, lacking = audit.field_completeness([("judge_calls.jsonl", "revision_final", [j])])
        self.assertIsNone(rows[0]["prompt_id"])
        self.assertEqual(lacking, [])


class IntegrityTests(unittest.TestCase):
    def test_prompt_hash_and_current_code(self):
        a, b = call(rep=1), call(rep=2)
        b["messages"] = [dict(m) for m in b["messages"]]
        b["messages"][0]["content"] += " changed"
        rows = audit.prompt_integrity([a, b], ["main"])
        self.assertEqual(rows[0]["distinct_prompt_sha256"], 1)          # hash field not updated
        self.assertEqual(rows[0]["stored_hash_matches_stored_messages"], 1)
        self.assertEqual(rows[0]["messages_equal_current_code"], 1)

    def test_guided_files_match_code(self):
        msgs = prompts.build_messages("guided", "main")
        rows = audit.guided_file_check([{"run_tag": "main", "condition": "guided", "task": "main", "messages": msgs}])
        row = next(r for r in rows if r["task"] == "main")
        self.assertEqual((row["system_message_equals_file"], row["user_message_equals_file_bytes"]), (1, 1))
        self.assertTrue(row["user_file_equals_json_dumps_of_itself"])

    def test_recompute_detects_tampering(self):
        r = call(answers=BAD)
        self.assertLess(max(audit.recompute_record(r).values()), 1e-9)
        r["CR"] += 0.01
        self.assertAlmostEqual(audit.recompute_record(r)["CR"], 0.01, places=9)

    def test_reparse_check(self):
        r = call(answers=BAD)
        rows = audit.reparse_check({"calls.jsonl": ([r], 0)})
        self.assertEqual(rows[0]["parse_ok_agrees"], 1)
        self.assertEqual(rows[0]["answers_or_raw_matrix_equal"], 1)

    def test_inventory(self):
        recs = [call(rep=1), call(rep=2, ok=False, parse_ok=False, cost=None)]
        row = audit.inventory({"calls.jsonl": (recs, 0)})[0]
        self.assertEqual((row["records"], row["calls_ok"], row["calls_failed"], row["parse_ok"]), (2, 1, 1, 1))
        self.assertAlmostEqual(row["cost_usd"], 0.01)
        self.assertEqual(row["prompt_tokens"], 200)


class SecretScanTests(unittest.TestCase):
    def test_counts_without_content(self):
        prefix, bearer = "sk-" + "or-", "Bear" + "er "
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "results").mkdir()
            (root / "results" / "a.txt").write_text(f"{prefix}v1-{'a' * 64} {bearer}{prefix}v1-{'b' * 64}", encoding="utf-8")
            (root / "results" / "b.txt").write_text(f"{bearer}{{key}} {prefix}", encoding="utf-8")
            (root / ".env").write_text(f"X={prefix}v1-{'c' * 64}", encoding="utf-8")   # outside scanned dirs
            rows = {r["file"]: r for r in audit.secret_scan(root)}
        self.assertEqual(set(rows), {"results/a.txt", "results/b.txt"})
        self.assertEqual((rows["results/a.txt"]["key_like_count"], rows["results/a.txt"]["bearer_token_like_count"]), (2, 1))
        self.assertEqual((rows["results/b.txt"]["key_like_count"], rows["results/b.txt"]["bearer_count"]), (0, 1))
        self.assertEqual(rows["results/b.txt"]["key_prefix_count"], 1)

    def test_source_has_no_literal_patterns(self):
        text = (ROOT / "ahp_bench" / "audit.py").read_text(encoding="utf-8")
        self.assertNotIn("sk-" + "or-", text)
        self.assertNotIn("Bear" + "er ", text)


class RevisionAuditTests(unittest.TestCase):
    def setUp(self):
        self.src = call(answers=BAD)
        self.assertGreater(self.src["CR"], 0.1)
        self.calls = [self.src, call(rep=2)]

    def test_clean_chain(self):
        revs, judges = revision_chain(self.src, [BAD[:9] + ["9"], GOOD])
        res = audit.revision_audit(self.calls, revs, judges)
        self.assertEqual(res["violations"], [])
        self.assertEqual((res["chains_expected"], res["chains_present"], res["calls"]), (1, 1, 2))
        self.assertEqual(res["judge_calls_by_purpose"], {"feedback": 2, "verdict": 2})
        self.assertEqual(res["chains_reached"], 1)
        self.assertAlmostEqual(res["cost_judge"], 0.004)

    def test_call_after_consistent_is_flagged(self):
        revs, judges = revision_chain(self.src, [GOOD, GOOD])
        rules = {v["rule"] for v in audit.revision_audit(self.calls, revs, judges)["violations"]}
        self.assertIn("calls continued after a consistent revision", rules)

    def test_early_stop_and_missing_judge_call(self):
        revs, judges = revision_chain(self.src, [BAD])
        res = audit.revision_audit(self.calls, revs, judges[:1])
        rules = {v["rule"] for v in res["violations"]}
        self.assertIn("chain stopped before the limit without reaching consistency", rules)
        self.assertTrue(any(r.startswith("judge calls feedback=1 verdict=0") for r in rules))

    def test_tampered_message_is_flagged(self):
        revs, judges = revision_chain(self.src, [GOOD])
        revs[0]["messages"][-1] = {"role": "user", "content": "Make Quality the most important."}
        rules = {v["rule"] for v in audit.revision_audit(self.calls, revs, judges)["violations"]}
        self.assertIn("last user message differs from the code-built revision message", rules)
        self.assertIn("prompt_sha256 does not match stored messages", rules)


class ParamTestTests(unittest.TestCase):
    def test_distinct_answer_sets(self):
        e = {"temperature": 0}
        recs = [{"model_key": "m", "extra": e, "status": 200, "content": lean_reply(a), "cost": 0.001}
                for a in (GOOD, GOOD, BAD)]
        recs.append({"model_key": "m", "extra": e, "status": 400, "content": None,
                     "error": json.dumps({"message": "nope"})})
        row = audit.param_tests(recs)[0]
        self.assertEqual((row["calls"], row["http_200"], row["distinct_answer_sets"]), (4, 3, 2))
        self.assertEqual(row["errors"], "nope (1)")


class EndToEndTests(unittest.TestCase):
    def test_run_audit_and_outputs(self):
        src = call(answers=BAD)
        calls = [src] + [call(model="gpt-4o", cond=c, task=t, rep=r) for c in ("lean", "cot") for t in ("main", "delivery")
                         for r in (1, 2) if (c, t, r) != ("lean", "main", 1)]
        revs, judges = revision_chain(src, [GOOD])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = root / "results" / "raw"
            raw.mkdir(parents=True)
            for name, recs in (("calls.jsonl", calls), ("revisions.jsonl", revs), ("judge_calls.jsonl", judges)):
                (raw / name).write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
            before = {p.name: p.read_bytes() for p in raw.iterdir()}
            res = audit.run_audit(root / "results")
            out = audit.write_outputs(res, root / "results" / "report")
            self.assertEqual(before, {p.name: p.read_bytes() for p in raw.iterdir()})   # read-only
            self.assertTrue((out / "audit.md").exists())
            for name in ("a_inventory.csv", "c_field_completeness.csv", "g_recompute.csv", "h_secret_scan.csv",
                         "i_checksums.csv", "j_revision_violations.csv", "expectations.csv"):
                self.assertTrue((out / name).exists(), name)
            exp = {e["check"]: e["status"] for e in audit.expectations(res)}
            self.assertEqual(exp["j revision_final: protocol rule violations"], "PASS")
            self.assertEqual(exp["b main: cells present"], "FAIL")      # 1 model x 2 conditions only
            self.assertEqual(res["revision"]["chains_present"], 1)


class AppendixTests(unittest.TestCase):
    def test_tables(self):
        expert = an.load_expert_data(DATA / "expert_matrices.json")
        src = call(answers=BAD)
        calls = [src, call(rep=2), call(task="delivery", rep=1), call(cond="cot", rep=1)]
        revs, _ = revision_chain(src, [GOOD])
        out = appendix_extra.build(calls, revs, expert, "main", "revision_final")
        self.assertEqual(set(out), {"A04_group_matrices", "A05_model_matrices", "A06_group_matrices_after_revision",
                                    "A07_run_level", "A08_judge_feedback"})
        a04 = out["A04_group_matrices"][1]
        self.assertEqual(len(a04), (5 + 4) + (5 + 4) + 5)     # experts, lean main+delivery, cot main
        self.assertEqual(len(out["A07_run_level"][1]), 4)
        row = next(r for r in out["A07_run_level"][1] if r[:4] == ["gpt-4o", "direct", "main", "1"])
        h = out["A07_run_level"][0]
        self.assertEqual(row[h.index("needs revision (CR > 0.10)")], "y")
        self.assertEqual(row[h.index("final CR")], "0.0000")
        self.assertEqual(row[h.index("verdict final answers")], "yes")
        self.assertEqual(len(out["A08_judge_feedback"][1]), 1)
        a06 = out["A06_group_matrices_after_revision"][1]
        lean_main = [r for r in a06 if r[0] == "LLM panel Unguided, zero-shot (direct)" and r[2] == "main"]
        self.assertEqual(lean_main[0][-1], "0.0000")             # both lean main matrices now all-ones
        tex = appendix_extra.tex("A08_judge_feedback", *out["A08_judge_feedback"][:2], "d", ("feedback",))
        self.assertIn("p{4.5cm}", tex)

    def test_no_revisions(self):
        expert = an.load_expert_data(DATA / "expert_matrices.json")
        out = appendix_extra.build([call()], [], expert)
        self.assertEqual(set(out), {"A04_group_matrices", "A05_model_matrices", "A07_run_level"})


if __name__ == "__main__":
    unittest.main()
