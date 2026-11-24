"""Problem definitions, configurable model panel, key fallback and the template writer (offline)."""
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_data import DATA, STUDY_DIR  # noqa: E402
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ahp_bench import ahp, client, config, new_problem, prompts  # noqa: E402
from ahp_bench import analysis as an  # noqa: E402
from ahp_bench import problem as pb  # noqa: E402


def _no_openrouter_env():
    return {k: v for k, v in os.environ.items() if not k.startswith("OPENROUTER_")}


class StudyProblemTests(unittest.TestCase):
    def setUp(self):
        self.p = pb.load_problem(pb.STUDY_PROBLEM_PATH)

    def test_structure(self):
        self.assertEqual(self.p.task_names, ("main", "delivery"))
        self.assertTrue(self.p.is_study)
        self.assertEqual(self.p.conditions, ("lean", "cot", "guided"))
        self.assertEqual(self.p.pairs("main"), list(combinations(range(5), 2)))
        self.assertEqual(self.p.pairs("delivery"), list(combinations(range(4), 2)))
        self.assertEqual(self.p.leaf_short, an.DELIVERY_SHORT + an.MAIN_SHORT[1:])
        self.assertEqual([c for c, *_ in self.p.leaves], an.DELIVERY_SUBCRITERIA + an.MAIN_CRITERIA[1:])
        self.assertEqual(self.p.main.short, an.MAIN_SHORT)

    def test_global_weights_match_study_function(self):
        rng = np.random.default_rng(1)
        main, deliv = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(4))
        np.testing.assert_allclose(self.p.global_weights({"main": main, "delivery": deliv}),
                                   an.global_weights(main, deliv))
        self.assertIsNone(self.p.global_weights({"main": main}))

    def test_ranking_block_equals_data_file(self):
        data = an.load_performance(DATA / "performance_matrix.json")
        perf = self.p.performance(transformed=False)
        np.testing.assert_array_equal(perf["X"], data["matrix"])
        np.testing.assert_array_equal(perf["benefit"], data["benefit_mask"])
        self.assertEqual(perf["alternatives"], data["alternatives"])
        self.assertEqual(perf["codes"], data["codes"])
        self.assertEqual(perf["names"], data["criteria"])
        np.testing.assert_allclose(self.p.performance()["X"], an.delivery_rates(data["matrix"]))
        w = np.arange(1, 9) / 36
        np.testing.assert_allclose(self.p.indicator_weights(w), w)

    def test_prompts_from_problem_equal_module_constants(self):
        for cond in ("lean", "cot", "guided"):
            for task in ("main", "delivery"):
                self.assertEqual(prompts.build_messages(cond, task, self.p), prompts.build_messages(cond, task))
                self.assertEqual(prompts.prompt_id(cond, task, self.p), prompts.prompt_id(cond, task))
        self.assertEqual(prompts.LEAN_SYSTEM, "You are a procurement professional with experience in supplier "
                         "evaluation.\nAnswer the following questions. Respond only in the requested JSON format.")
        self.assertTrue(prompts.INTRO["main"].startswith("Five criteria used in supplier evaluation are compared"))
        self.assertTrue(prompts.INTRO["delivery"].startswith(
            'Four sub-criteria of "Delivery and Operational Performance" are compared in pairs.'))

    def test_prompts_equal_stored_study_records(self):
        if STUDY_DIR is None:
            self.skipTest("AHP_STUDY_DIR (study repository) not set")
        path = STUDY_DIR / "results" / "raw" / "calls.jsonl"
        seen = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                k = (r.get("condition"), r.get("task"))
                if r.get("run_tag") == "main" and k not in seen:
                    seen[k] = r
                if len(seen) == 6:
                    break
        self.assertEqual(len(seen), 6)
        for (c, t), r in seen.items():
            msgs = prompts.build_messages(c, t)
            self.assertEqual(r["messages"], msgs)
            self.assertEqual(r["prompt_sha256"], prompts.prompt_sha256(msgs))


class ExampleProblemTests(unittest.TestCase):
    def test_example_loads_and_builds_prompts(self):
        p = pb.load_problem(pb.EXAMPLE_PROBLEM_PATH)
        self.assertEqual(p.task_names, ("main", "performance"))
        self.assertFalse(p.is_study)
        self.assertEqual(p.conditions, ("lean", "cot"))
        self.assertEqual(len(p.leaves), 5)
        msgs = prompts.build_messages("lean", "main", p)
        self.assertEqual(msgs[0]["content"], p.role + prompts.SYSTEM_TAIL)
        self.assertTrue(msgs[1]["content"].startswith(
            "Three criteria used in choosing a laptop for university studies are compared in pairs."))
        self.assertIn("1. Left: Performance | Right: Price", msgs[1]["content"])
        self.assertIn("3. Left: Price | Right: Portability", msgs[1]["content"])
        cot = prompts.build_messages("cot", "performance", p)[1]["content"]
        self.assertTrue(cot.startswith('Three sub-criteria of "Performance" are compared in pairs.'))
        self.assertIn("represents in choosing a laptop for university studies.", cot)
        with self.assertRaises(ValueError):
            prompts.build_messages("guided", "main", p)
        self.assertEqual(prompts.prompt_id("lean", "main", p), "example_laptop_choice:lean_main_v2")
        self.assertIn("illustrative", json.dumps(p.raw).lower())

    def test_indicator_weight_order(self):
        p = pb.load_problem(pb.EXAMPLE_PROBLEM_PATH)
        g = p.global_weights({"main": [0.5, 0.3, 0.2], "performance": [0.5, 0.25, 0.25]})
        np.testing.assert_allclose(g, [0.25, 0.125, 0.125, 0.3, 0.2])
        np.testing.assert_allclose(p.indicator_weights(g), g)
        self.assertEqual(p.performance()["X"].shape, (4, 5))


class ValidationTests(unittest.TestCase):
    def base(self):
        return json.loads(pb.EXAMPLE_PROBLEM_PATH.read_text(encoding="utf-8"))

    def bad(self, mutate, msg):
        d = self.base()
        mutate(d)
        with self.assertRaisesRegex(pb.ProblemError, msg):
            pb.problem_from_dict(d)

    def test_errors(self):
        self.bad(lambda d: d.update(criteria=["A", "B"]), "3-9")
        self.bad(lambda d: d.update(criteria=[f"C{i}" for i in range(10)]), "3-9")
        self.bad(lambda d: d.update(criteria=["A", "a", "B"]), "duplicate")
        self.bad(lambda d: d.pop("goal"), "'goal' is required")
        self.bad(lambda d: d.update(name="Bad Name"), "lower-case")
        self.bad(lambda d: d["groups"][0].update(parent="Nope"), "not a main criterion")
        self.bad(lambda d: d["groups"][0].update(task="main"), "other than 'main'")
        self.bad(lambda d: d.update(question_order={"main": [[1, 2], [1, 3], [2, 1]]}), "appears twice")
        self.bad(lambda d: d.update(question_order={"main": [[1, 2], [1, 3]]}), "expected 3")
        self.bad(lambda d: d.update(question_order={"other": [[1, 2]]}), "unknown tasks")
        self.bad(lambda d: d["ranking"]["indicators"][0].update(direction="up"), "benefit")
        self.bad(lambda d: d["ranking"]["criterion_indicator"].pop("Price"), "missing leaves")
        self.bad(lambda d: d["ranking"]["matrix"].pop(), "shape")
        self.bad(lambda d: d["ranking"]["matrix"][0].__setitem__(0, 0), "> 0")
        self.bad(lambda d: d["ranking"].update(transform={"type": "log"}), "transform")

    def test_optional_parts(self):
        d = self.base()
        d.pop("ranking")
        d.pop("groups")
        d["criteria"] = [f"Criterion {k}" for k in range(1, 10)]
        d.pop("short")
        d["question_order"] = {"main": [[j, i] for i, j in combinations(range(1, 10), 2)]}
        p = pb.problem_from_dict(d)
        self.assertFalse(p.has_ranking)
        self.assertEqual(p.task_names, ("main",))
        self.assertEqual(p.pairs("main")[0], (1, 0))
        self.assertTrue(prompts.intro("main", p).startswith("Nine criteria"))

    def test_random_index_table(self):
        self.assertEqual(ahp.RI, {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45})
        for n in range(3, 10):
            w = np.arange(1, n + 1, dtype=float)
            self.assertAlmostEqual(ahp.consistency(np.outer(w, 1 / w))["CR"], 0.0, places=9)

    def test_cli_validate(self):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(pb.main([str(pb.EXAMPLE_PROBLEM_PATH)]), 0)
        self.assertIn("task performance: 3 criteria, 3 questions", out.getvalue())


class PanelAndKeyTests(unittest.TestCase):
    def test_default_models_file_equals_study_panel(self):
        panel = config.load_panel()
        self.assertEqual(panel["models"], config.MODELS)
        self.assertEqual(panel["judge"], config.JUDGE)
        self.assertEqual((panel["temperature"], panel["max_tokens"], panel["default_reasoning"]),
                         (config.TEMPERATURE, config.MAX_TOKENS, config.DEFAULT_REASONING))
        for m in config.MODELS:
            self.assertEqual(config.reasoning_for(m, panel["models"], panel["default_reasoning"]),
                             config.reasoning_for(m))

    def test_custom_models_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            path.write_text(json.dumps({"models": {"a": {"id": "x/a", "reasoning": None}},
                                        "judge": {"id": "x/j"}, "defaults": {"temperature": 0.2}}), encoding="utf-8")
            panel = config.load_panel(path)
        self.assertEqual(panel["models"]["a"]["env"], config.SHARED_KEY_ENV)
        self.assertEqual(panel["models"]["a"]["developer"], "x")
        self.assertIsNone(config.reasoning_for("a", panel["models"]))
        self.assertEqual(panel["judge"]["key"], "j")
        self.assertEqual(panel["temperature"], 0.2)

    def test_shared_key_fallback(self):
        with mock.patch.dict(os.environ, _no_openrouter_env(), clear=True):
            self.assertIsNone(client.api_key_for(config.MODELS["glm-5"]))
            os.environ["OPENROUTER_API_KEY"] = "shared-value"
            for m in config.MODELS.values():
                self.assertEqual(client.api_key_for(m), "shared-value")
            self.assertEqual(client.api_key_for(config.JUDGE), "shared-value")
            os.environ["OPENROUTER_KEY_GLM_5"] = "own-value"
            self.assertEqual(client.api_key_for(config.MODELS["glm-5"]), "own-value")
            os.environ["OPENROUTER_KEY_JUDGE"] = "judge-value"
            self.assertEqual(client.api_key_for(config.JUDGE), "judge-value")


class NewProblemTests(unittest.TestCase):
    def test_template_written_and_valid(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("sys.stdout", new_callable=io.StringIO):
            out = Path(d) / "My-Problem.json"
            self.assertEqual(new_problem.main(["--out", str(out)]), 0)
            p = pb.load_problem(out)
            self.assertEqual(p.name, "my_problem")
            self.assertTrue(p.has_ranking)
            self.assertIn("_help", json.loads(out.read_text(encoding="utf-8")))
            with mock.patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(new_problem.main(["--out", str(out)]), 2)
            self.assertEqual(new_problem.main(["--out", str(out), "--force", "--name", "other"]), 0)
            self.assertEqual(pb.load_problem(out).name, "other")

    def test_template_independent_copies(self):
        a = new_problem.template()
        b = copy.deepcopy(a)
        pb.problem_from_dict(a)
        self.assertEqual(a, b)  # loading does not mutate the input


if __name__ == "__main__":
    unittest.main()
