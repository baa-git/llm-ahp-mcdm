"""The skill folder: frontmatter, referenced files and commands, installer (offline)."""
import importlib.util
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "llm-ahp-mcdm"


def _installer():
    spec = importlib.util.spec_from_file_location("install_skill", ROOT / "scripts" / "install_skill.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SkillTests(unittest.TestCase):
    def test_frontmatter_and_size(self):
        text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        self.assertIsNotNone(m)
        fields = dict(line.split(":", 1) for line in m.group(1).splitlines())
        self.assertEqual(fields["name"].strip(), SKILL.name)
        self.assertTrue(20 < len(fields["description"].strip()) <= 1024)
        self.assertLessEqual(len(text.splitlines()), 500)  # Agent Skills guidance: keep SKILL.md under 500 lines

    def test_references_exist_and_commands_are_real(self):
        text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        for ref in re.findall(r"`(references/[\w.-]+)`", text):
            self.assertTrue((SKILL / ref).exists(), ref)
        for mod in set(re.findall(r"python -m ahp_bench\.(\w+)", text)):
            self.assertTrue((ROOT / "ahp_bench" / f"{mod}.py").exists(), mod)
        from ahp_bench.__main__ import COMMANDS
        cmds = set(re.findall(r"ahp-bench ([a-z][a-z-]*)", text))
        self.assertTrue({"host", "report", "audit", "run", "revise", "problem", "new-problem", "example", "eda",
                         "criteria", "hierarchy", "rank", "guidance"} <= cmds, cmds)
        self.assertTrue(cmds <= set(COMMANDS), cmds - set(COMMANDS))
        for sub in set(re.findall(r"ahp-bench host ([a-z-]+)", text)):
            self.assertIn(sub, ("prompts", "ingest", "revise-prompts", "revise-ingest", "criteria-prompt",
                                "criteria-ingest"))
        for sub in set(re.findall(r"ahp-bench criteria ([a-z-]+)", text)):
            self.assertIn(sub, ("template", "validate", "build", "propose"))
        for ref in SKILL.joinpath("references").glob("*.md"):
            self.assertIn(f"references/{ref.name}", text)  # every reference file is linked from SKILL.md
        self.assertTrue((ROOT / "AGENTS.md").read_text(encoding="utf-8").count("skills/llm-ahp-mcdm/SKILL.md"))

    def test_installer_copies_folder(self):
        inst = _installer()
        with tempfile.TemporaryDirectory() as d, mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(inst.main([d]), 0)
            dest = Path(d) / "llm-ahp-mcdm"
            self.assertTrue((dest / "SKILL.md").exists())
            self.assertTrue((dest / "references" / "pitfalls.md").exists())
            with mock.patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(inst.main([d]), 2)
            self.assertEqual(inst.main([d, "--force"]), 0)
            other = Path(d) / "dry"
            self.assertEqual(inst.main([str(other), "--dry-run"]), 0)
            self.assertFalse(other.exists())

    def test_install_targets(self):
        from ahp_bench import install_skill
        with tempfile.TemporaryDirectory() as d, mock.patch("sys.stdout", new_callable=io.StringIO),                 mock.patch("pathlib.Path.home", return_value=Path(d) / "home"),                 mock.patch("pathlib.Path.cwd", return_value=Path(d) / "proj"):
            (Path(d) / "proj").mkdir()
            self.assertEqual(install_skill.main(["--codex"]), 0)
            self.assertTrue((Path(d) / "home" / ".codex" / "skills" / "llm-ahp-mcdm" / "SKILL.md").exists())
            self.assertEqual(install_skill.main(["--claude"]), 0)
            self.assertTrue((Path(d) / "home" / ".claude" / "skills" / "llm-ahp-mcdm" / "SKILL.md").exists())
            self.assertEqual(install_skill.main(["--project"]), 0)
            self.assertTrue((Path(d) / "proj" / ".claude" / "skills" / "llm-ahp-mcdm" / "references").is_dir())
            agents = (Path(d) / "proj" / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn(".claude/skills/llm-ahp-mcdm/SKILL.md", agents)
            self.assertEqual(install_skill.main(["--project", "--force"]), 0)
            self.assertEqual((Path(d) / "proj" / "AGENTS.md").read_text(encoding="utf-8"), agents)  # hint once

    def test_console_entry_and_plugin_manifests(self):
        import json
        from ahp_bench import __main__ as entry
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(entry.main([]), 0)
        self.assertIn("install-skill", out.getvalue())
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(entry.main(["nope"]), 2)
        plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
        market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
        self.assertEqual(plugin["name"], SKILL.name)
        self.assertEqual(market["plugins"][0]["name"], SKILL.name)
        self.assertEqual(market["plugins"][0]["source"], "./")
        from ahp_bench import __version__
        self.assertEqual(plugin["version"], __version__)
        toml = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(f'version = "{__version__}"', toml)
        self.assertIn('ahp-bench = "ahp_bench.__main__:main"', toml)
        cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn(f'version: "{__version__}"', cff)


if __name__ == "__main__":
    unittest.main()
