# Contributing

Thank you for your interest. Issues and pull requests are welcome.

- Keep the dependencies to the Python standard library and numpy (matplotlib only for figures). Stages 0-2
  (`tabular.py`, `eda.py`, `criteria.py`, `hierarchy.py`) use the standard library only; do not add pandas.
- Run the offline tests before submitting: `python -m unittest discover -s tests`. New code needs tests;
  model calls in tests go through a fake transport or canned host-mode replies, never the network.
- Changes to `prompts.py`, `parse.py`, `ahp.py`, `analysis.py`, `revise.py` or `report.py` must keep the
  published study reproducible: with a clone of https://github.com/baa-git/llm-ahp-mcdm-study,
  `python -m ahp_bench audit --results <study>/results --out <tmp>` must report 0 failed checks and
  `python -m ahp_bench report --results <study>/results --data <study>/data --tag main --revision-tag revision_final --out <tmp>`
  must reproduce the published tables (or the change must be released together with regenerated study files).
  Run the tests once with `AHP_STUDY_DIR=<study>` as well.
- Never commit API keys or a `.env` file; the audit's secret scan must stay clean.
- Changes to the skill (`skills/llm-ahp-mcdm/`) must keep `SKILL.md` free of tool-specific syntax so that
  different agents can follow it, and keep the rules on budgets, keys, neutral prompts and host-mode
  orchestration.
- New problem examples must be labelled clearly if their numbers are illustrative.
