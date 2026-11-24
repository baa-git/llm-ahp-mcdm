# Instructions for coding agents

This repository contains an agent skill in the open Agent Skills format.

- When the user asks to turn a data set (CSV) into a multi-criteria decision, for an AHP / MCDM analysis with
  LLM respondents, for LLM-elicited criteria weights, for TOPSIS / ARAS / EDAS rankings, or to verify the
  published study, read `skills/llm-ahp-mcdm/SKILL.md` and follow its staged workflow (0 EDA, 1 criteria and
  hierarchy, 2 performance matrix, 3 weights, 4 consistency, 5 ranking, 6 report). Its `references/` folder
  holds the data workflow, guidance levels, formulas, prompt texts and the study's pitfalls.
- Show the user the EDA, the criteria with their directions, the hierarchy and the performance matrix, and get
  their confirmation before any weighting. Never assign weights or an importance order yourself; a guided
  priority order must be the user's own words.
- Run commands from the repository root (`python -m ahp_bench <command>`, or `ahp-bench <command>` after
  `pip install`). Tests are offline: `python -m unittest discover -s tests`.
- In subscription (host) mode you orchestrate only: give each prompt file verbatim to a fresh subagent, one
  prompt per subagent, never answer the questions yourself and never show one subagent another's answers.
- Never ask the user to paste an API key into the conversation, and never print, log or open `.env` or any
  key. Keys come from environment variables (`OPENROUTER_API_KEY` or the names in `models.json`).
- Every command that calls a model through the API costs money: run it with `--dry-run` first, state the
  planned calls, the estimated cost and the `--budget` cap, and wait for the user's explicit approval.
- The study's records are in https://github.com/baa-git/llm-ahp-mcdm-study; never modify files under any
  `raw/` results folder.
