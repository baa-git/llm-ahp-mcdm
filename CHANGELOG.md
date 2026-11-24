# Changelog

## 1.0.0 (2026-11-24)

The workflow now starts from a raw data set.

- `ahp-bench eda`: profile of a CSV (encodings utf-8 / utf-8-sig / cp1254, delimiters, decimal commas, empty
  cells, comment lines): types, missing values, distinct counts, numeric summaries, category frequencies,
  candidate alternative columns and candidate indicators; `eda.md` and `eda.json`.
- Criteria spec (JSON) and `ahp-bench criteria template / validate / build`: indicators computed from the data
  (mean, sum, median, min, max, count, share, ratio, mean_days_between; row filters; scale and offset),
  a goal -> main criteria -> sub-criteria hierarchy with one indicator per leaf, and a problem file in the
  existing schema with a `provenance` block (data file and hash, rows, spec hash, proposer,
  `criteria_confirmed_by_user`). Weighting commands warn when the criteria are not confirmed.
- Criteria proposal by an LLM: `ahp-bench host criteria-prompt / criteria-ingest` (subagent) and
  `ahp-bench criteria propose` (OpenRouter, budget cap, dry run). Prompt, raw reply and parsed spec are
  logged; weight or importance fields in a reply are removed.
- `ahp-bench hierarchy`: text tree, `hierarchy.md` (Mermaid diagram, indicator table) and a grayscale
  `hierarchy.png`. The generic report now opens with the hierarchy, the indicator table (H01), the
  performance matrix (H02) and the data summary.
- Guidance levels for problems other than the study: `described` (neutral criterion descriptions, importance
  cues refused unless `--allow-cues`) and `guided` (the user's own priority order and hints), each zero-shot
  or CoT; `ahp-bench guidance` drafts (code, subagent or API model), stores and confirms the blocks; records
  carry `guidance_level`, `technique` and block hashes; the report adds the compliance table G13 and a note
  when a guided condition was run.
- `ahp-bench rank`: TOPSIS / ARAS / EDAS with weights the user already has.
- `ahp-bench example`: bundled synthetic supplier-order data set (seeded generator) and criteria spec;
  the walk-through from CSV to ranking is tested offline in host mode and as an API dry run.
- Skill rewritten around stages 0-6 with confirmation points, subagent orchestration rules and entry points.
- Unchanged: record schema of existing fields, study prompts and prompt hashes, behaviour of existing commands;
  the study audit (86/86) and the regenerated study tables and report are byte-identical.

## 0.2.0

- Study data moved to the separate repository llm-ahp-mcdm-study; `--results` / `--data` point to it.
- Direct condition naming (`direct` for the stored key `lean`); ranking agreement table.

## 0.1.0

- First release: LLM panel AHP weighting (host mode and OpenRouter), parsing, consistency, revision with an
  optional judge, TOPSIS / ARAS / EDAS ranking, reports, audit and the agent skill.
