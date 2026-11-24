---
name: llm-ahp-mcdm
description: Take a decision from a raw tabular data set to a ranking with the Analytic Hierarchy Process (AHP) - exploratory data analysis, candidate criteria and an AHP hierarchy, an alternatives x indicators performance matrix, pairwise-comparison weights from LLM respondents (independent subagents of your own Claude Code or Codex session, or OpenRouter models), consistency checks and revision, TOPSIS / ARAS / EDAS rankings and an audited report. Use when a user wants to turn a CSV into a multi-criteria decision (MCDM), wants LLM-elicited AHP weights, already has criteria or weights and wants rankings, or wants to verify the published llm-ahp-mcdm study.
---

# LLM-AHP-MCDM: from a data set to a ranking

This skill drives the `ahp-bench` command. AHP is the core: weights come only from pairwise comparisons
answered by respondents; code turns answers into matrices, eigenvector weights and consistency ratios (CR).
Everything is logged; every number in a report is recomputed from the logs.

| Stage | What | Command | Show the user | Ask the user to confirm |
| --- | --- | --- | --- | --- |
| 0 | EDA of the data set | `ahp-bench eda` | rows, columns, types, missing values, candidate alternative column | the alternative column and the decision goal |
| 1 | criteria and hierarchy | `ahp-bench criteria template / validate`, `ahp-bench host criteria-prompt / criteria-ingest`, `ahp-bench criteria propose` | criteria list, directions, rationale, hierarchy (`ahp-bench hierarchy`) | criteria list, every direction, the hierarchy |
| 2 | performance matrix | `ahp-bench criteria build` | the alternatives x indicators matrix | the matrix; then record the confirmation |
| 3 | pairwise comparisons and weights | `ahp-bench host prompts / ingest` or `ahp-bench run` | one prompt, the plan (and cost in API mode) | mode, panel size, conditions, budget |
| 4 | consistency and revision | `ahp-bench report`, `ahp-bench host revise-prompts / revise-ingest` or `ahp-bench revise` | CR per matrix and condition | whether to revise |
| 5 | ranking | `ahp-bench report` (or `ahp-bench rank` with given weights) | TOPSIS / ARAS / EDAS ranks, equal-weight baseline, sensitivity | - |
| 6 | report and audit | `ahp-bench report`, `ahp-bench audit` | `report.md` paths, audit result, limitations | - |

References: `references/data-workflow.md` (stages 0-2: reading the EDA, spec format, aggregations, data
without alternatives), `references/guidance.md` (prompt guidance levels), `references/method.md` (formulas),
`references/prompts.md` (exact prompt texts), `references/pitfalls.md` (findings of the published study).

## Setup (once)

1. `ahp-bench --version` (1.0.0 or later). If missing, ask the user to run
   `pip install git+https://github.com/baa-git/llm-ahp-mcdm@v1.0.0` (Python 3.10+; numpy; matplotlib optional
   for figures), or use a clone and replace `ahp-bench <cmd>` by `python -m ahp_bench <cmd>` from its root.
2. Work in a folder of the user's choosing. Everything except API mode is offline.

## Five-minute walk-through (synthetic example, no key, no cost)

Do this first with a new user. Say clearly that the data are invented and, for a pure dummy, that the answers
are canned.
1. `ahp-bench example --out demo` and `cd demo` - copies `orders_synthetic.csv` (400 invented supplier orders),
   `orders_spec.json` and the generator script.
2. `ahp-bench eda --data orders_synthetic.csv --out eda` - show the column table and the candidate
   alternative column (`supplier`) from `eda/eda.md`.
3. `ahp-bench criteria validate --spec orders_spec.json --data orders_synthetic.csv` - show the tree and the
   matrix it prints (4 main criteria, Delivery with 3 sub-criteria, 6 indicators, 5 suppliers).
4. `ahp-bench criteria build --spec orders_spec.json --data orders_synthetic.csv --out problems/orders.json --confirmed`
   (in the demo the user confirms on the spot), then `ahp-bench hierarchy --problem problems/orders.json --out hierarchy`
   and show `hierarchy/hierarchy.png` or the Mermaid diagram in `hierarchy/hierarchy.md`.
5. `ahp-bench host prompts --problem problems/orders.json --conditions direct --panelists 3 --out work`; answer
   each `work/prompts/*.txt` with a fresh subagent (stage 3), or for a pure dummy write obviously artificial
   replies `{"answers": [{"question": 1, "value": "1"}, ...]}` and label the run "dummy".
6. `ahp-bench host ingest --problem problems/orders.json --dir work --results results/demo --tag main --model-name <model or dummy>`
7. `ahp-bench report --problem problems/orders.json --results results/demo` and
   `ahp-bench audit --problem problems/orders.json --results results/demo`; show `results/demo/report/report.md`
   (hierarchy, H01 indicators, H02 matrix, G02 consistency, G04 weights, G07-G09 rankings).

## Rules (every stage)

- **Never fabricate.** Do not invent data, criteria values, answers, weights, matrices or rankings; report only
  numbers from generated files, with paths.
- **Weights only from the pairwise step.** Criteria proposals, descriptions and EDA carry no weights or
  importance order. Never rank criteria for the user.
- **Confirm before weighting.** Stage 3 starts only after the user confirmed the criteria list, every
  direction, the hierarchy and the performance matrix (record it: `criteria build --confirmed` or
  `ahp-bench guidance confirm --criteria`). Weighting commands warn when this is missing; do not ignore it.
- **No hints in direct/CoT prompts.** The problem file holds neutral names, goal and role. Guidance levels
  (descriptions, priorities) are separate conditions, see `references/guidance.md`.
- **Logs are data.** Never edit or delete files under `<results>/raw/`; inconsistent matrices are kept.
- **Keys and money.** Never ask for an API key in the conversation; never open, print or log `.env`. Every
  paid call: `--dry-run` first, state calls, estimate and `--budget`, wait for an explicit yes.
- **Wording.** Agreement with expert weights is a similarity, not a quality measure: never call a condition
  better or worse for being closer to experts.

## Entry points

- Raw data set: stage 0.
- Criteria known, no data: `ahp-bench new-problem --out problems/<name>.json`, fill it in (ranking block
  optional), `ahp-bench problem problems/<name>.json`, `ahp-bench hierarchy --problem ...`; confirm; stage 3.
- Criteria and data known: write the spec yourself with the user (`references/data-workflow.md`); stage 1 step 3.
- Weights known: a problem file with a performance matrix plus `ahp-bench rank --problem P --weights W.json`
  (leaf or per-task weights); stage 5. Say that the weights are the user's, not elicited.

## Stage 0: EDA

`ahp-bench eda --data <file> --out eda [--alternative-column COL]`. Show from `eda/eda.md`: rows, columns,
types, missing values, notes, candidate alternative columns with rows per alternative. Ask which column names
the alternatives and what the decision is (goal noun phrase, respondent role). The suggested directions come
from column names only; say so.

## Stage 1: criteria and hierarchy

1. Get candidate criteria in one of three ways (offer them; the user chooses):
   - the user dictates them;
   - `ahp-bench criteria template --eda eda/eda.json --out spec.json` - a DRAFT from the EDA candidates,
     flat, with `REVIEW` placeholders for every open direction, the goal and the role; edit it with the user;
   - an LLM proposal. Subscription mode: `ahp-bench host criteria-prompt --eda eda/eda.json --out crit [--goal ...]`,
     give `crit/criteria_prompt.txt` verbatim to ONE fresh subagent, save its reply as `crit/criteria_reply.json`,
     then `ahp-bench host criteria-ingest --dir crit --data <file> --out spec.json --name <name>`. API mode:
     `ahp-bench criteria propose --eda eda/eda.json --data <file> --out spec.json --model <id> --budget <USD> --dry-run`,
     approval, then without `--dry-run`. The prompt forbids weights and importance order; any weight field in
     a reply is removed and logged (`criteria_calls.jsonl`).
2. `ahp-bench criteria validate --spec spec.json --data <file>` until it passes. Fix errors with the user,
   never by guessing a direction.
3. SHOW: the criteria list with direction, unit, source column, aggregation and rationale, and the hierarchy
   (the tree printed by validate; after stage 2 also `ahp-bench hierarchy --problem P --out hierarchy`).
   ASK the user to confirm or change criteria, directions and hierarchy. Set `proposed_by` truthfully
   (`human` only if a person wrote or rewrote the criteria).

## Stage 2: performance matrix

`ahp-bench criteria build --spec spec.json --data <file> --out problems/<name>.json` writes the problem file
(existing schema plus `provenance`: data file, sha256, rows, spec hash, proposer, confirmation). SHOW the
matrix it prints (alternatives x indicators, with directions). After the user confirms criteria, hierarchy
and matrix, rebuild with `--confirmed` (or `ahp-bench guidance confirm --problem P --criteria`).

## Stage 3: pairwise comparisons and weights

Conditions (technique and guidance are independent): `direct` (default; criterion names only), `cot`
(step-by-step reasoning), `described` / `described-cot` (plus neutral descriptions), `guided` / `guided-cot`
(plus the user's own priorities). Use `direct`; add `cot` on request. Run `guided` only together with an
unguided condition (suggested: `--conditions direct,guided`) and read `references/guidance.md` first: the
priority order must be the user's own words, shown back and confirmed; never invent one.

Mode A, subscription (Claude Code or Codex, no key):
1. `ahp-bench host prompts --problem P --conditions direct --panelists 5 --out work` (5 or more per condition).
2. Orchestration: for every file in `work/prompts/` (not README.txt) start ONE fresh subagent (Claude Code:
   Agent/Task tool; Codex: its subagent or parallel-task mechanism, else sequential fresh contexts). Give the
   file content verbatim and nothing else; ask for the JSON object only; save the reply unchanged as
   `work/replies/<same name>.json`. Panelists are independent: one prompt each, never your own answers, never
   one panelist's answers or results shown to another, no edits to replies.
3. `ahp-bench host ingest --problem P --dir work --results results/<name> --tag main --model-name <your model>`;
   a parse failure is recorded - give the same prompt to a new subagent and ingest again.
4. Limits to tell the user: one underlying model; temperature and reasoning cannot be set; panelists are
   repetitions of that model, not different models; no token or cost records.

Mode B, API (OpenRouter, the user's key in the environment or a `.env` they create):
1. `models.json` is the study's nine-model panel; pass `--models-file` for another.
2. `ahp-bench run --problem P --conditions direct --reps 3 --budget <CAP> --tag main --dry-run --print-prompts`;
   show prompts, call count and estimate; wait for a yes; run without `--dry-run`. Limits: model versions,
   provider routing and prices change; some models ignore temperature or cannot disable reasoning
   (`references/pitfalls.md`).

## Stage 4: consistency and revision

`ahp-bench report --problem P --results results/<name>`; from G02 report matrices with CR <= 0.10 per
condition, mean and maximum CR and the group CR. Revision is optional; explain first that it often reaches
CR <= 0.10 but changes priorities, so revised weights are a different measurement (report both).
- Mode A: `ahp-bench host revise-prompts --problem P --dir work --results results/<name> --source-tag main --tag revision`,
  one fresh subagent per `work/revise/*.txt` into `work/revise_replies/`, then
  `ahp-bench host revise-ingest --problem P --dir work --results results/<name> --tag revision --model-name <model>`;
  repeat until no prompt is written. `--judge` only if the user asks (a separate subagent per `work/judge/*.txt`).
- Mode B: `ahp-bench revise --problem P --source-tag main --tag revision --budget <CAP> --dry-run`, approval, run.
- `ahp-bench report ... --revision-tag revision`: G10-G12.

## Stage 5: ranking

From the report: G07 (TOPSIS, ARAS, EDAS per condition), G08 (equal weights), G09 (sensitivity). Report
whether the methods agree, whether the top alternative differs from the equal-weight top, and from which
weight level it changes. With guided runs, G13 shows how closely the replies followed the stated order.

## Stage 6: report, audit, summary

`ahp-bench audit --problem P --results results/<name>`: every check in `report/audit/audit.md` must pass;
quote any FAIL. Summarise with paths (data file, spec, problem file, `raw/calls.jsonl`, `report/report.md`,
hierarchy, audit), the mode, models or host model, cost (API mode), who proposed the criteria and that the
user confirmed them, and the limitations: LLM answers are not expert judgement; few repetitions; results
depend on models, prompts and dates; revision changes priorities; guided weights restate the guide's
priorities; no human reference unless expert matrices were given (`--experts FILE`).

## When the data set has no clear alternatives or fewer than 3 criteria

- No alternative column (EDA note): ask what is being chosen. If rows are themselves the alternatives (one
  row each), add or name an identifying column and use `max` (or `mean`) of each indicator. If the data only
  describe one entity over time, AHP weights can still be elicited (stage 3 without a ranking block), but
  there is nothing to rank; say so.
- Fewer than 3 usable criteria: an AHP matrix needs 3-9 per level. Ask whether derived indicators (rates,
  durations, shares) or criteria measured outside the data set should be added; never pad with irrelevant
  columns. With only 2 criteria, a single direct trade-off question is simpler than AHP; say so.
- Fewer than 3 alternatives: ranking works from 2, but sensitivity results are thin; say so.

## Verifying the published study

Records are in https://github.com/baa-git/llm-ahp-mcdm-study (no calls): `ahp-bench audit --results <study>/results`
(86 checks pass) and `ahp-bench report --results <study>/results --data <study>/data --tag main --revision-tag revision_final --out <DIR>`
reproduces the published tables byte for byte.
