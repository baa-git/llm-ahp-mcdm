# Prompt guidance levels (problems other than the study)

Technique (zero-shot or CoT) and guidance are independent; every level exists with both. Code:
`ahp_bench/guidance.py`, `ahp_bench/prompts.py`.

| Level | Condition (command line) | Stored key | Prompt content |
| --- | --- | --- | --- |
| unguided | `direct`, `cot` | `lean`, `cot` | criterion names only (default) |
| described | `described`, `described-cot` | `described`, `described_cot` | names plus neutral descriptions |
| guided | `guided`, `guided-cot` | `user_guided`, `user_guided_cot` | descriptions plus the user's priority order and hints |

The study problem keeps its own `guided` condition (verbatim thesis prompt files); its records, prompts and
hashes are unchanged.

## Described: neutral descriptions

Stored in the problem file:

```json
"descriptions": {"drafted_by": "ahp-bench (from the indicator table)", "confirmed_by_user": false,
                 "criteria": {"Cost": "Measured by Mean unit price (USD per unit); lower values are better; ..."}}
```

One description per criterion of the hierarchy (main and sub-criteria): what it measures, unit, direction,
source indicator. Drafting options:
- `ahp-bench guidance draft-descriptions --problem P` - written by code from the indicator table (neutral by
  construction);
- subscription mode: `ahp-bench guidance descriptions-prompt --problem P --out DIR`, one fresh subagent,
  reply to `DIR/descriptions_reply.json`, `ahp-bench guidance descriptions-ingest --problem P --dir DIR`;
- API mode: `ahp-bench guidance descriptions-propose --problem P --model ID --budget USD --dry-run`, then
  without `--dry-run`.
Show them, let the user edit, then `ahp-bench guidance confirm --problem P --descriptions`.

Importance cues (important, critical, crucial, key, priority, vital, essential, primary, secondary, major,
minor, top, highest, lowest, rank..., significant, "more/less/most/least important", "matters most", and the
Turkish önem..., kritik..., öncelik...) are flagged by `ahp-bench guidance check`. The described condition
refuses to run with cues unless `--allow-cues` is given; the choice is stored per record (`cues_allowed`).

## Guided: the user's priorities

```json
"guidance": {"source": "user", "confirmed_by_user": true, "user_statement": "Price matters most to me, then speed.",
             "order": {"main": ["Price", "Performance", "Portability"]},
             "hints": {"performance": ["Memory matters for my statistics courses."]}}
```

Rules:
- The order and the hints come from the user's own stated priorities. Ask the user; quote their words in
  `user_statement`; turn them into the block with
  `ahp-bench guidance set-order --problem P --task main --order "A | B | C" --source user --user-statement "..."`
  (and `--hint` in the user's words); show the block (`ahp-bench guidance check --problem P`) and record the
  confirmation (`ahp-bench guidance confirm --problem P --guidance`).
- Never invent an order, never derive one from the data, from the descriptions or from your own view. An
  order an agent drafted is `--source orchestrator-draft`; weighting with it, or with an unconfirmed block,
  prints a prominent warning and the report carries it.
- Run guided only next to an unguided condition (`--conditions direct,guided`). Weights obtained under
  guidance largely reproduce the guidance and should be read as the guide author's priorities expressed in
  AHP form, not as independent judgments. The study showed this: the hinted criterion gained +14.65 pp over
  the expert weight with the hint versus +3.22 pp without it (`pitfalls.md`, section 1).

## Records and report

Records of problems other than the study carry `guidance_level` (unguided / described / guided) and
`technique` (zero-shot / CoT); described and guided records add `descriptions_sha256`,
`descriptions_drafted_by`, `descriptions_confirmed_by_user`, and `cues_allowed` (described) or
`guidance_sha256`, `guidance_source`, `guidance_confirmed_by_user` (guided). Report tables label the
conditions direct, CoT, described, described CoT, guided, guided CoT with Guidance and Technique columns.
When a guided condition was run, the report adds the descriptions, the guidance block, the sentence above,
any warning, and G13: per model and panel, the share of replies whose eigenvector ranking equals the prompted
order, the share with the prompted first criterion ranked first and the mean Spearman rho with the prompted
order, with the other conditions as reference rows.
