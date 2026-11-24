# Prompt texts

The code builds every prompt (`ahp_bench/prompts.py`, `ahp_bench/revise.py`); do not write prompts by
hand. Placeholders: `<role>`, `<goal>` (from the problem file), `<N>` (number word Three..Nine),
`<parent>` (main criterion of a sub-criteria group). With the study problem the direct and CoT messages are
byte-identical to the ones stored in the study's logs (the audit checks every record).

## direct: unguided, zero-shot (stored key `lean`)

System message:

```
<role>
Answer the following questions. Respond only in the requested JSON format.
```

User message (main criteria; for a sub-criteria group the first line is
`<N> sub-criteria of "<parent>" are compared in pairs.` and the question says "two sub-criteria"):

```
<N> criteria used in <goal> are compared in pairs.

For each question: "Which of the following two criteria is more important to you?"

Answer scale (left criterion = A, right criterion = B):
9   : A is extremely more important than B.
8   : A is between very strongly and extremely more important than B. (Intermediate value)
7   : A is very strongly more important than B.
6   : A is between strongly and very strongly more important than B. (Intermediate value)
5   : A is strongly more important than B.
4   : A is between moderately and strongly more important than B. (Intermediate value)
3   : A is moderately more important than B.
2   : A is between equally and moderately more important than B. (Intermediate value)
1   : A and B are equally important.
1/2 : B is between equally and moderately more important than A. (Intermediate value)
1/3 : B is moderately more important than A.
1/4 : B is between moderately and strongly more important than A. (Intermediate value)
1/5 : B is strongly more important than A.
1/6 : B is between strongly and very strongly more important than A. (Intermediate value)
1/7 : B is very strongly more important than A.
1/8 : B is between very strongly and extremely more important than A. (Intermediate value)
1/9 : B is extremely more important than A.

Questions:
1. Left: <criterion 1> | Right: <criterion 2>
...

Response format (only this JSON):
{"answers": [{"question": 1, "value": "3"}, {"question": 2, "value": "1/5"}, ...]}
```

## CoT: unguided, chain-of-thought (stored key `cot`)

CoT (chain-of-thought) is also unguided as to the weights: it only adds a step-by-step reasoning instruction to the direct
(zero-shot) prompt.

Same system message; the user message replaces the last block (response format) with:

```
Before answering, think step by step:
1. State in one sentence what each criterion represents in <goal>.
2. Rank the criteria from most to least important in your view.
3. Choose each value so that it is consistent with this ranking
   (for example, if A is more important than B and B than C, A must be even more important than C).

Response format (only this JSON):
{"reasoning": "your brief reasoning for steps 1-3",
 "ranking": ["most important criterion", "...", "least important criterion"],
 "answers": [{"question": 1, "value": "3"}, ...]}
```

## described and guided (problems other than the study)

Same system message. The user message inserts, after the intro line and the question line, the problem
file's neutral descriptions (`references/guidance.md`):

```
The criteria:
- <criterion 1>: <description>
...
```

(`The sub-criteria:` for a group). The guided level adds, right after that block, the user's priorities
from the `guidance` block (only the parts given for the task):

```
Priorities stated by the decision maker:
Order of importance, most important first: 1. <criterion>, 2. <criterion>, ...
- <hint in the user's words>
```

Then the scale, the questions and the direct or CoT response format, exactly as above. The direct and CoT
messages never contain descriptions or priorities, whatever the problem file holds.

## Criteria proposal (stage 1; `ahp-bench host criteria-prompt`, `ahp-bench criteria propose`)

System message:

```
You help structure a multi-criteria decision problem from a tabular data set. You propose decision criteria and the indicators that measure them, using only the columns described. Respond only in the requested JSON format.
```

The user message names the goal (or "a decision"), the data file and row count, one line per column from
the EDA (type, missing, distinct, numeric summary, date range, category frequencies; no values per
alternative), the candidate alternative columns, then these rules and the response format:

```
Rules:
1. Propose only criteria that the columns above can measure. Each indicator is computed per alternative with one of these aggregations: mean, sum, median, min, max (of a numeric column); count (number of rows, or of non-empty cells of a column); share (share of rows where a column equals a value); ratio (sum of one numeric column divided by the sum of another); mean_days_between (mean number of days between two date columns). An optional row filter restricts the rows used: {"column": "...", "equals": "..."} (also not_equals, in, not_in, gt, ge, lt, le).
2. For every indicator state its direction ("benefit" = higher values are better, "cost" = lower values are better), its unit and a one-sentence rationale that names the columns used.
3. Do not assign weights, importance levels or a priority order, and do not say which criterion matters more. The order in which you list the criteria carries no meaning. Weights are elicited later in a separate pairwise-comparison step.
4. Arrange the criteria as an AHP hierarchy: 3 to 9 main criteria; a main criterion may have 3 to 9 sub-criteria. Every leaf criterion (a main criterion without sub-criteria, or a sub-criterion) maps to exactly one indicator, and every indicator to exactly one leaf criterion.
5. Use neutral criterion names without evaluative words. Do not use identifier columns or the alternative column as indicators.
6. If the columns cannot support at least 3 criteria, return the criteria they can support and explain the gap in "notes".
```

Reply format: `{"alternative_column", "goal", "indicators": [...], "hierarchy": {"criteria": [...]}, "notes"}`
(full text in `ahp_bench/criteria.py`, `CRITERIA_FORMAT`). Prompt id `criteria_proposal_v1`.

## Descriptions draft (`ahp-bench guidance descriptions-prompt / descriptions-propose`)

System: `You write neutral descriptions of decision criteria for pairwise-comparison questions. Respond only in
the requested JSON format.` The user message lists every criterion with its indicator, unit, direction and
source, and asks for one or two sentences each without importance or rank words, without comparisons and
without mentioning the alternatives; reply `{"descriptions": {"<criterion>": "..."}}`.

## Revision message (code-written)

```
The comparison matrix built from your answers has a consistency ratio CR = <cr, 3 decimals>.
The accepted limit is 0.10; your answers contradict each other.

Most contradictory comparisons:
Questions <a>, <b> and <c>: <X> vs <Y> = <v>, <Y> vs <Z> = <v>, but <X> vs <Z> = <v>
Questions <a>, <b> and <c> form a circular preference: <X> over <Y>, <Y> over <Z>, <Z> over <X>.

Your previous answers:
Q1: Left <criterion> | Right <criterion> = <value>
...

Review your answers. You do not have to change your view; make only the changes needed
to remove the contradiction.
Use only these values: 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9.
You may change the direction of an answer if that is what resolves a contradiction.
Answer all questions again in the same JSON format.
```

## Judge (with `--judge`)

Feedback call, system message:

```
You coordinate an AHP elicitation. You never tell the respondent which criterion should be more important; you only point out contradictions among their own answers. Respond only in JSON.
```

The user message lists the criteria, a note on the scale, the respondent's current answers, the CR, the
code-computed contradictions, the respondent's own stated ranking (CoT only) and this instruction:

```
Write a short message (max 120 words) to the respondent that explains which of their answers contradict each other and asks them to reconsider those answers changing only the answers that need to change, keeping their own view of what matters. Do not suggest values. Do not state or imply which criterion deserves more weight. If the list contains a circular preference, say so explicitly and state that at least one of those three answers must move toward the opposite direction; do not say which one. JSON: {"feedback": "..."}
```

The panel model then receives the judge's feedback followed by the code-written previous answers, the
allowed values, the direction sentence and "Answer all questions again in the same JSON format."

Verdict call, system message `You evaluate answer revisions in an AHP elicitation. Respond only in JSON.`;
the user message shows the original and revised answers with their implied rankings and CRs, and:

```
Judge whether the revised answers preserve the logic of the original answers (same view of what matters most and least) or whether priorities were changed merely to reach numerical consistency. JSON: {"logic_preserved": "yes" | "partly" | "no", "changed_priorities": ["..."], "comment": "max 60 words"}
```

## Why the study's guided prompts are not reused for new problems

The study also ran a `guided` condition: the original thesis prompts (in `problems/guided/`), which
contain a typical priority order ("Quality - HIGHEST ... Volume - LOWER"), criterion descriptions in
Turkish that signal importance (for example "çok kritik", very critical, for quality and "orta önemde",
of moderate importance, for volume), the full data-analysis report on the alternatives, and the rule
"never use all 1s". Those matrices were almost always consistent and moved the
weights toward the hinted order (Quality +14.65 pp above the expert weight versus +3.22 pp for direct; see
`pitfalls.md`). A hinted prompt measures how well a model follows the hint, not its own judgement, and
showing the alternatives' data mixes stage 1 (weights) with stage 2 (ranking). New problems get the
neutral direct and CoT prompts, the neutral described level and, only with the user's own priorities, the
generic guided level (`guidance.md`); never add hints, descriptions of importance or the alternatives' data
to the direct or CoT questions.
