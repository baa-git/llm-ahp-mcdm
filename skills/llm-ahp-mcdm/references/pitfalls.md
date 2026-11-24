# Pitfalls (findings of the original study)

Study: 9 LLMs via OpenRouter, supplier selection with 5 main criteria and 4 delivery sub-criteria,
conditions guided (zero-shot, guided), direct (zero-shot, unguided) and CoT (chain-of-thought,
unguided), 5 repetitions per cell (270 first-answer calls, run tag `main`,
temperature 0.6), 7 human experts as reference. Numbers are taken from the study's tables in
`results/report/tables/` (Txx) and `results/report/audit/` of the study repository
(https://github.com/baa-git/llm-ahp-mcdm-study). They describe these models
on these dates (November 2026) and this problem; treat them as warnings, not as constants.

## 1. Priority hints inflate the hinted criterion (T04, T03)

The guided prompt ranked Quality first. Panel weight of Quality: guided 0.4651, direct 0.3507, CoT 0.3757,
experts 0.3186, i.e. +14.65 pp above the experts with the hint versus +3.22 pp (direct) and +5.72 pp (CoT)
without it. Guided matrices also looked far more consistent (main criteria: 43/45 with CR <= 0.10 versus
15/45 direct and 24/45 CoT). Consistency obtained by telling the model the order is not evidence of
judgement. Never add hints, importance-laden descriptions or the alternatives' data to the questions.

## 2. Many unguided matrices are inconsistent (T03, T02)

Matrices with CR <= 0.10, first answers:

| Condition | Main (5 criteria) | Delivery (4 sub-criteria) |
| --- | ---: | ---: |
| direct | 15/45 | 23/45 |
| CoT | 24/45 | 29/45 |

Models differ strongly: gpt-4o direct 0/5 (main) and 0/5 (delivery), mean main CR 0.4381; qwen3.5-397b CoT
mean main CR 0.8310; claude-sonnet-4.6 direct 5/5 and 5/5. Humans were inconsistent too: experts 4/7 (main)
and 0/7 (delivery). The geometric-mean group matrices were consistent (group CR main: direct 0.0616, CoT
0.0228; experts 0.0321). Report consistency per model; do not drop inconsistent matrices silently.

## 3. Step-by-step prompting helps only partly (T03)

CoT raised the consistent share of main matrices from 15/45 to 24/45 but not uniformly: qwen3.5-397b went
from 0/5 to 0/5 with a higher mean CR (0.2967 -> 0.8310).

## 4. Revision usually reaches consistency but changes priorities (T16, T20, audit j)

Revision tag `revision_final` (judge x-ai/grok-4.20, at most 5 revisions): 89 inconsistent matrices,
69 reached CR <= 0.10 (direct main: 19 of 30 chains; 11 never did). Comparing final with first answers
(panel rows of T20):

| Condition / task | Chains | Answers changed (mean) | Top criterion unchanged | Full ranking unchanged | Largest weight change (pp, mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct / main | 30 | 4.233 of 10 | 0.80 | 0.40 | 9.06 |
| direct / delivery | 22 | 3.955 of 6 | 0.64 | 0.18 | 16.46 |
| CoT / main | 21 | 5.333 of 10 | 0.62 | 0.19 | 15.86 |
| CoT / delivery | 16 | 4.250 of 6 | 0.94 | 0.38 | 10.91 |

The judge rated logic preservation on all 250 revision calls as yes 45, partly 81, no 124. Treat revised
weights as a different measurement and always report the first answers as well.

## 5. Lower temperature does not remove inconsistency (T18)

direct, panel: CR <= 0.10 for main matrices 13/27 at T = 0.0, 15/45 at T = 0.6, 10/27 at T = 1.0. Mean
coefficient of variation of the main weights across repetitions: 0.148, 0.232, 0.230. Use repetitions
(at least 3) and report run-to-run variation (T09).

## 6. GPT-5.2 ignores the temperature parameter (audit k, param_tests.jsonl)

The request is accepted but not applied: 5 calls at T = 0 gave 5 different answer sets; 5 calls at
T = 1.5 also gave 5 different answer sets. Variability of this model cannot be attributed to the
requested temperature.

## 7. MiniMax M2.5 reasoning cannot be disabled (audit k, T01)

Three requests that tried to switch reasoning off were rejected with HTTP 400 ("Reasoning is mandatory
for this endpoint and cannot be disabled."). The client steps down none -> minimal -> low -> omitted;
the study applied `effort=minimal` and the model produced 691.6 reasoning tokens per call on average
(all other models: reasoning off, 0 reasoning tokens; gpt-4o: parameter not sent). The applied setting is
stored per record (`applied_params`).

## 8. Provider routing varies (T01)

OpenRouter routed the same model to different providers: glm-5 was served by 6 providers (Amazon Bedrock,
Baidu, Novita, SiliconFlow, StreamLake, Z.AI), deepseek-v3.2 and qwen3.5-397b by 3 each. Provider
differences were not studied. The provider is stored per record; mention routing as a limitation or pin
providers in a new study.

## 9. Agreement with experts depends on the level (T08)

Main-criteria rankings agreed with the experts (Spearman rho direct 0.900, CoT 1.000), but the eight global
weights agreed much less (direct 0.429, CoT 0.548; RMSE about 4.2-4.5 pp). A matching top criterion does
not mean matching weights. Agreement with the experts is a similarity measure, not a quality measure: a
condition closer to the expert weights is not better for that reason.

## 10. Weights may matter less than the data (T11, T13, T15)

For the study's suppliers (delivery indicators as rates), ARAS put Epsilon_Group first under every
weight vector (experts, direct, CoT, guided, equal weights); TOPSIS did the same except with the guided
weights (Alpha_Inc first); EDAS put Alpha_Inc first with expert, guided and equal weights and
Epsilon_Group first with the direct and CoT weights. Always show the equal-weight
baseline and the sensitivity analysis before attributing a ranking to the elicited weights.

## 11. Costs are small but not zero (TEKNIK_RAPOR section 9)

The whole study, including pilots and parameter tests, cost 4.2660 USD; the main run 1.4783 USD for 270
calls; the final revision 0.6633 USD (panel) + 0.4735 USD (judge). The guided prompts were long; direct
and CoT prompts are short. Always dry-run first and set `--budget`.
