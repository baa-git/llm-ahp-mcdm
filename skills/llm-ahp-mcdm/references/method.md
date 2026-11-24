# Method

All formulas below are what `ahp_bench` computes (`ahp.py`, `analysis.py`, `revise.py`).

## 1. Questions to a pairwise-comparison matrix

A task with criteria c_1..c_n has n(n-1)/2 questions. Question k asks "Left c_i | Right c_j" for the
pair (i, j) given by the problem file's question order (default (1,2), (1,3), ..., (n-1,n)).

The answer is one of 17 Saaty-scale values: 9, 8, ..., 2, 1, 1/2, ..., 1/9, where v > 1 means the left
criterion is more important, 1/v the right one, 1 equal. Answers are parsed strictly (no repair of
off-scale values; unquoted fractions such as 1/3 in otherwise valid JSON are quoted, values unchanged).

    a_ii = 1,   a_ij = v_k,   a_ji = 1 / v_k

## 2. Priority vector (eigenvector method)

w is the principal right eigenvector of A (largest real eigenvalue lambda_max), taken in absolute value
and normalised to sum 1.

## 3. Consistency

    CI = (lambda_max - n) / (n - 1)
    CR = CI / RI(n)

Random index RI (Saaty): n = 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45.
A matrix is called consistent when CR rounded to 4 decimals is <= 0.10 (`analysis.is_consistent`).

## 4. Group aggregation (several models, repetitions or experts)

Aggregation of individual judgements (AIJ): element-wise geometric mean of the matrices, then the
eigenvector of the aggregated matrix.

    g_ij = (prod_k a_ij^(k))^(1/K)

All matrices are aggregated, consistent or not (the same rule was applied to the human experts).
Panel weights of a condition pool every matrix of every model; model-level weights pool that model's
repetitions.

## 5. Global weights

For a main criterion m with weight W_m and a sub-criteria group with local weights u_s:

    global(s) = W_m * u_s

Main criteria without a group keep their main weight. Leaves are ordered as the main criteria, each
replaced by its sub-criteria in place.

## 6. Indicators from a data set (stages 0-2)

For alternative a and indicator j with rows R_a (rows whose alternative column equals a, after the spec's
filters): mean / sum / median / min / max of the numeric column over non-missing cells; count = |R_a| (or
non-empty cells); share = |{r in R_a : column = value}| / |{r in R_a : column non-empty}|; ratio =
sum numerator / sum denominator over rows with both values; mean_days_between = mean(end - start) in days.
x_aj = value * scale + offset, written with 6 significant digits; every x_aj must be > 0.
See `data-workflow.md`.

## 6b. Ranking methods

X = performance matrix (alternatives i x indicators j, all entries > 0), w = indicator weights (the global
leaf weights mapped to their indicator columns), J+ = benefit indicators, J- = cost indicators. An
optional `row_share_percent` transform first replaces the listed columns by their percentage of the row
total (the study turned four delivery counts into rates).

### TOPSIS

    r_ij = x_ij / sqrt(sum_i x_ij^2),   v_ij = w_j r_ij
    A+_j = max_i v_ij (j in J+), min_i v_ij (j in J-);   A-_j = the opposite
    D+_i = sqrt(sum_j (v_ij - A+_j)^2),   D-_i = sqrt(sum_j (v_ij - A-_j)^2)
    C_i = D-_i / (D+_i + D-_i)          (higher is better)

### ARAS

    b_ij = x_ij (j in J+), 1 / x_ij (j in J-)
    b_0j = max_i b_ij                                   (optimal alternative)
    n_ij = b_ij / (sum_i b_ij + b_0j)                   (the optimal row is in the denominator)
    S_i = sum_j w_j n_ij,   K_i = S_i / S_0             (utility degree, higher is better)

### EDAS

    AV_j = mean_i x_ij
    PDA_ij = max(0, (x_ij - AV_j) / AV_j) for J+,  max(0, (AV_j - x_ij) / AV_j) for J-
    NDA_ij = max(0, (AV_j - x_ij) / AV_j) for J+,  max(0, (x_ij - AV_j) / AV_j) for J-
    SP_i = sum_j w_j PDA_ij,   SN_i = sum_j w_j NDA_ij
    NSP_i = SP_i / max SP,   NSN_i = 1 - SN_i / max SN
    AS_i = (NSP_i + NSN_i) / 2                          (higher is better)

Ranks: 1 = highest score; ties share the average rank.

### Baseline and sensitivity

- Equal-weight baseline: every indicator weight 1/J.
- One-at-a-time sensitivity: indicator j gets weight s in {0.125, 0.20, 0.30, ..., 0.90}; the other J-1
  indicators share 1 - s equally. Reported: the ranking at each level and the lowest level at which the
  top alternative differs from the equal-weight top alternative.

## 7. Revision protocol (optional)

For a first answer with CR > 0.10 a chain of at most `--max-revisions` calls is run:

- The conversation is: original messages + one assistant turn (the original reply, or the best answers so
  far written in the same JSON format) + a revision message. Earlier turns are not accumulated.
- The revision message lists, computed by code, up to three most inconsistent triads (largest
  |ln a_ij + ln a_jk - ln a_ik| over i < j < k) and up to three circular (ordinal) triads
  (x over y, y over z, z over x), the previous answers and the allowed values.
- With `--judge`, a judge model rewrites the explanation (it may not suggest values or say which
  criterion matters more); after each parsable revision it gives a verdict on whether the original logic
  was preserved (yes / partly / no).
- The chain stops at the first consistent revision or at the limit. The final answers are the best valid
  answers (lowest CR among the original and all parsable revisions; ties keep the earlier one).
- Deterministic logic metrics compare final with original answers: answers changed, Spearman rho of
  the weights, top criterion unchanged, full ranking unchanged, largest absolute weight change (pp).

## 8. Similarity to experts

Spearman rho and Kendall tau-b between weight vectors, RMSE and mean absolute difference in percentage
points (weights x 100), for the main weights and the global weights.
