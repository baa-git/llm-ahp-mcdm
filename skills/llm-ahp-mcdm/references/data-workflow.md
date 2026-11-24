# Stages 0-2: from a data set to a problem file

Code: `ahp_bench/tabular.py` (reading), `ahp_bench/eda.py` (stage 0), `ahp_bench/criteria.py` (stages 1-2),
`ahp_bench/hierarchy.py` (display). Standard library only; matplotlib only for `hierarchy.png`.

## Reading the file

- Encodings tried in order: UTF-8 with BOM, UTF-8, cp1254 (Turkish Windows), then latin-1.
- Delimiter: comma, semicolon, tab or pipe, whichever gives the most rows with the header's field count.
- Leading lines starting with `#` and blank lines are skipped (the synthetic example uses one).
- Missing cells: empty, `NA`, `N/A`, `null`, `none`, `nan`, `-`, `?`.
- Numbers: per column the decimal mark is `.` or `,` (whichever parses more cells; `1.234,5`, `12,75`,
  `1,234.5` and a trailing `%` are understood). A column is numeric when at least 95% of its non-missing
  cells parse; the rest are counted as `non_numeric` and treated as missing.
- Dates: `yyyy-mm-dd`, `dd.mm.yyyy`, `dd/mm/yyyy` (time part ignored).
- Rows with too many or too few fields are padded or cut and counted as malformed.

## Reading `eda.md` / `eda.json`

| Field | Meaning |
| --- | --- |
| `type` | `integer`, `numeric`, `date`, `categorical`, `text`, `identifier` (unique per row), `empty` |
| `missing`, `distinct` | per column |
| `numeric` | min, max, mean, median, sd (sample), non-numeric cells, decimal mark |
| `values` | category frequencies (all values up to 12 distinct, else the top 5) |
| `candidate_alternative_columns` | categorical/text columns with 2-100 distinct values that repeat over rows; name hints (supplier, vendor, product, ...) rank first |
| `candidate_indicators` | suggested indicators: mean of numeric columns, ratio of a count to a quantity (defect/return/... over quantity/units/...), share of each value of a categorical column (2-6 values; for 2 values only one), mean days between two date columns, row count |
| `suggested_direction`, `direction_basis` | from words of the column name or value only (cost: price, cost, defect, late, lead, days, cancel, ...; benefit: quality, rating, on time, delivered, discount, compliance, ...). `null` = cannot be inferred: the user decides |
| `notes` | identifiers, constant columns, heavy missingness, no alternative column, fewer than 3 candidates |

Candidates are suggestions. Discuss with the user which ones measure what matters for the decision,
whether a direction is right (a discount is a benefit for a buyer, a cost for a seller), and whether
counts should be rates (a count favours alternatives with more rows).

## Criteria spec

```json
{"spec_version": 1,
 "proposed_by": "human",
 "problem": {"name": "orders_synthetic", "title": "...", "goal": "evaluating suppliers of purchased components",
             "role": "You are a procurement professional with experience in supplier evaluation."},
 "alternative_column": "supplier",
 "alternatives": ["Supplier_A", "Supplier_B"],
 "filter": {"column": "category", "not_equals": "Samples"},
 "indicators": [
   {"id": "ONTIME", "label": "On-time delivery rate", "aggregation": "share", "column": "delivery_status",
    "value": "On time", "filter": {"column": "delivery_status", "not_equals": "Cancelled"}, "scale": 100,
    "direction": "benefit", "unit": "% of delivered orders", "rationale": "..."}],
 "hierarchy": {"criteria": [
   {"name": "Cost", "indicator": "PRICE"},
   {"name": "Delivery", "sub_criteria": [{"name": "On-time Delivery", "indicator": "ONTIME"}, ...]}]}}
```

- `proposed_by`: `human`, `host agent` or a model id (recorded in the problem file).
- `alternatives` (optional) restricts and orders the alternatives; default: every value, sorted.
- `filter` (optional, top level or per indicator): `{"column": C, OP: V}`, OP one of `equals`,
  `not_equals`, `in`, `not_in` (lists), `gt`, `ge`, `lt`, `le` (numbers); a list of conditions means AND.
- `scale` multiplies the aggregated value (100 for percent); `offset` is added and is meant only to keep
  values > 0 (ARAS and EDAS divide by them) - say why when you use it.
- Keys starting with `_` are comments. Strings starting with `REVIEW` fail validation on purpose.

| Aggregation | Fields | Value per alternative |
| --- | --- | --- |
| `mean`, `sum`, `median`, `min`, `max` | `column` (numeric) | over non-missing cells |
| `count` | optional `column` | rows (or non-empty cells of `column`) |
| `share` | `column`, `value` (or a list) | rows with that value / rows with a non-empty `column` |
| `ratio` | `numerator`, `denominator` | sum / sum over rows where both are present |
| `mean_days_between` | `start`, `end` (dates) | mean of end - start in days |

Hierarchy rules (the existing problem format): goal -> 3-9 main criteria -> optional 3-9 sub-criteria per
main criterion; each leaf maps to exactly one indicator and each indicator to exactly one leaf. Deeper
hierarchies and several indicators per leaf are not supported; combine indicators into one (e.g. a ratio) or
split a group. Two related leaves cannot form a group (groups need 3); keep them as main criteria.

`criteria validate` checks the structure, that every column exists and has the right type, that share values
occur, and that every alternative gets a value > 0; it prints the tree and the matrix. `criteria build`
writes the problem file with a `provenance` block:

| Key | Content |
| --- | --- |
| `data_file`, `data_sha256`, `data_rows`, `rows_used` | the data set (file name only) |
| `spec_file`, `spec_sha256` | the criteria spec |
| `criteria_proposed_by` | from the spec |
| `criteria_confirmed_by_user` | true only with `--confirmed` or `ahp-bench guidance confirm --criteria` |
| `data_summary` | encoding, delimiter, columns, alternative column, rows per alternative, missing cells in used columns |

The build is deterministic (no timestamp): the same data and spec give the same file and hash, so stored
records stay valid. Indicators carry `source`, `aggregation`, `filter`, `rationale` for the hierarchy table.

## Showing the hierarchy

`ahp-bench hierarchy --problem P --out DIR` prints an ASCII tree and writes `hierarchy.md` (Mermaid
`graph TD`, the tree, and the indicator table: main criterion, criterion, indicator, unit, direction, source
column(s), aggregation, filter) and `hierarchy.png` (grayscale, square boxes, orthogonal connectors; skipped
without matplotlib). The generic report repeats the hierarchy, the indicator table (H01), the performance
matrix (H02) and the data summary before the weights.
