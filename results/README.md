# Results

## `audited/`

Per-checkpoint result JSONs from the strict audit pipeline. Each contains the
model name, seed, selection rule (`best_macro` / `best_acc`), epoch, α,
parameter count, validation and development metrics, per-class F1 and the
confusion matrix. Also included: the matched-Transformer development result,
tree-selection result, gradient-verification results, CPU cost measurements, and
the 20-step PennyLane training-consistency check.

## `comparisons/`

Paired object-level comparisons. Fields: `quantum` / `control` metrics,
`quantum_minus_control`, `interval95` (10,000 paired bootstrap resamples,
conditional on the two fixed models), `quantum_only_correct` /
`control_only_correct`, `exact_mcnemar_accuracy_p`, and `quantum_proof` /
`control_proof` linking both sides to checkpoint and prediction SHA-256 hashes.
Intervals do not include seed, model-search, or multiplicity variance.

## `tables/`

- `main_results.csv`: one row per audited checkpoint. Scores are **fractions**
  (0.9198653 = 91.9865%).
- `paired_comparisons.csv`: one row per comparison. Deltas and CI bounds are in
  **percentage points** (0.8193 = +0.8193 pp).

## Reading the evidence honestly

- The hybrid beats the same-parameter pure MLP and the fixed 500-tree RF on both
  audited seeds (42, 44), intervals excluding zero.
- A separate fixed-fusion model beats the approximately leaf-budget-matched LightGBM
  on validation Macro-F1 and ACC, and on extra-split Macro-F1. On the official
  report+extra merged test, however, LightGBM is slightly ahead (quantum minus
  LightGBM: **−0.0353 pp F1**, −0.1640 pp ACC). Against the same-architecture
  classical fusion, the quantum model is ahead on merged-test Macro-F1 by
  **+0.2831 pp** (ACC −0.0280 pp). See
  `audited/lgbm_leaf_budget_pooled_comparison.json`. The LightGBM comparison is an
  approximate leaf-output budget match, not a comparison with the strongest trees.
- The full-budget class-weighted LightGBM beats the 44,238-parameter hybrid on both
  audited seeds, intervals excluding zero.
- Against the same-parameter MLP replacement branch, seed 44 has a positive F1
  interval and seed 42 does not; frozen-quantum and no-CNOT re-trainings are
  statistically tied with the full model at seed 42.
- Seed 43 failed the strict numerical audit and is excluded; the failure record
  is preserved. Seed 45 and the quadratic controls have completed and their audited
  results are included.
