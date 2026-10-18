# Test results

Every classification result uses exactly 20,000 objects from the same `data/test.npz` cohort. `audited/` contains recomputed test metrics and source proofs; `comparisons/` contains 45 paired comparisons with 10,000-resample percentile intervals; `tables/` contains all 57 model rows and the comparison CSV. CSV scores are fractions, while columns ending in `_pp` are percentage points.

The five gradient, implementation-consistency, and cost JSONs are diagnostic checks, not classification test cohorts. They do not change the reported test sample size. Test predictions were extracted by exact ID/label matching from existing audited prediction caches, with fixed validation-selected checkpoints, corrections, learning rates, and fusion coefficients. This is a post-hoc reporting-scope update; no new training or test-driven model selection occurred.

Bootstrap intervals are ordinary paired-object intervals conditional on the two fitted models. They do not account for training-seed, model-search, or multiplicity uncertainty. Point estimates and uncertainty bounds are both retained, including inconclusive and negative comparisons.
