# Data

All test results use exactly the same 20,000 unique object IDs and labels. `test.npz` contains the corresponding 359-dimensional float32 features (`x`), integer class labels (`y`), IDs, and class names. `predictions_test.npz` stores those IDs and labels in the same order, plus one prediction vector per audited model. Class order: CEP, DSCT, EA, EW, M, Non-var, ROT, RRAB, RRC, SR.

`splits.json` records class counts, archive SHA256 checksums, and training/validation archive links. Training has 633,034 objects; model selection retains the original validation protocols (105,297 objects for the main study, 10,000 for the fixed-fusion controls). Only test-cohort metrics are published in the result tables.

Preprocessing is frozen: median/scale statistics from the original training subset, nonfinite values replaced by the median, values clipped to [-10,10]. Models apply atan internally where specified; do not standardise again. IDs are internal feature-bank row indices, not resolvable catalog names. The test set was previously examined in exploratory work; no sealed confirmation objects are distributed or evaluated.
