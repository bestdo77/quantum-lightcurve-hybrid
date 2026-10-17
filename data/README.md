# Data

Three compressed NumPy archives, distributed as release assets (they are too
large for the git tree). `splits.json` records object counts, per-class counts,
and SHA-256 of each archive; `data_checksums.csv` is the flat checksum table.

| Split | Objects | Role in the study |
|---|---|---|
| `fit.npz` | 633,034 | Training pool. |
| `validation.npz` | 105,297 | **Only** source of model selection: checkpoints, prior-correction coefficient α, and the pre-declared learning rates. Includes the original 10,000-object pilot validation split as its first 10,000 rows. |
| `development.npz` | 65,631 | Development evaluation, disjoint from validation. The `cohort` array separates `report` (first 5,000 rows), `extra` (next 20,000), and `previously_unused` (last 40,631, the historical expansion subset — no longer "unseen" after this study). |

## Arrays

- `x`: float32, shape `(N, 359)`. Handcrafted light-curve features after frozen
  preprocessing: median/scale standardisation (statistics frozen from the
  original 50K training subset), non-finite values replaced by the median,
  clipped to `[-10, 10]`. Models apply `atan(x)` themselves; do not standardise
  again.
- `y`: int64, shape `(N,)`, class index into `classes`.
- `classes`: the 10 class names, `['CEP','DSCT','EA','EW','M','Non-var','ROT','RRAB','RRC','SR']`.
- `ids`: int64 row indices into the authors' internal feature bank, for split
  provenance only (not resolvable without the internal catalog).
- `cohort` (development only): string array described above.

## What is deliberately absent

The sealed confirmation set (`confirmation2`) is not included: no features,
labels, ids, or predictions. It has only ever been checked for ID non-overlap.
Publishing it would void the planned final independent confirmation.

Raw light curves are not part of this release; the 359-dimensional features are
the exact model inputs used by every audited experiment.
