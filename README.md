# Hybrid Quantum–Classical Light-Curve Classification

Data, code, and audited results accompanying the preprint
*"Positive but Bounded Evidence for a Hybrid Quantum–Classical Light-Curve Classifier:
Matched-Parameter Comparisons, Error Complementarity, and Mechanism Probes"* (arXiv link TBD).

**Task.** Multi-class classification of variable stars from survey light curves
(ZTF/LSST-style photometry) into 10 classes: CEP, DSCT, EA, EW, M, Non-var, ROT,
RRAB, RRC, SR. Each object is represented by 359 handcrafted, preprocessed features.
The released training pool contains 633,034 objects.

**Headline result — and its boundary.** A hybrid model with a 540-parameter quantum
readout (9 qubits, 12 layers, angle re-uploading; 44,238 parameters in total) beats a
same-parameter pure MLP and a 500-tree random forest on the 65,631-object development
set, with paired bootstrap intervals excluding zero on two audited seeds. It remains
**clearly below a class-weighted LightGBM baseline**, and it does **not** consistently
beat same-parameter classical replacement branches inside the identical architecture.
**No general quantum advantage is claimed.** All experiments are exact classical
state-vector simulations (CPU/GPU); no quantum hardware was used.

## Repository layout

| Path | Content |
|---|---|
| `data/` | Split metadata, class list, checksums. The feature/label archives themselves are distributed as **GitHub Release assets** (`fit.npz`, `validation.npz`, `development.npz`). |
| `src/quantum_readout.py` | Clean, standalone PyTorch implementation of the 9-qubit, 12-layer quantum readout (540 parameters). No data, credentials, or network access required. |
| `verification/` | Portable, data-free reproductions of the mechanism checks: the exact interaction form / CNOT witness, and the 16-pair projection-separability witness with audited seed-44 readouts. |
| `experiments/` | The exact training/evaluation scripts used in the audited runs, copied unmodified for transparency (see `experiments/README.md` for their environment requirements). |
| `results/` | Audited per-checkpoint result JSONs, paired bootstrap comparison JSONs, and consolidated CSV tables. |
| `plans/` | Frozen pre-registered training/evaluation plans and their SHA-256 checksums. |

## Quickstart (mechanism verification, no data needed)

```bash
pip install numpy torch            # pennylane optional, for the independent physical check
cd verification/mechanism
python verify_mechanism.py         # interaction-form + CNOT/local-information witness

cd ../projection_witness
python verify_projection_information.py
```

## Data

Download the three archives from the release page and verify against
`data/data_checksums.csv`:

| Archive | Objects | Arrays |
|---|---|---|
| `fit.npz` | 633,034 | `x` (float32, N×359), `y` (int64), `ids`, `classes` |
| `validation.npz` | 105,297 | same; used for *all* model selection |
| `development.npz` | 65,631 | same, plus `cohort` (`report` 5,000 / `extra` 20,000 / `previously_unused` 40,631) |

```python
import numpy as np
d = np.load('fit.npz')
x, y = d['x'], d['y']
```

The sealed confirmation set (`confirmation2`) is **not** included in any form
(features, labels, ids, or predictions); it remains sealed for the final
independent check. See `data/README.md` for details.

## Results

`results/tables/main_results.csv` collects every audited model (scores as
fractions); `results/tables/paired_comparisons.csv` collects every paired
comparison (deltas and 95% bootstrap intervals in **percentage points**, 10,000
resamples, conditional on the fixed models; no seed/search/multiplicity
correction). Every comparison JSON links both sides to checkpoint and prediction
SHA-256 proofs. Negative results and the failed seed-43 audit are retained.

## Status

Seed-45 replication and the full-input classical-interaction controls
(`quadratic_s44/s45`) are still running under the frozen plans in `plans/`;
their audited results will be appended to `results/` without altering existing
files. The sealed-set final confirmation has not been performed.

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0. If you use this package, please cite
the preprint (citation block to be added once the arXiv id is available).
