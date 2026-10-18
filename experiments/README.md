# Evaluation and implementation references

Run `evaluate_test20000.py --predictions ../data/predictions_test.npz` from this directory (or use the root README command). It needs only NumPy and reproduces all reported test metrics, confusion matrices, and paired intervals from fixed prediction vectors. The explicit scope and comparison list are in `../plans/`.

The remaining scripts document the original model factories, training loops, and numerical checks. Reporting-only scope strings in the main and interaction training references have been updated; original checkpoint source hashes refer to their immutable historical versions. Training code assumes the authors' cluster layout, PyTorch/CUDA/Triton, custom gate kernels and optional PennyLane/QKAN dependencies. It is not a portable retraining command. The clean readout is `../src/quantum_readout.py`; data-free checks are under `../verification/`.

Original pre-recomputation source files and frozen training plans remain traceable through Git history. The current evaluation entry point and all active result tables use the single test cohort only.
