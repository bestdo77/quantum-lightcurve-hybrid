# Experiments (original audited scripts)

These are the **exact scripts** used for the audited runs, copied unmodified so
that every number in `results/` can be traced to the code that produced it.
They are provided for transparency, not as a one-command reproduction package.

## Environment requirements

The original runs used a dedicated environment with: PyTorch + CUDA, Triton, a
custom `fused_gate` module, the author QKAN package (`author_qkan_4de1642/src`
on `PYTHONPATH`), PennyLane (independent physical checks only), LightGBM, and
scikit-learn. Several scripts assume the original directory layout and absolute
paths from the authors' cluster; you will need to adapt paths before running
them elsewhere. The clean, dependency-light model definition is
`src/quantum_readout.py`.

## Map

| Script | Role |
|---|---|
| `paper_evidence_neural.py` | Main expanded-validation study: paired-seed 44,238-parameter models (quantum, pure MLP, matched MLP, no-CNOT, frozen quantum, quadratic, strong MLP), preflight, training, audit. |
| `paper_evidence_followup.py` / `..._evaluation.py` | Seed-44 mechanism controls and seed-45 replication; independent CPU evaluation with paired bootstrap comparisons. |
| `paper_evidence_fullinteraction.py` / `..._evaluation.py` | Full-359-dimensional classical bilinear interaction controls (`quadratic_s44/s45`). |
| `paper_evidence_statistics.py` / `paper_evidence_references.py` | Paired object-level bootstrap and comparison helpers. |
| `paper_evidence_gradients.py` | Gradient verification against PennyLane adjoint differentiation and parameter-shift. |
| `priority_14235_models.py` | Hybrid model: backbone + readout + residual-logit fusion head (`logits = base + residual`). |
| `global_hybrid_transformer_models.py` | Global coherent (amplitude-encoding) readout used by the q9d12 branch. |
| `p1_matched_classical_comparison.py` | Fixed-projection classical replacement readout. |
| `train_priority_14235.py` / `audit_priority_14235_results.py` | Shared training loop and strict numerical audit. |
| `safe_capturable_adamw.py` | CUDA-graph-safe AdamW used by the protocol. |
| `export_projection_information.py` | Exports the audited seed-44 readouts for the public projection witness (no dataset access). |

Frozen plans and their SHA-256 hashes are in `../plans/`.
