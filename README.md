# HyQ-LC: Improving Light-Curve Classification with Hybrid Quantum–Classical Learning

Data, code, per-object predictions, and audited comparisons for ten-class stellar light-curve classification (nine variable-star categories and one non-variable category). Every reported test result in this version uses the **same 20,000 objects with distinct feature-bank row IDs**. Released IDs are not original survey identifiers. Training uses 633,034 objects and 359 preprocessed features; frozen preprocessing statistics were estimated from an original 50,000-object training subset.

The existing validation-selected models are retained. The main models use 105,297 validation objects; the older fixed-fusion comparisons use 10,000 validation objects. The test cohort had been inspected in prior exploratory analyses. It is the final reported test cohort, and is not an unopened independent confirmation sample. No sealed confirmation data are included or evaluated.

Data source: [LEAVES, Fei et al. (2024)](https://doi.org/10.3847/1538-4365/ad785b), with the [original dataset record](https://doi.org/10.12149/100962). The window/token model design adapts the VQC/FQT constructions in [Chen and Kuo (2026), arXiv:2604.23931](https://arxiv.org/abs/2604.23931), using classical MLP blocks for the window backbone and local VQC blocks for the token-wise quantum classifier.

## Strictly parameter-matched pure MLP

Both models have 44,238 parameters. F1 and ACC are percentages; differences are percentage points (pp).

The main hybrid adds a residual MLP prediction from a window MLP backbone and a global nine-qubit readout. The readout uses amplitude encoding plus grouped angle re-uploading and produces 27 X/Y/Z expectations. The two primary fine-tuning seeds share one pre-trained backbone. The pure MLP is trained from scratch; matching parameter totals and maximum training-object presentations does not match initialization, architecture, or computational cost.

| Seed | Hybrid F1 | Pure-MLP F1 | ΔF1 (pp) | F1 difference 95% interval (pp) | ΔACC (pp) |
|---|---:|---:|---:|---|---:|
| 42 | 92.2913 | 91.4637 | +0.8276 | [-0.0650, +1.6945] | +0.6000 |
| 44 | 92.3005 | 91.0611 | +1.2393 | [+0.5153, +2.0053] | +0.8450 |

Macro-F1 point estimates favour the hybrid on both seeds. The seed-42 F1 interval includes zero; the seed-44 interval is positive.

## Classical branch replacements

| Comparison | Seed | ΔF1 (pp) | 95% interval (pp) |
|---|---:|---:|---|
| Same-parameter MLP replacement | 42 | +0.0205 | [-0.3299, +0.3726] |
| Same-parameter MLP replacement | 44 | +0.2965 | [-0.0333, +0.6982] |

These branch comparisons have positive F1 point differences, with intervals spanning zero. The two 540-parameter readouts retain the same backbone and fusion-head construction. They do not establish an improvement specific to the quantum branch.

## LightGBM controls

| Model | Capacity | Test Macro-F1 (%) | Test ACC (%) |
|---|---|---:|---:|
| Fixed quantum+MLP | 55,581 reported scalars | 92.3371 | 94.7450 |
| Classical MLP+MLP fusion | 54,631 reported scalars | 92.1510 | 94.7500 |
| Budget-matched LightGBM | 55,500 scalar leaf outputs | 92.2204 | 94.8650 |
| Full-budget LightGBM | 112,220 scalar leaf outputs | 92.9681 | 95.1100 |

Against budget-matched LightGBM, the fixed hybrid has slightly higher F1 (+0.1166 pp) and lower ACC (−0.1200 pp). The largest positive class-level F1 differences are in CEP, RRC, and ROT; full-budget LightGBM remains stronger overall. Scalar leaf-output counts are an approximate capacity proxy, not an exact equivalence to neural-network parameters.

Fixed fusion is a separate configuration: it averages corrected logits from a token-wise quantum classifier and a window MLP with weights 0.5/0.5. The classical fusion control mixes corrected window-MLP and Dense-MLP probabilities with weights 0.6/0.4. Their reported scalar totals include one fixed mixing coefficient. This is an approximate-budget ensemble comparison; its F1 difference is +0.1860 pp with a 95% interval of [−0.2869, +0.6740] pp.

### Test class distribution and classwise F1

The following comparison uses fixed quantum fusion and the 55,500-leaf-value LightGBM control. The last row reports total objects and the unweighted mean of class F1 scores.

| Class | Objects | Share (%) | Quantum-fusion F1 (%) | Budget-LightGBM F1 (%) | ΔF1 (pp) |
|---|---:|---:|---:|---:|---:|
| CEP | 45 | 0.225 | 84.3373 | 83.3333 | +1.0040 |
| DSCT | 337 | 1.685 | 86.3636 | 86.4536 | -0.0899 |
| EA | 912 | 4.560 | 96.8097 | 97.1239 | -0.3142 |
| EW | 6,307 | 31.535 | 94.7460 | 94.7427 | +0.0034 |
| M | 352 | 1.760 | 97.5955 | 97.5887 | +0.0068 |
| Non-var | 2,566 | 12.830 | 96.0356 | 96.7356 | -0.7000 |
| ROT | 2,257 | 11.285 | 86.3818 | 85.8691 | +0.5128 |
| RRAB | 796 | 3.980 | 97.5579 | 97.7358 | -0.1779 |
| RRC | 357 | 1.785 | 86.0908 | 84.9478 | +1.1429 |
| SR | 6,071 | 30.355 | 97.4523 | 97.6740 | -0.2217 |
| Total / Macro-F1 | 20,000 | 100.000 | 92.3371 | 92.2204 | +0.1166 |

Machine-readable values: `results/tables/classwise_fixed_fusion_vs_budget_lightgbm.csv`.

## Reproduce the reported test results

```bash
pip install numpy
python experiments/evaluate_test20000.py --predictions data/predictions_test.npz
```

This checks the released prediction vectors and recomputes 57 model rows and 45 paired comparisons. It does not fit, retune, or select models. All intervals use 10,000 ordinary paired-object bootstrap resamples, conditional on each fitted pair, with no training-seed/search uncertainty or multiplicity correction. Negative results and all previously reported successful checkpoint variants are included. The hybrid seed-43 checkpoint failed the original numerical audit and has no valid test-result row.

## Files

| Path | Contents |
|---|---|
| `data/test.npz` | Exactly 20,000 objects: `x` (float32, N×359), `y`, `ids`, and class names. |
| `data/predictions_test.npz` | The same ordered `ids`, `y`, and one named prediction vector per audited model. |
| `data/test_model_catalog.json` | Frozen model choices, checkpoint/source-prediction hashes, and test-vector hashes. |
| `data/feature_manifest.csv` and `data/feature_groups.json` | All 359 input-column names in order, their seven groups, dimensions, and provenance checksums. |
| `results/` | Only the reported test cohort: model JSONs, paired comparison JSONs, and CSV tables. |
| `plans/` | The explicit post-hoc test-recomputation scope and fixed comparison list. |
| `models/lgbm_budget.txt` | The validation-selected 15-leaf, 370-round LightGBM checkpoint. |
| `src/` and `verification/` | Standalone quantum readout and data-free mechanism checks. |
| `experiments/` | Portable test evaluator and training implementation references. |
| `paper/` | Matching manuscript source and PDF, two current PNG schematics with generation records, the earlier editable vector diagram and generator, and the class file. |

The [v1.1.3 data release](https://github.com/bestdo77/quantum-lightcurve-hybrid/releases/tag/v1.1.3) contains the test package, a manuscript snapshot, and ordered 359-feature manifest. The [current manuscript PDF](paper/HyQ-LC.pdf) includes the two updated schematics. Training and validation archives retain their original checksums and download links in `data/splits.json`. The portable evaluator reproduces statistics from released predictions; the original training references require the authors' cluster environment and do not provide portable end-to-end retraining.

The [final fact and citation audit](audit/final_fact_check_20261019.md) records the 57-model/45-comparison recomputation, displayed-number checks, all 19 reference entries, factual wording corrections, and the limits of verification from the public artifacts.

All quantum circuits were evaluated with classical state-vector simulation. No hardware or general computational quantum advantage is claimed. Code: MIT; data: CC BY 4.0.
