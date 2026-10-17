# Verification (data-free)

Both packages run without the dataset, pretrained backbones, network access, or
cluster credentials. `quantum_readout.py` is duplicated in each package so that
each is self-contained; it is byte-identical to `../../src/quantum_readout.py`.

## `mechanism/`

```bash
python verify_mechanism.py
```

Reproduces the exact global-interaction identity and the CNOT/local-information
witness: the paired amplitude states `(|0> ± |255>)/sqrt(2)` are indistinguishable
to all local readouts of the CNOT-free circuit (max difference ~4.17e-7, i.e.
float error) but are separated by the full CNOT circuit (max difference
0.0828799). An independent PennyLane physical check runs if `pennylane` is
installed. `readout_initial_seed42.json` holds the fixed initial parameters.

## `projection_witness/`

```bash
python verify_projection_information.py
```

Reproduces the 16-pair synthetic separability check with the **audited seed-44**
readouts (`projection_information_seed44.json`, exported without any dataset
access): the trained fixed-26-dimensional projection branch cannot distinguish
the constructed pairs (0/16 above threshold), while the quantum readout
separates all 16. `projection_information_result.json` is the recorded result;
`projection_information_result_plan.json` is the frozen construction plan.

**Interpretation boundary:** an untrained classical bilinear readout over the
full 359-dimensional input also separates all 16 pairs. The witness
distinguishes fixed low-dimensional projections from interaction-accessing
readouts, classical or quantum — it is not evidence of a quantum-unique
capability.
