# Manuscript figures

The current manuscript includes two PNG schematic assets:

- `fig_overview.png`: residual nine-qubit model and the separate fixed score-fusion comparison.
- `fig_quantum_circuit.png`: input encoding, the nine-qubit variational circuit, and Pauli-expectation readout.

These raster illustrations were generated with the built-in `image_gen` tool using an author-supplied architecture image as a style reference. The prompts and revisions are recorded in `figure_generation_prompts.md`. Model dimensions, circuit order, parameter counts, residual connections, and fusion rules were reviewed against the manuscript and released implementation references. The light curves are schematic, not measured data. The circuit figure abbreviates the Euler gate labels; each gate's trainable angle depends on layer and qubit as defined in the manuscript.

The earlier `fig_architecture.pdf`, `fig_architecture.svg`, and `draw_architecture.py` remain available as an editable vector schematic of the same model configurations. They do not regenerate the current PNG illustrations.

Compile `main.tex` with pdfLaTeX, retaining both current PNG files beside the source. `quantumarticle.cls` is included. The manuscript uses cross-references and requires a second LaTeX pass.
