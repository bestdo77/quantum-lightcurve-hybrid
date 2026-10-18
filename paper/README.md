# Manuscript and architecture diagram

`main.tex` uses `quantumarticle.cls` and `fig_architecture.pdf`. Compile with pdfLaTeX twice. `HyQ-LC.pdf` is the matching compiled manuscript. All manuscript test tables use the same 20,000 objects as the root evaluation package.

The Methods section includes 13 numbered equations covering the backbone, amplitude and angle encoding, circuit layers, Pauli measurements, residual fusion, matched classical readout, fixed fusion rules, weighted training loss, class correction, metrics, and paired bootstrap. A feature-composition table describes all seven groups of the 359-dimensional input, and the classwise comparison table reports counts, shares, and F1 for fixed quantum fusion and budget-matched LightGBM. LEAVES is credited for the original light curves; Chen and Kuo (arXiv:2604.23931) are credited for the token-based VQC/FQT design adapted here.

The architecture diagram distinguishes the main nine-qubit residual hybrid from the separate token-wise quantum score-fusion configuration. `fig_architecture.svg` retains editable text and vector objects; the PNG is a preview. Regenerate SVG and PDF with:

```bash
pip install reportlab
python paper/draw_architecture.py
```

The generator embeds Arial when the Windows font files are available and otherwise uses Helvetica. It does not access experiment data, credentials, or network resources. The final diagram was checked against the model implementation and visually inspected in the compiled manuscript.
