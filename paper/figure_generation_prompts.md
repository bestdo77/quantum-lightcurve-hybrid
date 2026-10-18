# HyQ-LC figure-generation record

Tool: built-in image_gen (imagegen).
Status: generated design drafts, PNG raster images; not editable vector originals.
Reference: user-provided blue/green architecture figure. Reference used for visual style; model content follows the current manuscript.

## Figure 1 — model overview

Create Figure 1 for a research paper, a polished scientific neural-network architecture illustration. Use the attached image ONLY as a STYLE reference: elegant white canvas, pale blue quantum branch and pale green classical branch, softly rounded containers, blue and green directional arrows, translucent stacked tensor tiles, crisp black Helvetica-style labels, tiny restrained shadows. Replace the old architecture completely with the scientifically correct structure specified below. Wide landscape approximately 2:1 aspect ratio, high resolution and sharp readable English labels. This must look as carefully composed as the reference, with more generous spacing and larger text. No photo, no watermark, no invented data.

MAIN PANEL (a), approximately upper 80 percent:
Left to right pipeline. On the far left draw a small attractive schematic light-curve glyph (blue/orange dots and thin curves, no numeric data); label "LEAVES light curves" and small "schematic". An arrow leads to a compact feature/preprocessing panel labelled "359 handcrafted features". List seven feature groups in small clean lines: "Base 81 · Global 17", "Histogram 32 · Pair 49", "Triad 63 · Time 64 · Spectral 53". Below: "Training-set imputation & scaling", "clip [-10, 10] → atan". This is a fixed feature vector, not time-series tokenisation. The transformed vector branches into TWO parallel pathways:

UPPER BLUE CONTAINER titled "Global quantum readout" with a tiny "9 qubits · 12 layers · 540 parameters" header. Show a stacked input-feature tile → "Amplitude encoding" with "359 → 512" below → "9-qubit VQC" (a tasteful small 9-wire circuit icon, detailed in Figure 2) → three compact output groups labelled "⟨X⟩  ⟨Y⟩  ⟨Z⟩" → tensor tiles labelled "27 features" → small "LayerNorm". Use blue arrows. This is ONE global circuit over the feature vector, not 120 separate circuits.
LOWER GREEN CONTAINER titled "Classical Window MLP". Show input feature vector → small "Pad to 360" box → row of stacked tokens "120 × 3 feature tokens" → rounded "Token MLP + window mixing" module with "× 2 stages" under a bracket → slim tensor column "h ∈ R^120". From this column split into TWO outputs: one through "LayerNorm" toward concatenation; another through a small "Base classifier" box to a clearly visible separate SKIP pathway labelled "z_base", going beneath the fusion head and into the final addition node. Green arrows. Do NOT use attention or raw-time windows.
To the right of both containers, BLUE LayerNorm and GREEN LayerNorm merge into a vertical "Concat" feature tile explicitly labelled "120 + 27 = 147". Then "Residual MLP" box with "147 → 128 → 10" and tiny "GELU". This outputs a short line labelled "Δz" into a round '+' node. The base-classifier skip z_base must ALSO visibly enter that same '+' node from below. Only after this sum, draw compact "Class correction" box with "z̃_c = z_c − α log w_c", then "Softmax", then an output panel labelled "10 classes" with the labels in two neat columns: CEP, DSCT, EA, EW, M, Non-var, ROT, RRAB, RRC, SR. Use no numerical probability bars. The main diagram should have one unobtrusive footer "Residual HyQ-LC · 44,238 total parameters".

SEPARATE SMALL INSET (b), bottom 20 percent, pale grey rounded boundary clearly detached from all main-panel arrows. Title "Fixed score fusion (separate comparison model)". Two small independent branches "Token-wise 3-qubit classifier" (blue) and "Window MLP" (green) each → "Corrected logits", then both into "Mean: 0.5 / 0.5" → "10-class prediction". Footer "55,581 reported scalars". The inset is an alternative model, never connected in series to the 9-qubit main model.

Prioritise perfect alignment, balanced whitespace, exact dimensions, legible module labels, and unmistakeable directed connectivity. Avoid unnecessary prose, decorative particle glows, huge headings, dense paragraph text, and crossings through labels. Panel letters (a), (b) top-left of their areas. Scientific diagram, clean publication illustration in the reference's visual language.

### Revision 1

Edit this scientific architecture figure carefully, preserving its existing style, layout, colours and all correct main-model text. Make ONLY these corrections:
1. In bottom panel (b), the blue "Token-wise 3-qubit classifier" box currently has incorrect invented descriptive lines "120 × 3 tokens → 3-qubit VQC → 10 logits (mean over tokens)". REMOVE those descriptive lines entirely, replace them with the single exact line "32,410 parameters".
2. In bottom panel (b), the green "Window MLP" box currently contains incorrect invented "120 × 3 tokens → MLP → 10 logits (mean over tokens)". REMOVE those descriptive lines entirely, replace them with the single exact line "23,170 parameters".
3. In top panel (a) the small illustrative circuit inside the "9-qubit VQC" box must have EXACTLY NINE fine horizontal wires. It is only a compact icon; keep gate details minimal.
4. In panel (b) bottom fusion formula use corrected-logit notation consistently: z = 0.5 z̃^(q) + 0.5 z̃^(c).
5. In panel (a), the quantum box that currently says "Amplitude encoding" with "359 → 512": change the heading to "Input encoding" and use two readable small lines "Amplitude: 359 → 512" and "Angles: 9 × 40". Do not change the dimensions or branch directions elsewhere.
Do not invent extra reductions or operations. All main-model dimensions, addition node, skip path, label and inset remain unchanged. Render sharp high-resolution English text.

### Revision 2

Make one extremely localized edit to this finished scientific architecture figure. In panel (a), inside the small pale-blue module labelled "9-qubit VQC", REMOVE the tiny individual circuit-wire/gate drawing entirely. Keep the module title "9-qubit VQC"; beneath it, put a clean blue rounded rectangle carrying just "×12 layers", with a tasteful single abstract quantum-state cube icon next to it. It must be a symbolic module, with NO individually drawn circuit wires, so it cannot imply a wrong qubit count. Keep the input arrow from "Input encoding" and output arrow to the XYZ readout tiles in precisely their existing positions. Preserve ALL other text, labels, dimensions, shapes, diagrams, colours and branch arrows exactly. Do not redraw the whole layout or invent content.

## Figure 2 — quantum circuit

Create Figure 2 of a scientific manuscript: a precise and beautiful 9-qubit variational quantum circuit explanatory illustration. Match the attached style reference's white background, pale blue rounded branch containers, luminous but restrained blue tensor stacks, small green classical-output accents, crisp black Helvetica-style labels and neatly aligned arrows. Scientific schematic, high resolution, landscape approximately 1.8:1, generous whitespace. This is a new diagram, not a copy of the architecture in the reference. Every technical instruction below is mandatory. NO invented scientific data, no photo, no watermark. Use legible English text and proper standard quantum circuit gate symbols.

Arrange three clearly separated panels:
(a) INPUT ENCODING, a compact horizontal upper band.
A 359-component feature stack labelled "r = atan(x)" splits into two routes.
First route: "Zero-pad to 512" → "L2 normalise" → "Amplitude preparation" with the concise formula "|ψ₀⟩ = Σₖ aₖ|k⟩". A blue connector supplies the initial state in panel (b), ONLY ONCE.
Second route: "Zero-pad to 360" → "9 groups × 40 features" → "φⱼ = (2/40) Σₜ r̄₄₀ⱼ₊ₜ". Tiny clear note "j = 0,…,8; t = 0,…,39". A second distinct blue connector supplies φ into the final data rotation gates in panel (b).

(b) QUANTUM LAYER, the dominant middle panel containing EXACTLY nine horizontal wires, labelled q₀, q₁, q₂, q₃, q₄, q₅, q₆, q₇, q₈ from top to bottom.
On left all nine wires pass through ONE tall global "Amplitude preparation" box.
Then an outlined rounded repeat container covering all nine wires, labelled "One variational layer · repeated ×12". State preparation remains OUTSIDE this repeated layer.
Inside repeated layer left-to-right:
1. Three aligned columns of blue gate boxes on EACH of the nine wires: Rz(θ₁), Ry(θ₂), Rz(θ₃). A small heading above says "Trainable Euler rotations".
2. Standard CNOT ring over EXACTLY NINE successive time columns. In column one a filled control dot on q₀ connected by a vertical line to circled-plus target on q₁. Column two q₁ control → q₂ target. Column three q₂ → q₃. Column four q₃ → q₄. Column five q₄ → q₅. Column six q₅ → q₆. Column seven q₆ → q₇. Column eight q₇ → q₈. Column nine q₈ control connected all the way UP to circled-plus target on q₀. No other dots at crossed wires. These nine CNOTs must be in DISTINCT columns, creating a staircase. Heading "Ordered CNOT ring". Under circuit add "0→1→2→3→4→5→6→7→8→0".
3. Final column of separate light-cyan gate boxes on each wire, labelled Rz(ωφ + b). Heading "Data re-uploading". φ comes from panel (a). Small note "Parameters depend on layer ℓ and qubit j".
Wires continue to the right, outside repeat container, with a collective arrow to panel (c).
Footer below circuit: "324 rotation + 108 frequency + 108 shift = 540 trainable parameters". Smaller footnote: "q₀ is the least significant basis bit".

(c) PAULI READOUT, a clean lower horizontal band with three pale-blue vector cards:
"⟨X₀⟩ … ⟨X₈⟩" and "9 values";
"⟨Y₀⟩ … ⟨Y₈⟩" and "9 values";
"⟨Z₀⟩ … ⟨Z₈⟩" and "9 values".
All three connect to a stacked vector labelled "q ∈ R²⁷", then arrow → small green "LayerNorm → residual fusion (Fig. 1)" box.
A restrained blue caption under readout: "Classically simulated state-vector expectations". No hardware device imagery, no single-shot simultaneous XYZ measurement, no nine disconnected Bloch spheres.

Make the nine-wire circuit exact, comfortably spaced, and the largest visual focus. Pale blue containers and translucent small tensor tiles should match the reference aesthetic, while standard circuit symbols remain flat and precise. Use panel letters (a), (b), (c). Avoid oversized decorative titles and avoid dense paragraphs.

### Revision 1

Edit this scientific 9-qubit circuit figure precisely, preserving its layout, colours, circuit wires and the CORRECT nine-column ordered CNOT staircase. Make the following limited scientific corrections:
1. In panel (a) the box currently called "Amplitude preparation" should instead read "512 amplitudes" with concise formula "a = pad₅₁₂(r) / ‖r‖₂". This is amplitude vector construction, not an extra quantum state-preparation stage.
2. The blue line leaving this "512 amplitudes" box currently goes across the top and down to the rightmost re-uploading region of panel (b); REMOVE that incorrect entire route. Reroute this amplitude vector using a thin blue arrow from panel (a) down along the LEFT MARGIN of panel (b), ending at the left edge of the tall "Amplitude preparation" box. The tall box in (b) is the ONLY quantum amplitude preparation. Keep the separate φ_j input arrow going to the "Data re-uploading" gates on the right exactly where it belongs.
3. On wire q₈, the MIDDLE Euler gate mistakenly reads Ry(θ₃); change it to Ry(θ₂). Inspect all nine rows: EVERY row must have Rz(θ₁), then Ry(θ₂), then Rz(θ₃). No index substitutions.
4. Panel (c) has three readout cards for X, Y, Z. Draw clean branching connector lines from ALL THREE cards into a shared concatenation junction that leads to q ∈ R²⁷. Currently only the Z card appears connected. They must all visibly feed the output feature vector without text overlap.
Preserve EXACTLY nine wires q₀–q₈ and nine sequential CNOTs 0→1,1→2,2→3,3→4,4→5,5→6,6→7,7→8,8→0. Do not alter gate counts, order,12 repeats,540 parameters,other formulas, or correct text. Keep the attractive pale-blue scientific figure style and sharp high-resolution typography.

### Revision 2

Make one extremely localized edit to this finished scientific 9-qubit circuit figure. REMOVE the long blue line at the FAR LEFT that currently starts beside the 359-component r=atan(x) label in panel (a), descends along the left side, and enters the amplitude-preparation box in panel (b). It is incorrectly connected to the raw features. Replace it with a SHORT vertical blue arrow above the tall amplitude-preparation box in panel (b), pointing down into the TOP of that box, with a small label above the short arrow reading "a from (a)". This refers clearly to the already existing normalized 512-amplitude vector in panel (a) without a long connecting line. Keep the 512-amplitude card formula, all black input-encoding arrows, the separate φ arrow on the right, EXACTLY nine circuit wires, ALL Euler rotation gates and subscripts, ALL nine sequential CNOTs, repeat boundary, footer, three readout connector lines and every other element unchanged. Do not add new gates, wires, formulas or long blue lines. Preserve the exact existing layout and style.

### Final typography revision

Edit only the labels inside the 27 Euler-rotation gate boxes in panel (b). Remove all theta symbols, parentheses and subscripts. The first column must say "Rz" in each of its nine boxes; the second column "Ry" in each of its nine boxes; the third column "Rz" in each of its nine boxes. Keep all nine wires, nine sequential CNOTs, all arrows, formulas, other labels, colours, layout and panels exactly unchanged.

## Scientific consistency checks

- Main architecture: 359 input features; 9-qubit global readout; 12 circuit layers; 540 quantum parameters; 27 Pauli expectations; 120 + 27 = 147 concatenated features; residual 147→128→10 head plus base-logit skip; class correction after addition.
- Separate fixed-fusion panel: 32,410-parameter token-wise quantum classifier and 23,170-parameter Window MLP; corrected logits averaged 0.5/0.5; 55,581 reported scalars includes one fixed coefficient.
- Circuit: preparation once; Rz/Ry/Rz rotations, ordered CNOT ring 0→1,…,7→8,8→0, followed by angle re-uploading; layer repeats 12 times; q0 is least significant bit.
- The light curves are illustrative schematic curves, not observations from a selected object.
- Rotation parameters depend on layer and qubit; gate labels are abbreviated to Rz/Ry/Rz to avoid generated subscript errors.
- Amplitude encoding formula describes nonzero inputs. The implementation's zero-norm fallback remains in the manuscript, not expanded in the figure.
- These two assets were subsequently integrated into the manuscript as Figures 1 and 2. Their prompts are retained here for provenance.

