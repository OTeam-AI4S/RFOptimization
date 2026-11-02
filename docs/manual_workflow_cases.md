# Manual workflow cases

These are end-to-end checks for the RFO → OFoldX model adapters. Run them one at a time from the RFO repository root. They are intentionally manual; no test runner is involved.

## Before starting

Use the Python environment in which both `rfo` and `ofoldx` import successfully. OFoldX Hub defaults are used automatically; no model-path exports are required. To override them, set only the variables you need:

```bash
# Optional model overrides; omit these to use OFoldX Hub defaults.
# export OFOLDX_PROTENIX_MODEL=oteam/protenix
# export BOLTZ_MODEL=oteam/boltz2
# export MPNN_MODEL=oteam/proteinmpnn
# export MPNN_MODEL_TYPE=protein_mpnn
```

Use local paths or other compatible OFoldX repositories only when you need to override the defaults.

Download the additional structure input from the official [RCSB entry for 2O9V](https://www.rcsb.org/structure/2O9V), or run:

```bash
mkdir -p "$HOME/rfo-manual-test/inputs"
curl -fL https://files.rcsb.org/download/2o9v-assembly1.cif \
  -o "$HOME/rfo-manual-test/inputs/2o9v_assembly1.cif"
```

2O9V is a small protein–peptide complex: Ponsin SH3 domain (chain A, 67 residues) and a Paxillin peptide (chain B, 10 residues). Its biological assembly contains these two protein chains without the zinc or sulfate components that blocked the previous candidates. RFO currently treats chain A as the sequence to redesign, so this is a workflow smoke test rather than a biological design recommendation.

## Case 1: Protenix branch with the 2O9V complex

This forces the differentiable Protenix branch and runs one outer RFO cycle:

```bash
./rfo \
  --config src/rfo/cycling/configs/example.yaml \
  --input "$HOME/rfo-manual-test/inputs/2o9v_assembly1.cif" \
  --out-dir "$HOME/rfo-manual-test/outputs/2o9v-protenix" \
  --backprop-fraction 1 \
  --total-cycles 1 \
  --num-samples 1 \
  --seed 42
```

## Case 2: Boltz branch with the 2O9V complex

This forces the Boltz branch on the same input, so the result isolates the backend change:

```bash
./rfo \
  --config src/rfo/cycling/configs/example.yaml \
  --input "$HOME/rfo-manual-test/inputs/2o9v_assembly1.cif" \
  --out-dir "$HOME/rfo-manual-test/outputs/2o9v-boltz" \
  --backprop-fraction 0 \
  --total-cycles 1 \
  --num-samples 1 \
  --seed 42
```

## Case 3: Repeat the known small protein–peptide input through Boltz

If the cleaned 1YCR input from the first manual run is still available, this checks the other backend on the earlier, smaller structure:

```bash
./rfo \
  --config src/rfo/cycling/configs/example.yaml \
  --input /tmp/rfo-ofoldx-input/1ycr_clean.pdb \
  --out-dir "$HOME/rfo-manual-test/outputs/1ycr-boltz" \
  --backprop-fraction 0 \
  --total-cycles 1 \
  --num-samples 1 \
  --seed 42
```

## What to check

For every run, confirm that the command exits successfully and that the output directory contains the cycle result, model prediction, and MPNN result. Check the run log/metrics to confirm the requested backend was selected. Compare the input and final structures to ensure the run produced finite coordinates and retained the expected protein chains; for 8S1X, also check whether actinonin remains represented in the output.

MPNN may report that no sequence sampling was needed if the configured pLDDT threshold fixes every residue. That is a valid workflow outcome; verify the reported fixed-residue count and whether an MPNN result was emitted. For the 2O9V cases, confirm that both protein chains remain in the handoff complex. A successful cycle should not be judged by a particular sequence or score, since model output can vary with model version and hardware.
