# Quickstart

This guide runs one outer RFO cycle using the bundled example complex. It assumes RFO and the compatible OFoldX integration branch are installed in the same activated Python 3.12+ environment. See [Installation](installation.md) first.

## 1. Check the command

From the RFO repository root:

```bash
rfo --help
```

## 2. Run one cycle

The bundled configuration points to a small example CIF and writes output under `outputs/example`. The first run may download model artifacts through OFoldX and can take longer than later runs.

```bash
rfo --config src/rfo/cycling/configs/example.yaml --total-cycles 1
```

The default `backprop_fraction: 0.5` randomly selects Protenix or Boltz-2 for the structure step. Set it to `1` to force Protenix or `0` to force Boltz-2:

```bash
# Protenix gradient branch
rfo --config src/rfo/cycling/configs/example.yaml \
  --backprop-fraction 1 --total-cycles 1

# Boltz-2 structure-prediction branch
rfo --config src/rfo/cycling/configs/example.yaml \
  --backprop-fraction 0 --total-cycles 1
```

Both branches then run the configured OFoldX MPNN model to redesign unfixed residues of chain A.

## 3. Use your own complex

Provide a PDB or mmCIF input, where chain A is the protein to redesign and the remaining chains are partners. Copy the example YAML and update the input and output paths, or pass them directly:

```bash
rfo \
  --input /path/to/complex.cif \
  --out-dir /path/to/runs/design-01 \
  --backprop-fraction 1 \
  --total-cycles 1 \
  --seed 42
```

Input requirements and optional MSA/template conditioning are described in [Workflow](workflow.md) and [Configuration](configuration.md).
