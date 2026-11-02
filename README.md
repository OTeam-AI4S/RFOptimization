# RFOptimization (RFO)

RFOptimization improves an initial biomolecular design through repeated
sequence optimization and structure prediction. Each cycle selects a
Protenix gradient-guided sequence optimization step with probability
`backprop_fraction`, or a Boltz-2 structure prediction step otherwise. OFoldX
ProteinMPNN or LigandMPNN then redesigns the unfixed residues of chain A, and
the redesigned complex seeds the next cycle.

RFO owns the optimization and cycling logic. It uses the
[RFO-compatible OFoldX fork](https://github.com/OTeam-AI4S/OFoldX-for-RFO),
specifically its `rfo-integration` branch, as an external Python package for
Protenix, Boltz-2, and MPNN inference. OFoldX and RFO must be installed in the
same Python environment; OFoldX is installed separately and is intentionally
not an RFO package dependency. Model weights are downloaded and cached by
OFoldX.

## Installation

Use Python 3.12 or newer. Install a PyTorch build suitable for your hardware,
then install RFO and OFoldX in the same environment:

```bash
python -m pip install -e .
python -m pip install "git+https://github.com/OTeam-AI4S/OFoldX-for-RFO.git@rfo-integration"
```

See the [installation guide](docs/installation.md) for environment setup,
model defaults, cache settings, and GPU notes.

## Run an optimization

From the RFO project root:

```bash
rfo --config src/rfo/cycling/configs/example.yaml
```

Or provide input and output paths directly:

```bash
rfo --input /path/to/start.cif --out-dir outputs/run1 \
    --backprop-fraction 0.5 --total-cycles 10
```

Chain **A** is the designed protein. Other chains are retained as partners.
The Protenix branch uses interface contacts to select fixed residues for MPNN;
the Boltz-2 branch fixes high-confidence regions. See the [cycling guide](src/rfo/cycling/README.md)
and [optimizer guide](src/rfo/backprop/README.md).

## Configuration

```yaml
input_structure: start.cif
out_dir: outputs/run1
backprop_fraction: 0.5  # 1 = Protenix only; 0 = Boltz-2 only
optimizer_config: src/rfo/backprop/configs/cycle_ppi.yaml
total_cycles: 10
loss: pae_interface_mean
msa_paths: {B: /path/to/target.a3m}
paired_msa_paths: {A: /path/to/binder.a3m, B: /path/to/target.a3m}
templates: {B: /path/to/target.cif}
mpnn_num_seqs: 1
mpnn_temperature: 0.1
seed: 42
assets:
  mpnn_model_type: protein_mpnn  # or ligand_mpnn
  # protenix_model: oteam/protenix  # optional model override
  # boltz_model: oteam/boltz2     # optional model override
  # mpnn_model: /path/to/ofoldx-model-repository
```

Protenix defaults to `oteam/protenix`, Boltz-2 to `oteam/boltz2`, and ProteinMPNN
to `oteam/proteinmpnn`. Set `mpnn_model_type: ligand_mpnn` to select
`oteam/ligandmpnn`. Model overrides are optional. Command-line flags override
YAML values; explicit `assets:` values override environment variables. Use
`rfo --help` to see supported flags and objectives.

The Boltz-2 adapter reads the complex from the input structure; AF3 JSON input
is not currently supported. Set `msa_paths` to provide per-chain MSAs. Cyclic
chain A requires a terminal covalent closure in the input structure. The
Protenix branch uses the structure's bond graph for cyclic-chain geometry.

## Outputs

`out_dir/` contains one `recycle_<n>/` directory per cycle, `inputs/` with
redesigned complexes, and `<name>_records.json` with per-cycle scores, timing,
and designed sequences. Each cycle's `mpnn/` directory contains the MPNN
handoff structure and sequence/design metadata. RFO carries the first MPNN
candidate forward when `mpnn_num_seqs > 1`; it does not rank the candidates.

## Citation

If you use this code or data in your work, please consider citing:

```bibtex
@article{zhang2026_rfoptimization,
    author = {Zhang, Odin and Wang, Jiaqi and Thompson, Tuscan Rock and You, Ziyi and Song, Zihao and DiMaio, Frank and Baker, David},
    title = {{RFOptimization}: Guiding Design Optimization with All-Atom Structure Prediction},
    elocation-id = {2026.09.04.749184},
    year = {2026},
    doi = {10.64898/2026.09.04.749184},
    publisher = {openRxiv},
    URL = {https://www.biorxiv.org/content/10.64898/2026.09.04.749184v2},
    eprint = {https://www.biorxiv.org/content/10.64898/2026.09.04.749184v2.full.pdf},
    journal = {bioRxiv}
}
```
