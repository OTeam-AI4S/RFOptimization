# Configuration

RFO accepts a YAML configuration file and command-line overrides. Command-line values take precedence over YAML values. Model paths can be supplied under `assets:` in YAML, through the supported environment variables, or with command-line flags.

## Minimal configuration

```yaml
input_structure: /path/to/complex.cif
out_dir: outputs/design-01
backprop_fraction: 0.5
optimizer_config: src/rfo/backprop/configs/cycle_ppi.yaml
total_cycles: 10
loss: pae_interface_mean
mpnn_num_seqs: 1
mpnn_temperature: 0.1
seed: 42
assets:
  mpnn_model_type: protein_mpnn  # or ligand_mpnn
```

The bundled example is at `src/rfo/cycling/configs/example.yaml`.

## Run options

| Option | Purpose | Default |
| --- | --- | --- |
| `input_structure` / `--input` | Input PDB or mmCIF complex. Chain A is redesigned. | Required |
| `out_dir` / `--out-dir` | Directory for cycle outputs and records. | Required |
| `backprop_fraction` / `--backprop-fraction` | Probability of choosing Protenix instead of Boltz-2 each cycle. | `0.5` |
| `total_cycles` / `--total-cycles` | Number of outer RFO cycles. | `10` |
| `loss` / `--loss` | Protenix sequence-optimization objective. | Optimizer config value |
| `optimizer_config` / `--optimizer-config` | Protenix gradient optimizer configuration. | Bundled `cycle_ppi.yaml` |
| `num_samples` / `--num-samples` | Boltz-2 diffusion samples per prediction. | `1` |
| `mpnn_num_seqs` / `--mpnn-num-seqs` | Number of MPNN candidates to generate. | `1` |
| `mpnn_temperature` / `--mpnn-temperature` | MPNN sampling temperature. | `0.1` |
| `template_plddt_threshold` / `--template-plddt-threshold` | Confidence threshold used for Boltz-2 fixed regions. | `80` |
| `cyclic` / `--cyclic` | Mark designed chain as cyclic; input must contain closure bond. | `false` |
| `seed` / `--seed` | Random seed. | `42` |

See `rfo --help` for the exact command-line options available in the installed version.

## Protenix objectives

The Protenix objective may be selected from:

- `pae_interface_mean`
- `pae_interface_min`
- `pae_mean`
- `pde_mean`
- `plddt_mean`
- `iptm`

The selected objective guides the sequence optimization. Other confidence values can also be recorded with the cycle results.

## MSA and template inputs

MSAs and templates are optional. Provide paths keyed by chain identifier:

```yaml
msa_paths:
  B: /path/to/target.a3m
paired_msa_paths:
  A: /path/to/binder.a3m
  B: /path/to/target.a3m
templates:
  B: /path/to/target.cif
```

`msa_paths` provides per-chain unpaired alignments for model branches. `paired_msa_paths` and `templates` are used by the Protenix path. Paths are expanded and resolved from the current working directory when RFO starts.

## Model repositories

OFoldX manages model artifact download and caching. The current defaults and optional overrides are:

| Model | Default repository | Environment override | YAML asset key |
| --- | --- | --- | --- |
| Protenix | `oteam/protenix` | `OFOLDX_PROTENIX_MODEL` | `protenix_model` |
| Boltz-2 | `oteam/boltz2` | `BOLTZ_MODEL` | `boltz_model` |
| ProteinMPNN | `oteam/proteinmpnn` | `MPNN_MODEL` | `mpnn_model` |
| LigandMPNN | `oteam/ligandmpnn` | `MPNN_MODEL` | `mpnn_model` |

Choose the MPNN backend with `mpnn_model_type: protein_mpnn` or `mpnn_model_type: ligand_mpnn`. Set optional overrides in `assets:` or use corresponding CLI options. Hugging Face cache locations can be controlled with `HF_HOME` and `HF_HUB_CACHE`.

See [Installation](installation.md) for environment setup and the OFoldX integration branch requirement.
