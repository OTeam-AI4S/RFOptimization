# RFO sequence optimization

The Protenix branch optimizes the binder sequence by propagating gradients
through OFoldX Protenix's differentiable trunk and confidence heads. Protenix
model weights remain frozen; RFO updates the soft sequence representation and
retains its mutation proposal, objective, and accept/reject logic. Diffusion
sampling is outside the gradient path and is used to produce the structure for
the following MPNN step.

The optimizer is usually launched by the RFO cycling driver. For a direct run,
use a configuration file based on `configs/cycle_ppi.yaml`:

```bash
python -m rfo.backprop.optimize --config_name my_optimizer --config_path /path/to/configs
```

## Configuration

| Field | Meaning |
| --- | --- |
| `input` | Input structure or supported input directory |
| `output_path` | Destination for optimizer results |
| `algorithm` | `gradient_mcmc` or `gradient_proposal` |
| `algorithm_config.opt_steps.total_steps` | Number of optimization steps |
| `protenix_model` | OFoldX Protenix repository or local path; defaults to `oteam/protenix` |
| `n_recycles` / `num_steps` | Protenix recycling and diffusion settings |
| `optimize_binder` / `interface_only` | Which residues may be optimized |
| `random_seq` | Randomize chain A at its existing length |
| `msa_paths` | Per-chain unpaired A3M/FASTA MSAs |
| `paired_msa_paths` | Per-chain paired alignments for cross-chain pairing |
| `fix_template_dict` | Mapping from target chain IDs to template structure paths |
| `loss_type` | Protenix objective, such as `pae_interface_mean`, `pae_interface_min`, `pae_mean`, `pde_mean`, `plddt_mean`, or `iptm` |

The cycle driver writes its effective per-cycle configuration under the run's
output directory and supplies input and output paths automatically.

## Outputs

Each optimizer run writes `history.json`, predicted structures under
`modelhub_pred/`, and an acceptance trajectory for MCMC optimization. The
lowest-loss history entry is selected for the MPNN handoff.

The Protenix adapter uses OFoldX's differentiable model path for sequence
gradients and the standard prediction path to write a structure. Template
inputs require an OFoldX Protenix checkpoint that supports templates. Random
chain-A initialization is capped at the input chain length.
