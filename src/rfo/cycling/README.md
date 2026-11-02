# RFO cycling

Each cycle chooses Protenix gradient/MCMC optimization with probability
`backprop_fraction` (default 0.5), or Boltz-2 prediction otherwise. Both
branches then redesign unfixed chain-A residues with OFoldX MPNN and feed the
redesigned complex into the next cycle.

```text
Protenix gradient optimization ─┐
                          ├─ structure → OFoldX MPNN → next cycle
Boltz-2 prediction ────────┘
```

Protenix, Boltz-2, and MPNN run through OFoldX in the RFO driver's Python
environment. See the [project README](../../../README.md) and
[installation guide](../../../docs/installation.md) for setup.

## Inputs and branch selection

- `input_structure`: initial PDB or mmCIF complex; the designed protein is chain A.
- `optimizer_config`: Protenix gradient-branch config; defaults to
  `backprop/configs/cycle_ppi.yaml`.
- `loss`: optional Protenix objective override.
- `msa_paths`: per-chain unpaired MSA files for Boltz-2 and Protenix.
- `paired_msa_paths`: optional per-chain paired MSAs for Protenix row pairing.
- `templates`: optional per-chain structural templates for Protenix.

Paths in run configurations are relative to the working directory. Chain A
must be a standard protein with complete backbone atoms for OFoldX MPNN.
Partner chains are retained and are not redesigned. For cyclic chain A, OFoldX
reads cyclic conditioning from a terminal covalent closure in the input
structure. A cyclic flag without that bond is rejected.

`backprop_fraction: 0` uses Boltz-2 and MPNN; `1` uses Protenix and MPNN.
Intermediate values select a structure-prediction branch randomly each cycle.
Boltz-2 fixes contiguous chain-A regions above `template_plddt_threshold`
(default 80). Protenix fixes chain-A residues that contact chain B within 8 Å.
The MPNN adapter translates these masks to structure residue identifiers and
preserves partner chains. RFO carries the first generated candidate forward
when `mpnn_num_seqs` is greater than one.

## Outputs

```text
inputs/<stem>_<cycle>.pdb
recycle_<n>/
  protenix_config.yaml
  mpnn/
    mpnn_design.pdb
    mpnn_result.json
    mpnn_spec.json
  <stem>/
    history.json
    modelhub_pred/<step>.cif
<stem>_records.json
```

The Protenix objective can be selected from `pae_interface_mean`,
`pae_interface_min`, `pae_mean`, `pde_mean`, `plddt_mean`, and `iptm`. Confidence
scores are recorded alongside the selected objective. Diffusion sampling does
not participate in the sequence gradient; the differentiable trunk and
confidence heads provide gradients to the sequence input.
