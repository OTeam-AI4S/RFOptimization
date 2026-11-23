# Workflow

RFO alternates between a structure-prediction step and an OFoldX MPNN sequence-redesign step. The RFO driver controls branch selection, sequence objectives, fixed-residue masks, and the handoff between cycles. Model pipelines are supplied by the RFO-compatible OFoldX integration branch.

## Branch selection

For each cycle, RFO selects the Protenix gradient branch with probability `backprop_fraction`; otherwise it selects Boltz-2.

| `backprop_fraction` | Behavior |
| ---: | --- |
| `1` | Always use Protenix gradient optimization, then MPNN. |
| `0` | Always use Boltz-2, then MPNN. |
| Between `0` and `1` | Randomly choose a branch independently for each cycle. |

## Protenix gradient branch

RFO optimizes the chain-A sequence representation against a configurable Protenix confidence objective. OFoldX supplies Protenix's differentiable model path. The Protenix weights remain frozen; gradients flow through the trunk and confidence heads to the input sequence representation. Diffusion sampling is not part of this gradient calculation. RFO retains its optimization, mutation proposal, and accept/reject logic.

After optimization, RFO obtains a structure for the MPNN handoff. The optimizer supports objectives such as `pae_interface_mean`, `pae_interface_min`, `pae_mean`, `pde_mean`, `plddt_mean`, and `iptm`; see [Configuration](configuration.md).

## Boltz-2 branch

Boltz-2 predicts the complex structure through OFoldX. The branch can use input MSA files. It does not run the Protenix sequence-gradient optimizer. RFO uses the predicted structure and confidence information to identify the chain-A regions to keep fixed during the subsequent MPNN redesign.

The current adapter reads the complex from the input structure; AF3 JSON input is not supported. Cyclic chain A requires a terminal covalent closure in the input structure.

## MPNN redesign and next cycle

Both prediction branches pass a structure and a fixed-residue mask to OFoldX ProteinMPNN or LigandMPNN. MPNN redesigns unfixed positions on chain A. Partner chains are retained. If multiple MPNN sequences are requested, RFO carries the first candidate forward; it does not rank those candidates.

The resulting redesigned complex becomes the input for the next cycle. RFO repeats until `total_cycles` is reached.

## Input expectations

- Input structure is a PDB or mmCIF complex.
- Chain A is the standard-protein binder or design chain and must have complete backbone atoms for OFoldX MPNN.
- Other chains remain as partners and are not redesigned.
- MSA and template paths in configuration are resolved relative to the directory from which `rfo` is launched.
- For cyclic chain A, the input structure must encode a terminal covalent closure; setting the cyclic flag alone is not enough.

See [Manual Runs](manual_workflow_cases.md) for example commands and checks.
