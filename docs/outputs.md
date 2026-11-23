# Outputs

The `out_dir` contains per-cycle predictions and MPNN handoffs, along with a JSON record for the run. Exact intermediate files vary by branch and selected optimizer.

```text
outputs/run1/
├── inputs/                       # Complex inputs passed between cycles
├── recycle_0/                    # First cycle outputs
│   ├── protenix_config.yaml      # Effective Protenix config, when selected
│   ├── <run-name>/               # Optimizer run outputs, when Protenix is selected
│   │   ├── history.json
│   │   └── modelhub_pred/        # Predicted structures
│   └── mpnn/                     # MPNN handoff structure and metadata
├── recycle_1/
└── <run-name>_records.json       # Per-cycle scores, timings, and designed sequences
```

The run record includes the selected method, status, cycle and step, objective and confidence metrics, timing, and MPNN sequence when available. Not every metric applies to every branch or objective.

When `mpnn_num_seqs` is greater than one, multiple candidate sequences may be generated. The current driver carries the first candidate into the next cycle and does not rank the candidate set.

Use the Protenix optimizer history to inspect objective progression and acceptance decisions. Inspect the cycle directories for structures and MPNN metadata. The workflow should be evaluated from successful completion, expected chain retention, finite structure coordinates, and the requested backend in the output records; an individual score or sequence is not a universal pass criterion.
