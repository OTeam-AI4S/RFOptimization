# Install RFO and OFoldX

RFO implements the optimization and cycle driver. OFoldX supplies the
Protenix, Boltz-2, ProteinMPNN, and LigandMPNN model pipelines. The two
projects are installed separately but must be available in the **same Python
environment**, because RFO calls OFoldX in-process.

## 1. Create an environment

RFO requires Python 3.12 or newer. Create and activate a virtual environment
using your preferred environment manager. For GPU use, install the PyTorch
build that matches the machine's CUDA setup, following the current PyTorch
installation instructions.

## 2. Install RFO

From the RFO project root:

```bash
python -m pip install -e .
```

The editable install registers the `rfo` command and installs RFO's declared
runtime dependencies. To install a non-editable build, use `python -m pip
install .` instead.

## 3. Install OFoldX separately

In the same activated environment, install OFoldX explicitly. The current
OFoldX release can be installed from PyPI:

```bash
python -m pip install ofoldx
```

If that release does not include the APIs required by your RFO checkout, install
the latest OFoldX source instead:

```bash
python -m pip install "git+https://github.com/OTeam-AI4S/OFoldX.git"
```

OFoldX is intentionally not listed as an RFO dependency. The installed
OFoldX version must provide the Protenix differentiable pipeline, Boltz-2
pipeline, and MPNN pipelines used by this RFO version. Install or upgrade OFoldX
in the same environment as RFO before running the workflow.

## 4. Model repositories and cache

On first use, OFoldX downloads model artifacts from its configured Hub
repositories and caches them locally. The defaults are:

| Model | Default repository | Optional override |
| --- | --- | --- |
| Protenix | `oteam/protenix` | `OFOLDX_PROTENIX_MODEL` or `assets.protenix_model` |
| Boltz-2 | `oteam/boltz2` | `BOLTZ_MODEL` or `assets.boltz_model` |
| ProteinMPNN | `oteam/proteinmpnn` | `MPNN_MODEL` or `assets.mpnn_model` |
| LigandMPNN | `oteam/ligandmpnn` | `MPNN_MODEL` or `assets.mpnn_model` |

Set `MPNN_MODEL_TYPE=ligand_mpnn` or `assets.mpnn_model_type: ligand_mpnn`
to select LigandMPNN. To control Hugging Face downloads and cache placement,
configure `HF_HOME` and `HF_HUB_CACHE` before running RFO. Follow OFoldX's
documentation for model repository access and cache behavior.

## 5. Run a small manual workflow

From the RFO project root, first inspect the available options:

```bash
rfo --help
```

Then run one cycle with the example configuration after setting its input and
output paths:

```bash
rfo --config src/rfo/cycling/configs/example.yaml --total-cycles 1
```

`backprop_fraction: 1` selects Protenix and `backprop_fraction: 0` selects
Boltz-2. Both paths use OFoldX MPNN for the sequence redesign step. An
intermediate value randomly selects between the two structure-prediction
branches on each cycle.

RFO is designed for Linux with an NVIDIA GPU for model inference. Actual GPU
memory requirements depend on the input complex and model configuration.
