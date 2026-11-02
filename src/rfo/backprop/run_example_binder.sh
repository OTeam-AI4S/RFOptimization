#!/usr/bin/env bash
# Run a Protenix optimizer config in an environment with RFO and OFoldX installed.
set -euo pipefail
CONFIG="${1:?Usage: bash run_example_binder.sh /path/to/optimizer.yaml}"
CONFIG_DIR="$(cd "$(dirname "$CONFIG")" && pwd)"
CONFIG_NAME="$(basename "$CONFIG" .yaml)"
python -m rfo.backprop.optimize --config_name "$CONFIG_NAME" --config_path "$CONFIG_DIR"
