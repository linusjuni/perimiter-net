#!/usr/bin/env bash
# Generate MIL training history plots for every run directory under a MIL checkpoints root.
# Usage: bash scripts/plotting/run_all_mil_training_plots.sh /work3/s225224/ucf-crime/checkpoints/mil

set -euo pipefail

CHECKPOINT_ROOT=${1:-}

if [[ -z "$CHECKPOINT_ROOT" ]]; then
  echo "Usage: $0 <checkpoint_root>"
  exit 1
fi

if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
  echo "Checkpoint root not found: $CHECKPOINT_ROOT"
  exit 1
fi

for run_dir in "$CHECKPOINT_ROOT"/*/; do
  [[ -d "$run_dir" ]] || continue

  if [[ ! -f "${run_dir%/}/training_history.csv" ]]; then
    continue
  fi

  echo "Plotting: ${run_dir%/}"
  uv run python -m scripts.plotting.plot_mil_training_history "${run_dir%/}"
done

echo "Done."
