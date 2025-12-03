#!/usr/bin/env bash
# Generate training history plots for every run directory under a checkpoints root.
# Usage: bash scripts/plotting/run_all_training_plots.sh /work3/s225224/ucf-crime/checkpoints

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
  # Skip non-directories
  [[ -d "$run_dir" ]] || continue

  # Only process runs that have a training_history.csv
  if [[ ! -f "${run_dir%/}/training_history.csv" ]]; then
    continue
  fi

  echo "Plotting: ${run_dir%/}"
  uv run python -m scripts.plotting.plot_training_history "${run_dir%/}"
done

echo "Done."
