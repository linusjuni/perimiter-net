#!/bin/sh
#BSUB -q hpc
#BSUB -J extract_ucf_parallel
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 04:00
#BSUB -o outputs/extract_par_%J.out
#BSUB -e outputs/extract_par_%J.err

cd ~/projects/perimiter-net
mkdir -p outputs

# Load environment
uv sync

# Run the PARALLEL script
uv run python scripts/scratch/ucf/create_splits.py