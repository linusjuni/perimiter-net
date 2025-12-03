#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_r3d_temporal_a100
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 12:00
#BSUB -o outputs/output_r3d_temporal_a100_%J.out
#BSUB -e outputs/output_r3d_temporal_a100_%J.err

cd ~/projects/perimiter-net
mkdir -p outputs

uv sync

uv run python scripts/training/train_r3d_temporal.py
# IMPORTANT: Update batch_size=128 in train_r3d.py before submitting
