#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_r3d_c02516
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o outputs/output_r3d_c02516_%J.out
#BSUB -e outputs/output_r3d_c02516_%J.err

cd ~/projects/perimiter-net
mkdir -p outputs

uv sync

uv run python scripts/training/train_r3d.py
# IMPORTANT: Update batch_size=32 in train_r3d.py before submitting
