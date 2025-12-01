#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_r3d
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o outputs/output_r3d_%J.out
#BSUB -e outputs/output_r3d_%J.err

mkdir -p outputs

cd ~/projects/perimiter-net
uv sync

uv run python scripts/training/train_r3d.py