# Perimiter-Net: Weakly-Supervised Video Anomaly Detection on UCF-Crime

This repository implements a two-stream Multiple Instance Learning (MIL) pipeline for weakly-supervised anomaly detection in surveillance videos using the UCF-Crime dataset.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

```bash
uv sync
```

## Usage

- **Training:** See `scripts/training/train_mil.py` for MIL training and `scripts/training/train_r3d.py` for R3D backbone training.
- **Feature Extraction:** Use `scripts/feature_extraction.py` to extract features from videos.
- **Evaluation:** Run `scripts/evaluation/weight_fusion_search.py` for late fusion and frame-level AUC.
- **Notebook:** See `notebooks/demo.ipynb` for a step-by-step reproducible workflow.

## HPC Note

Model training and large-scale evaluation are performed on GPU nodes of the HPC cluster. The provided notebook runs on CPU for demonstration and reproducibility.

## Project Structure

```plaintext
src/                # Core models, datasets, utilities
scripts/            # Training, feature extraction, evaluation scripts
jobs/               # Job submission to GPU
notebooks/          # Reproducible results notebook
pyproject.toml      # Dependencies
README.md           # Project overview
```

## Citation

If you use this codebase, please cite the original UCF-Crime dataset and relevant papers.

---

**Contact:** For questions, send an e-mail to <s225224@dtu.dk>
