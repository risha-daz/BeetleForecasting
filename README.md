# BeetleForecasting: HDR Scientific Mood Challenge Sample

This repository contains training code and submissions for the [2025 HDR Scientific Mood Challenge: Beetles as Sentinel Taxa](https://www.codabench.org/competitions/9854/).

## Repository Structure

```
submission/
  <model weights>
  model.py
  requirements.txt
training/
  train.py
  evaluation.py
  model.py
  utils.py
notebooks/
  visualisation.ipynb
.gitignore
LICENSE
README.md
requirements.txt
pyproject.toml
```

> **IMPORTANT:**
> The model training in this repo requires a long time and prefers usage of GPU. Please reduce batch sizes/epochs to compensate accordingly

## Installation & Running

### Setup

Create a fresh conda environment and install dependencies:

```bash
conda create -n beetle-forecast -c conda-forge pip -y
conda activate beetle-forecast
pip install -r requirements.txt
```

### Training

There are 2 models to be trained, one is a contrastive model aimed at unifying 3 images and creating relevent features from the images. The second is the actual predictive model.

```bash
python training/contrastive.py
python training/train.py
```

The training data is available on Hugging Face: [Sentinel Beetles Dataset](https://huggingface.co/datasets/imageomics/sentinel-beetles).

### Evaluation

After training, evaluate your model locally:

```bash
python training/evaluation.py
```

## Training Data Exploration

`notebooks/visualisation.ipynb` contains some visualisation of what is happening in the model.

## References

Baselines are built off Dinov2:

* Oquab, Maxime, et al. "Dinov2: Learning robust visual features without supervision." arXiv preprint arXiv:2304.07193 (2023).
* [Sample repo from Y1 challenge](https://github.com/Imageomics/HDR-anomaly-challenge-sample)
