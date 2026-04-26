# DeepSpace-CNN: Exoplanet Transit Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![Astropy](https://img.shields.io/badge/Astropy-Astronomy-orange.svg)
![Status](https://img.shields.io/badge/Status-Active%20Development-success.svg)

## Abstract
This repository contains the architecture and training pipeline for a 1D Convolutional Neural Network (CNN) designed to autonomously identify exoplanetary transits from raw stellar photometry data. By processing time-series light curves from the Kepler and TESS missions, this model isolates the fractional drop in stellar flux ($\Delta F$) indicative of planetary orbits, distinguishing true transits from false positives like eclipsing binary systems and instrumental noise.

## Architecture Pipeline
1. **Data Ingestion:** Automated querying of the Mikulski Archive for Space Telescopes (MAST) via the `Lightkurve` API.
2. **Signal Processing:** * Removal of low-frequency stellar variability.
    * Outlier rejection and Savitzky-Golay filtering for noise reduction.
3. **Phase Folding:** Calculating the orbital period and folding the time-series data to amplify the signal-to-noise ratio.
4. **Deep Learning Model:** A custom 1D-CNN built in PyTorch that analyzes spatial hierarchies within the folded light curves to output a binary classification (Transit vs. Non-Transit).

## Dataset
The model trains on verified Kepler Objects of Interest (KOIs). The dataset is highly imbalanced (far more empty space than planets), which is mitigated during training using synthetic data augmentation and weighted loss functions.

## Environment Setup & Installation

This project is optimized for a CUDA-enabled WSL2 (Windows Subsystem for Linux) environment.

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/DeepSpace-CNN.git](https://github.com/YOUR_USERNAME/DeepSpace-CNN.git)
cd DeepSpace-CNN

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/WSL2

# Install dependencies
pip install -r requirements.txt
