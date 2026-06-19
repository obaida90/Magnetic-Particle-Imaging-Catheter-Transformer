# MPI Catheter Tip Position Tracking using Transformer
Transformer-based 3D catheter tip tracking in Magnetic Particle Imaging.
## 📋 Overview

This repository contains the official implementation of a **Transformer-based deep learning model** for real-time 3D catheter tip position tracking using Magnetic Particle Imaging (MPI) signals. The model achieves high accuracy and efficiency for Standard, Bending and Heart-beat like modes.

### Key Features
- **Transformer Architecture** with multi-head attention variants
- **Data Augmentation** (amplitude scaling + Gaussian noise) for improved generalization
- **Mixed Precision Training** for faster convergence
- **Support for Standard (3×26929) and Bending (2×80787) modes** 
- **Comprehensive Evaluation** with MAE, L2 error, and visualization tools


# Datasets
All the data used in this work are publicly available at
https://zenodo.org/records/3554935. These datasets were originally introduced in Griese et al
(2020), paper title : "In-Vitro MPI-Guided IVOCT Catheter Tracking in Real Time for Motion Artifact Compensation"
