# MPI Catheter Tip Position Tracking using Transformer
Transformer-based 3D catheter tip tracking in Magnetic Particle Imaging.
## 📋 Overview

This repository contains the official implementation of a **Transformer-based deep learning model** for real-time 3D catheter tip position tracking using Magnetic Particle Imaging (MPI) signals. The model achieves high accuracy and efficiency for Standard, Bending and Heart-beat like modes.

### Key Features
- **Transformer Architecture** with multi-head attention variants
- **Data Augmentation** (amplitude scaling + Gaussian noise) for improved generalization
- **Mixed Precision Training** for faster convergence
- **Support for Standard (3×26929) and Bending (2×80787) modes** 



# Datasets
All the data used in this work are publicly available at
https://zenodo.org/records/3554935. These datasets were originally introduced in Griese et al
(2020), paper title : "In-Vitro MPI-Guided IVOCT Catheter Tracking in Real Time for Motion Artifact Compensation"

If you use this code, please cite:  

@article{10.1088/1361-6560/ae794f,
	author={M Khair, Abuobaida and Jiang, Wenjing and Yang, Xiaoli and Wildgruber, Moritz and Ma, Xiaopeng},
	title={Harmonic-aware transformer for real-time catheter localization in interventional procedures of magnetic particle imaging},
	journal={Physics in Medicine & Biology},
	url={http://iopscience.iop.org/article/10.1088/1361-6560/ae794f},
	year={2026},
	
	
	abstract={Objective. Magnetic Particle Imaging (MPI) enables real-time, radiation-free tracking of magnetic nanoparticle–coated instruments, making it highly suitable for interventional procedures. This study proposes a harmonic-aware transformer framework that directly predicts catheter tip positions from raw MPI voltage signals, eliminating the need for image reconstruction and reducing computational latency. Approach. The framework incorporates frequency-domain preprocessing to isolate the 2nd–8th drive-field harmonics, enhancing the signal-to-noise ratio (SNR) while preserving motion-relevant features. A transformer architecture with six encoder layers and eight attention heads is employed to learn spatio-temporal dependencies across the three receive axes (x, y, z) for accurate 3D position estimation. The model is trained on simulated MPI signals and evaluated on real in vitro datasets under standard, bending, and heartbeat-like motion conditions. Main results. The proposed method achieves sub-millimeter localization accuracy, with a minimum L2 error of 0.103±0.092 mm and mean absolute errors (MAE) of 0.039 ± 0.046 mm, 0.054±0.049 mm, and 0.060±0.044 mm along the (x,y,z) axes, respectively, for the bending dataset. Across all datasets, the MAE ranges from 0.165 mm to 0.655 mm, demonstrating consistent performance. The optimized inference achieves a latency of 0.55 ms per frame and a throughput of approximately 1800 frames/s, confirming real-time capability. Significance. Compared with conventional MPI-guided approaches relying on image reconstruction, the proposed framework provides improved accuracy, reduced latency, and enhanced robustness under complex motion conditions. These results highlight the potential of harmonic-aware transformer models as efficient and scalable solutions for real-time catheter localization in interventional MPI.}
}
