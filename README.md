\---



\# RF Interference Recognition Research Pipeline



End-to-end research pipeline for \*\*synthetic RF interference generation, dataset validation, and CNN-based classification\*\*.



This repository supports a Master's thesis project on \*\*communication interference recognition using deep learning\*\*, with a focus on \*\*reproducible synthetic data generation and controlled evaluation\*\*.



\---



\## Project Overview



The system implements a \*\*deterministic MATLAB → Python pipeline\*\* for developing and evaluating RF interference classification models.



The workflow is designed to ensure:



\* deterministic synthetic signal generation

\* strict MATLAB ↔ Python data contracts

\* dataset validation before model training

\* reproducible experiments

\* extensibility toward \*\*open-set recognition\*\* and \*\*real-world RF datasets\*\*



The pipeline separates \*\*data generation\*\*, \*\*data validation\*\*, and \*\*model training\*\* to maintain scientific traceability.



\---



\## System Architecture



```

MATLAB

│

├── Clean Signal Generator

├── Impairment Layer

└── Dataset Export (.mat / HDF5)

&#x20;       │

&#x20;       ▼

Python

│

├── Data Loader \& Validation

├── Feature Extraction (STFT)

├── Dataset Assembly

└── CNN Training \& Evaluation

```



\---



\## Repository Structure



```

thesis\_project

│

├── artifacts/          # Generated datasets and experiment outputs

├── configs/            # Configuration files

├── contracts/          # MATLAB ↔ Python interface contracts

│

├── matlab/

│   ├── +clean/         # Clean signal generators

│   ├── +impaired/      # Impairment models

│   ├── +core/          # Shared MATLAB utilities

│   ├── export/         # Dataset export tools

│   └── tests/          # MATLAB verification scripts

│

├── python/

│   └── src/

│       ├── dataio/     # Dataset loaders and contracts

│       ├── preprocessing/

│       ├── models/     # CNN architectures

│       ├── train/      # Training pipeline

│       ├── validation/ # Dataset sanity checks

│       └── utils/

│

├── reports/            # Dataset figures and Statistical, experiment, and validation reports

├── scripts/            # Experiment runner scripts

├── specs/              # Dataset and signal specifications

├── env/                # Environment configuration

│

└── README.md

```



\---



\## Core Principles



This project enforces several \*\*strict research rules\*\*:



\* \*\*Deterministic data generation\*\* (seed-controlled)

\* \*\*Explicit MATLAB ↔ Python interface\*\*

\* \*\*No silent data assumptions\*\*

\* \*\*Validation gates before training\*\*

\* \*\*Reproducible experiment tracking\*\*



These rules ensure that model performance reflects \*\*learning quality\*\*, not dataset inconsistencies.



\---



\## Dataset Format



Synthetic signals are generated in MATLAB and exported as \*\*HDF5-backed `.mat` files\*\*.



Key dataset properties:



\* Sampling rate: defined in `signal\_spec`

\* Fixed signal length

\* Deterministic parameter generation

\* Metadata stored alongside signals

\* FNV-1a checksum used for determinism verification



Python performs \*\*strict validation before loading the dataset\*\*.



\---



\## Machine Learning Pipeline



The Python pipeline includes:



\* STFT feature extraction

\* CNN model architectures

\* training and evaluation utilities

\* dataset validation gates

\* experiment diagnostics



Baseline models currently implemented:



\* Baseline CNN

\* Residual CNN



\---



\## Research Goal



Develop a \*\*robust RF interference classification system\*\* that:



1\. Learns from controlled synthetic data

2\. Maintains full experiment reproducibility

3\. Supports extension to \*\*open-set interference recognition\*\*

4\. Enables evaluation on \*\*external RF datasets\*\*



\---



\## Status



Project is under active development as part of a Master's thesis.



Current focus:



\* synthetic dataset pipeline

\* MATLAB ↔ Python interface validation

\* CNN architecture experimentation



\---



\## Author



Master's Research Project

Information and Communication Engineering



\---



