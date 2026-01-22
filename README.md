
# Quantum Autoencoders for Anomaly Detection

This repository contains the code, experiments, and documentation accompanying the thesis:

**Quantum Autoencoders for Anomaly Detection: Latent Space Structure, Regularization, and the BrainBox Layer**

The project investigates quantum autoencoder (QAE) architectures for anomaly detection, with a particular focus on latent space structure, generalization behavior, and false positive rates. A novel BrainBox layer is proposed and evaluated as an architectural improvement over standard QAE designs.

---

## Overview

Quantum autoencoders are evaluated as feature extractors for anomaly detection rather than purely as compression mechanisms. The study emphasizes the role of latent space separability, regularization, and architectural constraints in achieving reliable anomaly detection performance.

The repository is designed to support full reproducibility of the experimental results reported in the thesis.

---

## Key Contributions

- Systematic evaluation of multiple quantum autoencoder architectures for anomaly detection
- Analysis of latent space separability and compression behavior
- Identification of limitations of shallow and unregularized QAE designs
- Proposal of the BrainBox layer to improve latent structure and robustness
- Comparison between classical kNN and quantum kNN in the learned latent space
- Evaluation using PR-AUC, F1-score, precision, recall, and false positive rate

---

## Methodology

### Quantum Autoencoders
- Variational quantum circuits used for dimensionality reduction
- Bottleneck latent space trained via reconstruction loss
- Performance evaluated based on anomaly separability rather than reconstruction accuracy alone

### BrainBox Layer
- Structured latent bottleneck architecture
- Designed to improve latent separability and reduce false positives
- Evaluated with ℓ2 regularization (λ = 10⁻⁴)

### Anomaly Detection
- k-Nearest Neighbors applied in latent space
- Both classical kNN and quantum kNN evaluated under identical conditions

---

## Datasets

Experiments are conducted on standard anomaly detection benchmarks, including:

- **KDD Cup 1999 (KDD’99)**

Selected features include:
- Duration
- Protocol type
- Service
- Source bytes
- Destination bytes
- Flag
- Count-based traffic statistics

All preprocessing steps are implemented in the repository and are fully reproducible.

---

## Evaluation Metrics

The following metrics are reported consistently across experiments:

- Precision–Recall Area Under the Curve (PR-AUC)
- F1-score
- Precision
- Recall
- False Positive Rate (FPR)

These metrics are chosen to account for class imbalance and the practical requirements of anomaly detection.


