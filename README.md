# 1D Convolutional Neural Network for Automated Arrhythmia Classification

This repository contains the complete implementation, data preprocessing, and training pipelines for a deep 1D Convolutional Neural Network (1D-CNN) designed to detect cardiac arrhythmias from raw electrocardiogram (ECG) waveforms. The network is built using PyTorch and evaluated on a patient-segmented subset of the MIT-BIH Arrhythmia Database.

## Project Summary

* **Objective:** Binary classification distinguishing 'Normal' heartbeats from active 'Arrhythmias' to serve as an high-sensitivity triage baseline for clinical monitoring.
* **Dataset Characteristics:** 87,554 training beats and 21,892 test beats. To prevent data leakage, the training and test sets are completely segmented by patient ID, ensuring the network is evaluated on entirely unseen cardiac structures.
* **Signal Structure:** Each sample is an individual heartbeat waveform containing exactly 187 continuous voltage data points sampled at a frequency of **125 Hz**. This captures a fixed temporal window of **1,496 ms** ($\approx 1.5$ seconds) per clip.
* **Core Architecture:** A 4-layer deep 1D-CNN utilizing 5-element kernels and alternating 2-element max-pooling layers, feeding into a 128-neuron fully connected hidden layer.

---

## Physiological Architecture Mapping

Traditional 2D vision kernels do not translate directly to time-series signals without considering sampling physics. At $125\text{ Hz}$, each discrete sample represents exactly **$8.0\text{ ms}$** of cardiac electrical activity. The network's 5-element kernels were engineered to map precisely to distinct electrophysiological benchmarks:

* **Conv Layer 1 (Receptive Field: 5 samples / $40.0\text{ ms}$):** Operates as a local differential voltage checker ($\frac{dV}{dt}$). It isolates fast, sudden deflections, allowing it to trigger strongly on the steep onset of the **R-peak**.
* **Conv Layer 2 (Receptive Field: 13 samples / $104.0\text{ ms}$):** Perfectly spans the physical width of a healthy human **QRS complex** ($70\text{--}100\text{ ms}$), capturing localized ventricular depolarization geometry.
* **Conv Layer 3 (Receptive Field: 29 samples / $232.0\text{ ms}$):** Broadens context to observe the QRS complex along with its surrounding baseline deflections (capturing either the preceding **PR interval** or the following **ST segment**).
* **Conv Layer 4 (Receptive Field: 61 samples / $488.0\text{ ms}$):** Spans nearly **half a second ($\approx 0.49\text{ s}$)** of continuous tracking, covering roughly **32.6% of the entire 187-sample recording**. This gives the final convolutional layer a macro-level view to differentiate narrow normal beats from wide, slurred premature ventricular contractions (PVCs).

---

## Experimental Iteration and Optimization History

The baseline configuration of the model faced standard real-world medical data hurdles: high class imbalance (83% normal vs. 17% arrhythmia) and an asymmetrical cost matrix where **False Negatives** (undetected arrhythmias) present a critical patient safety risk. Optimization progressed through four distinct experimental phases:

### Run 1: Baseline Architecture
* **Setup:** Standard data loading, `nn.BCEWithLogitsLoss()`, and a default $0.50$ decision boundary threshold.
* **Result:** Solid feature extraction, but a clear vulnerability to clinical blindspots with **119 False Negatives (3.15%)**. Validation curves also revealed an overfitting horizon around **Epoch 8**, requiring validation-loss checkpointing.

### Run 2: Data-Level Class Rebalancing
* **Setup:** Modified the custom PyTorch data loader to dynamically resample the training batches, pulling a higher percentage of minority arrhythmia clips to equalize training exposure.
* **Result:** Shifted the error bias slightly, bringing False Negatives down to **97 (2.57%)** while False Positives ticked up to **110 (0.61%)**.

### Run 3: Loss-Level Cost-Sensitive Learning
* **Setup:** Passed a cost-penalty scaling scalar directly into the loss calculation: `nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))`. This explicitly forces backpropagation to penalize missed arrhythmias **10 times harder** than false alarms during training.
* **Result:** Strongly biased the network weights toward critical sensitivity, cutting False Negatives nearly in half down to **58 (1.54%)** at the expense of **560 False Positives (3.09%)**.

### Run 4: Strategic Boundary Threshold Shifting
* **Setup:** Combined the Run 3 weighted loss weights with a post-hoc shift of the decision boundary threshold from $0.50$ down to **$0.250$**. If the model calculates a probability $> 25\%$ that a waveform is abnormal, it actively triggers an arrhythmia alert.
* **Result (Final Model):** Achieved an exceptional **ROC-AUC of 0.997** and a clinical **Recall of 98.65%**, sending False Negatives to an absolute minimum of **41 (1.09%)**. This creates an ideal medical triage tool: highly sensitive to catching cardiac anomalies while keeping false alarms manageable for medical review staff.

---

## Technical Specifications & Performance Metrics

* **Inference Speed:** The complete 4-layer network executes evaluation across all 21,892 test samples in just 0.21 seconds. This evaluates to an exceptional per-heartbeat inference speed of **$< 10\ \mu\text{s}$ ($0.0096\text{ ms}$)**, rendering the architecture lightweight and fully optimized for continuous edge deployment on battery-powered medical wearables.
* **Final Triage Metrics:**
  * **ROC-AUC:** `0.997`
  * **Sensitivity / Recall:** `98.65%` ($3,733 / 3,774$)
  * **False Negatives:** `41`
  * **False Positives:** `1,164`

---

## Repository Structure

```text
├── Heartbeat_Training.ipynb    # Training logic, cross-validation metrics, and distribution plots
├── model_definition.py         # PyTorch implementation of the Arrhythmia1D convolutional model
├── .gitignore                  # Active tracking exclusion for heavy dataset .csv source files
└── README.md                   # Project documentation and engineering metrics
