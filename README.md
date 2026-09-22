# ANFIS-and-NARX-Surrogate-Models-for-Nonlinear-MPC-Based-Artificial-Pancreas-Control}

# ANFIS and NARX Surrogate Models for Nonlinear MPC-Based Artificial Pancreas Control

[![Conference](https://img.shields.io/badge/NeurIPS%202026-LXAI%20Workshop-darkblue.svg)](https://www.latinxinai.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/CasADi-PyTorch-orange.svg)](https://web.casadi.org/)

This repository contains the official implementation, data collection protocols, and safety-auditing benchmarks for the extended abstract:  
**"ANFIS and NARX Surrogate Models for Nonlinear MPC-Based Artificial Pancreas Control"**, presented at the **LatinX in AI (LXAI) Workshop at NeurIPS 2026**.

**Author:** Karen Tatiana Bravo Rodríguez  
**Keywords:** Nonlinear Control Systems, Model Predictive Control, Neuro-Fuzzy Systems, Safety-Critical ML, Artificial Pancreas

---

## 📌 Overview

Nonlinear Model Predictive Control (NMPC) for artificial pancreas systems traditionally uses simplified physiological models (e.g., the Bergman minimal model). While learned differentiable dynamics (such as neural networks) achieve near-perfect open-loop accuracy, their reliability inside closed-loop feedback optimizers remains critical.

This project demonstrates a critical failure mode: **a differentiable NARX model with open-loop $R^2 = 0.994$ induces dangerous insulin overdosing under closed-loop control during hypoglycemia recovery.**

### Key Insights
* **The "Extrapolation Gap":** Safe data-collection policies naturally avoid infusing insulin during low glucose. Consequently, no training data exists in the low-glucose / high-insulin state-action quadrant.
* **Physiological Gradient Inversion:** When queried out-of-distribution (OOD), the learned surrogate inverted the sign of insulin sensitivity (predicting higher glucose under maximal insulin).
* **Optimizer Exploitation:** The NMPC solver exploited this hallucinated artifact to minimize its cost, infusing **266 mU** of insulin (over 200× more than PID/Bergman baselines) into a hypoglycemic patient ($G_0 = 60\text{ mg/dL}$).

---

## 🔬 Architecture & Controller Comparison

| Metric / Feature | PID (Anti-Windup) | NMPC (Bergman) | NMPC (NARX Surrogate) |
| :--- | :--- | :--- | :--- |
| **Model Type** | Model-free | 3-state ODE Analytical | 2-layer Neural Network ($W=3$) |
| **Parameters** | 3 gains | Clinical constants | 2,753 weights |
| **Open-Loop $R^2$** | N/A | Analytical baseline | **0.994** (MAE: $1.37\text{ mg/dL}$) |
| **Meal Disturbance (ISE)** | Baseline | Good | **Lowest tracking error** |
| **Hypoglycemia Safety Audit** | Safe ($<1.3\text{ mU}$) | Safe ($<1.3\text{ mU}$) | **Critical Overdose ($266\text{ mU}$)** |

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/                 # Trajectories generated via simglucose
│   └── processed/           # Sliding-window datasets [G, u, D]
├── models/
│   ├── anfis/               # ANFIS architecture (448 parameters)
│   ├── narx/                # PyTorch NARX (2,753 parameters)
│   └── casadi_export/       # Differentiable symbolic CasADi graph
├── src/
│   ├── controllers/         # PID, NMPC-Bergman, and NMPC-NARX implementations
│   ├── data_collection/     # Stochastic persistent-excitation protocol
│   └── audit/               # Targeted out-of-distribution safety audit
├── notebooks/
│   └── results_analysis.ipynb
├── requirements.txt
├── LICENSE
└── README.md
