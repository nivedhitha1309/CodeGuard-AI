<<<<<<< HEAD
\# CodeGuard AI



\*\*An Explainable Machine Learning Framework for Unified Code Risk, Technical Debt, and Bus Factor Prediction in Software Repositories\*\*



\## Overview



CodeGuard AI is a software maintenance risk prioritization framework designed to help developers identify files that may require maintenance attention.



It combines multiple software repository signals to assess and prioritize file-level risks with explainable evidence.



\## Phase 1 – Prototype



The initial prototype was developed and evaluated using the public GitHub repository `psf/requests`.



It integrates four risk signals:



1\. \*\*Bus Factor:\*\* Identifies developer knowledge concentration.

2\. \*\*Self-Admitted Technical Debt (SATD):\*\* Detects technical debt comments in source code.

3\. \*\*Static Code Risk:\*\* Analyzes code complexity and structural characteristics.

4\. \*\*Temporal ML Risk:\*\* Uses historical repository activity and defect-related information.



The signals are combined through an availability-aware fusion mechanism to generate file-level priorities.



\## Key Features



\- GitHub repository analysis

\- Repository history and contributor analysis

\- Technical debt detection

\- Static code analysis

\- Machine learning-based risk estimation

\- Multi-signal risk fusion

\- Explainable file prioritization

\- Streamlit dashboard



\## Technology Stack



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- GitPython

\- Radon

\- Streamlit

\- Matplotlib and Plotly



\## Project Status



Phase 1 prototype completed.



Phase 2 will focus on improving the framework, evaluating it across multiple repositories, conducting research experiments, and developing a more robust and explainable prototype.



\## Future Work



\- Multi-repository evaluation

\- Improved risk analysis

\- Ablation and comparative experiments

\- Reliability and validation studies

\- Enhanced dashboard and reporting

\- Research paper and thesis preparation



\## Disclaimer



CodeGuard AI provides maintenance risk prioritization and decision support. A high-risk score does not necessarily indicate the presence of a confirmed software defect.

=======
# CodeGuard AI

**An Explainable Machine Learning Framework for Unified Code Risk, Technical Debt, and Bus Factor Prediction in Software Repositories**

## Overview

CodeGuard AI is a software maintenance risk prioritization framework designed to help developers identify files that may require maintenance attention.

It combines multiple software repository signals to assess and prioritize file-level risks with explainable evidence.

## Phase 1 – Prototype

The initial prototype was developed and evaluated using the public GitHub repository `psf/requests`.

It integrates four risk signals:

1. **Bus Factor:** Identifies developer knowledge concentration.
2. **Self-Admitted Technical Debt (SATD):** Detects technical debt comments in source code.
3. **Static Code Risk:** Analyzes code complexity and structural characteristics.
4. **Temporal ML Risk:** Uses historical repository activity and defect-related information.

The signals are combined through an availability-aware fusion mechanism to generate file-level priorities.

## Key Features

- GitHub repository analysis
- Repository history and contributor analysis
- Technical debt detection
- Static code analysis
- Machine learning-based risk estimation
- Multi-signal risk fusion
- Explainable file prioritization
- Streamlit dashboard

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- GitPython
- Radon
- Streamlit
- Matplotlib and Plotly

## Project Status

Phase 1 prototype completed.

Phase 2 will focus on improving the framework, evaluating it across multiple repositories, conducting research experiments, and developing a more robust and explainable prototype.

## Future Work

- Multi-repository evaluation
- Improved risk analysis
- Ablation and comparative experiments
- Reliability and validation studies
- Enhanced dashboard and reporting
- Research paper and thesis preparation

## Disclaimer

CodeGuard AI provides maintenance risk prioritization and decision support. A high-risk score does not necessarily indicate the presence of a confirmed software defect.
>>>>>>> 637729c7d032c4cdcccaa5904e6da7fea5fcf560
