# Energy Forecasting & Anomaly Detection with SHAP Explainability

This project implements a solar power production forecasting system with anomaly detection and interpretable explanations using SHAP. The goal is to detect abnormal solar production patterns and explain predictions with model feature importance, helping with actionable insights.

---

## Features

- **Solar power forecasting** using XGBoost regression model.
- **Anomaly detection** based on residual thresholds from predictions.
- **SHAP explainability** to interpret individual anomaly predictions.
- **Streamlit dashboard** to visualize anomalies and SHAP explanations interactively.
- **Data pipeline** for processing historical solar production and weather features.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Packages: `pandas`, `numpy`, `xgboost`, `shap`, `streamlit`, `joblib`, `matplotlib`, `requests`

Install dependencies via:
```bash
pip install -r requirements.txt
