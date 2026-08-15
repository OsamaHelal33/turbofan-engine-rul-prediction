# Turbofan Engine — Remaining Useful Life (RUL) Prediction

Predicts how many operating cycles a jet engine has left before failure, using sensor readings, for predictive maintenance decision-making.

## Problem

Unplanned equipment failure is expensive and dangerous in industrial settings. Instead of fixed maintenance schedules or waiting for failure, this project predicts **Remaining Useful Life (RUL)** from live sensor data — so maintenance can be scheduled just in time, not too early (wasteful) or too late (risky).

Dataset: NASA's CMAPSS Turbofan Engine Degradation dataset — 100 simulated engines run from healthy to failure, with 21 sensor readings recorded per operating cycle.

## Approach

1. **Labeling** — built the RUL target for every row as `(engine's total lifespan) - (current cycle)`, capped at 125 cycles (degradation is roughly flat until late in an engine's life, so precise labels beyond that just add noise).
2. **Feature cleaning** — identified and dropped 7 sensors with zero variance (they never changed across an engine's life, so carried no predictive signal).
3. **Modeling** — compared Linear Regression vs Random Forest, split **by engine** (not by row) to avoid data leakage between train and test.
4. **Experiment tracking** — logged all runs (params, metrics, model artifacts) with MLflow.
5. **Deployment** — built a Streamlit app that takes sensor inputs and returns a predicted RUL with a risk status (Critical / Warning / Healthy).
6. **MLOps** — wrote a retrain-and-validate script that blocks a new model from replacing the current one if its error regresses beyond a set tolerance, wired into a GitHub Actions workflow to run automatically on every push. Also containerized the app with Docker for portable deployment.

## Results

| Model | MAE (cycles) | RMSE | R² |
|---|---|---|---|
| Linear Regression | 16.51 | 20.43 | 0.760 |
| Random Forest | **12.43** | **17.03** | **0.833** |

Random Forest outperforms Linear Regression, consistent with degradation being a nonlinear process that a linear model can't capture well.

## Key finding

The sensor with the highest **variance** (sensor_4) was *not* the sensor the model relied on most. Feature importance showed **sensor_11 accounts for ~65% of the model's decision-making** — a reminder that a sensor moving a lot doesn't automatically mean it's the most *useful* signal for prediction. Variance and predictive value are related but distinct, and it's worth checking both rather than assuming.

## Business takeaway

With an average prediction error of ~12 cycles, a reasonable maintenance policy would be to flag engines for service when predicted RUL drops below ~30 cycles — building in a safety margin above the model's typical error, rather than waiting until the prediction hits zero.

## Tech stack

Python, pandas, scikit-learn, MLflow, Streamlit, Docker, GitHub Actions

## Project structure

```
├── turbofan_app.py          # Streamlit deployment app
├── retrain_and_validate.py  # MLOps retrain + regression-check script
├── Dockerfile                # Containerization
├── requirements.txt
├── rul_model.pkl             # Trained model artifact
├── feature_cols.pkl
├── sensor_defaults.pkl
└── .github/workflows/
    └── model_validation.yml  # CI pipeline running the validation check
```

## Running locally

```
pip install -r requirements.txt
streamlit run turbofan_app.py
```

## Running with Docker

```
docker build -t turbofan-rul-app .
docker run -p 8501:8501 turbofan-rul-app
```
