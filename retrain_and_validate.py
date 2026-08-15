"""
retrain_and_validate.py

The core idea of MLOps "model validation gates": every time you retrain,
automatically check the new model is at least as good as the last approved
one BEFORE letting it replace the production model. This is what a CI/CD
pipeline would run automatically (see github_actions_workflow.yml).

Usage:
    python retrain_and_validate.py
Exits with code 0 (pass) or 1 (fail) - this exit code is what GitHub Actions
or any CI system uses to decide whether to block a deployment.
"""

import sys
import json
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

BASELINE_FILE = "baseline_metrics.json"
# Allow the new model to be slightly worse due to randomness, but not
# meaningfully worse. This tolerance is a judgment call - stricter for
# critical systems, looser for experimental ones.
MAE_REGRESSION_TOLERANCE = 1.0  # cycles


def load_and_prepare_data():
    columns = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + \
              [f"sensor_{i}" for i in range(1, 22)]
    url = "https://raw.githubusercontent.com/ericlrf/rul/main/CMAPSSData/train_FD001.txt"
    df = pd.read_csv(url, sep=r"\s+", header=None, names=columns)
    df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]

    dead_sensors = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                     "sensor_16", "sensor_18", "sensor_19"]
    df = df.drop(columns=dead_sensors)
    df["RUL"] = df["RUL"].clip(upper=125)

    engine_ids = df["unit"].unique()
    train_ids, test_ids = train_test_split(engine_ids, test_size=0.2, random_state=42)
    train_df = df[df["unit"].isin(train_ids)]
    test_df = df[df["unit"].isin(test_ids)]

    feature_cols = [c for c in df.columns if c not in ["unit", "cycle", "RUL"]]
    return (train_df[feature_cols], train_df["RUL"],
            test_df[feature_cols], test_df["RUL"], feature_cols)


def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model


def main():
    print("Loading data and retraining model...")
    X_train, y_train, X_test, y_test, feature_cols = load_and_prepare_data()
    model = train_model(X_train, y_train)

    preds = model.predict(X_test)
    new_mae = mean_absolute_error(y_test, preds)
    print(f"New model MAE: {new_mae:.2f}")

    # ---- Compare against baseline ----
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            baseline = json.load(f)
        baseline_mae = baseline["mae"]
        print(f"Baseline MAE: {baseline_mae:.2f}")

        if new_mae > baseline_mae + MAE_REGRESSION_TOLERANCE:
            print(f"\n❌ FAILED: new model MAE ({new_mae:.2f}) is worse than "
                  f"baseline ({baseline_mae:.2f}) by more than the allowed "
                  f"tolerance ({MAE_REGRESSION_TOLERANCE}). Blocking deployment.")
            sys.exit(1)
        else:
            print(f"\n✅ PASSED: new model MAE is within tolerance of baseline.")
    else:
        print("\nNo baseline found - this run establishes the first baseline.")

    # ---- Save this as the new baseline + model artifacts ----
    with open(BASELINE_FILE, "w") as f:
        json.dump({"mae": new_mae}, f)

    joblib.dump(model, "rul_model.pkl")
    joblib.dump(feature_cols, "feature_cols.pkl")
    joblib.dump(X_train.median().to_dict(), "sensor_defaults.pkl")
    print("Saved updated model artifacts.")
    sys.exit(0)


if __name__ == "__main__":
    main()
