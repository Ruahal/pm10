"""Train and compare PM10 prediction models on the Guwahati dataset.

Features: Year, Location (one-hot encoded), SO2, NO2. Target: PM10.
Models: Linear Regression, Random Forest Regressor, Support Vector Regression.
All metrics (MAE, RMSE, R²) are computed from the real data on a held-out test
split and saved to models/metrics.json. The best model (lowest test RMSE) is
saved with Joblib to models/pm10_model.pkl together with its preprocessing.
"""

import json
import os

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from preprocess import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pm10_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

FEATURES = ["Year", "Location", "SO2", "NO2"]
TARGET = "PM10"
RANDOM_STATE = 42
TEST_SIZE = 0.25


def build_preprocessor(locations):
    return ColumnTransformer(
        transformers=[
            (
                "location_onehot",
                OneHotEncoder(categories=[list(locations)], handle_unknown="ignore"),
                ["Location"],
            )
        ],
        remainder="passthrough",
    )


def build_models(locations):
    return {
        "Linear Regression": Pipeline(
            [
                ("prep", build_preprocessor(locations)),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", build_preprocessor(locations)),
                ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)),
            ]
        ),
        "SVR": Pipeline(
            [
                ("prep", build_preprocessor(locations)),
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=10.0, epsilon=1.0)),
            ]
        ),
    }


def train_and_evaluate():
    data, prep_summary = load_dataset()

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    locations = sorted(data["Location"].unique())
    models = build_models(locations)

    results = {}
    trained = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        results[name] = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": rmse,
            "r2": float(r2_score(y_test, y_pred)),
            "actual": [float(v) for v in y_test],
            "predicted": [float(v) for v in y_pred],
        }
        trained[name] = pipeline

    best_name = min(results, key=lambda name: results[name]["rmse"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "model_name": best_name,
            "pipeline": trained[best_name],
            "features": FEATURES,
            "locations": locations,
            "years": sorted(data["Year"].unique().tolist()),
        },
        MODEL_PATH,
    )

    payload = {
        "best_model": best_name,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "results": results,
        "preprocessing_summary": prep_summary,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Trained {len(models)} models on {len(data)} observations "
          f"({len(X_train)} train / {len(X_test)} test).")
    print(f"{'Model':<20}{'MAE':>10}{'RMSE':>10}{'R²':>10}")
    for name, metrics in results.items():
        print(f"{name:<20}{metrics['mae']:>10.2f}{metrics['rmse']:>10.2f}{metrics['r2']:>10.3f}")
    print(f"\nBest model (lowest RMSE): {best_name}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    return payload


if __name__ == "__main__":
    train_and_evaluate()
