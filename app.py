"""Flask backend for the PM10 Guwahati Air Pollution Dashboard.

Loads the real dataset from data/GHY2020.xlsx, serves the dashboard pages and
provides JSON APIs for charts, the dataset explorer and PM10 prediction using
the Joblib-saved model. If no trained model exists yet, training runs once at
startup.
"""

import io
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from preprocess import load_dataset
from train_model import METRICS_PATH, MODEL_PATH, train_and_evaluate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Data & model loading
# ---------------------------------------------------------------------------

DATA, PREP_SUMMARY = load_dataset()

if not (os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH)):
    print("No trained model found - training now (one-time)...")
    train_and_evaluate()

MODEL_BUNDLE = joblib.load(MODEL_PATH)
with open(METRICS_PATH, "r", encoding="utf-8") as handle:
    METRICS = json.load(handle)

LOCATIONS = sorted(DATA["Location"].unique().tolist())
YEARS = sorted(DATA["Year"].unique().tolist())

# PM10 status categories based on the CPCB (Central Pollution Control Board,
# India) National Air Quality Index break-points for PM10 (24-hour average),
# expressed in µg/m³. Source: CPCB NAAQI documentation.
PM10_CATEGORIES = [
    (50, "Good", "good"),
    (100, "Satisfactory", "satisfactory"),
    (250, "Moderate", "moderate"),
    (350, "Poor", "poor"),
    (430, "Very Poor", "very-poor"),
    (float("inf"), "Severe", "severe"),
]


def pm10_status(value):
    for upper, label, css_class in PM10_CATEGORIES:
        if value <= upper:
            return {"label": label, "css_class": css_class}
    return {"label": "Severe", "css_class": "severe"}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
@app.route("/dashboard")
def dashboard():
    return render_template("index.html", active="dashboard")


@app.route("/prediction")
def prediction_page():
    return render_template("prediction.html", active="prediction",
                           locations=LOCATIONS, years=YEARS)


@app.route("/trends")
def trends_page():
    return render_template("trends.html", active="trends", locations=LOCATIONS)


@app.route("/locations")
def locations_page():
    return render_template("locations.html", active="locations")


@app.route("/models")
def models_page():
    return render_template("models.html", active="models")


@app.route("/dataset")
def dataset_page():
    return render_template("dataset.html", active="dataset",
                           locations=LOCATIONS, years=YEARS)


@app.route("/about")
def about_page():
    return render_template("about.html", active="about",
                           prep_summary=PREP_SUMMARY,
                           n_rows=len(DATA))


# ---------------------------------------------------------------------------
# Prediction API
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be JSON with year, location, so2 and no2."}), 400

    errors = []
    try:
        year = int(payload.get("year"))
        if year not in YEARS:
            errors.append(f"Year must be one of {YEARS[0]}-{YEARS[-1]}.")
    except (TypeError, ValueError):
        year = None
        errors.append("Year is required and must be a number.")

    location = payload.get("location")
    if location not in LOCATIONS:
        errors.append("Please select a valid monitoring station.")

    def read_positive_number(key, label):
        try:
            value = float(payload.get(key))
        except (TypeError, ValueError):
            errors.append(f"{label} is required and must be a number.")
            return None
        if value < 0:
            errors.append(f"{label} cannot be negative.")
            return None
        return value

    so2 = read_positive_number("so2", "SO₂")
    no2 = read_positive_number("no2", "NO₂")

    if errors:
        return jsonify({"error": " ".join(errors)}), 400

    features = pd.DataFrame(
        [{"Year": year, "Location": location, "SO2": so2, "NO2": no2}]
    )[MODEL_BUNDLE["features"]]
    predicted = float(MODEL_BUNDLE["pipeline"].predict(features)[0])
    predicted = max(predicted, 0.0)

    return jsonify(
        {
            "predicted_pm10": round(predicted, 2),
            "status": pm10_status(predicted),
            "model_used": MODEL_BUNDLE["model_name"],
        }
    )


# ---------------------------------------------------------------------------
# Data APIs
# ---------------------------------------------------------------------------

@app.route("/api/summary")
def api_summary():
    max_row = DATA.loc[DATA["PM10"].idxmax()]
    min_row = DATA.loc[DATA["PM10"].idxmin()]
    return jsonify(
        {
            "average_pm10": round(float(DATA["PM10"].mean()), 2),
            "highest": {
                "pm10": float(max_row["PM10"]),
                "location": max_row["Location"],
                "year": int(max_row["Year"]),
            },
            "lowest": {
                "pm10": float(min_row["PM10"]),
                "location": min_row["Location"],
                "year": int(min_row["Year"]),
            },
            "stations": int(DATA["Location"].nunique()),
            "observations": int(len(DATA)),
            "years": YEARS,
            "locations": LOCATIONS,
            "status_scale": [
                {"max": (None if upper == float("inf") else upper), "label": label, "css_class": css}
                for upper, label, css in PM10_CATEGORIES
            ],
            "status_source": "CPCB National Air Quality Index (PM10, 24-h) break-points",
            "preprocessing_summary": PREP_SUMMARY,
        }
    )


@app.route("/api/trends")
def api_trends():
    pivot = DATA.groupby(["Year", "Location"])["PM10"].mean().reset_index()
    trends = {}
    for location in LOCATIONS:
        subset = pivot[pivot["Location"] == location].set_index("Year")["PM10"]
        trends[location] = [
            round(float(subset[year]), 2) if year in subset.index else None
            for year in YEARS
        ]
    yearly_avg = DATA.groupby("Year")["PM10"].mean().reindex(YEARS)
    return jsonify(
        {
            "years": YEARS,
            "locations": LOCATIONS,
            "trends_by_location": trends,
            "yearly_average": [round(float(v), 2) for v in yearly_avg],
        }
    )


@app.route("/api/location-analysis")
def api_location_analysis():
    by_station = (
        DATA.groupby("Location")
        .agg(avg_pm10=("PM10", "mean"), avg_so2=("SO2", "mean"),
             avg_no2=("NO2", "mean"), observations=("PM10", "count"))
        .reset_index()
        .sort_values("avg_pm10", ascending=False)
    )
    return jsonify(
        {
            "stations": [
                {
                    "location": row["Location"],
                    "avg_pm10": round(float(row["avg_pm10"]), 2),
                    "avg_so2": round(float(row["avg_so2"]), 2),
                    "avg_no2": round(float(row["avg_no2"]), 2),
                    "observations": int(row["observations"]),
                    "status": pm10_status(float(row["avg_pm10"])),
                }
                for _, row in by_station.iterrows()
            ]
        }
    )


@app.route("/api/relationships")
def api_relationships():
    return jsonify(
        {
            "so2_vs_pm10": [
                {"x": float(r["SO2"]), "y": float(r["PM10"]), "location": r["Location"], "year": int(r["Year"])}
                for _, r in DATA.iterrows()
            ],
            "no2_vs_pm10": [
                {"x": float(r["NO2"]), "y": float(r["PM10"]), "location": r["Location"], "year": int(r["Year"])}
                for _, r in DATA.iterrows()
            ],
        }
    )


@app.route("/api/model-performance")
def api_model_performance():
    results = METRICS["results"]
    return jsonify(
        {
            "best_model": METRICS["best_model"],
            "train_rows": METRICS["train_rows"],
            "test_rows": METRICS["test_rows"],
            "test_size": METRICS["test_size"],
            "models": [
                {
                    "name": name,
                    "mae": round(metrics["mae"], 2),
                    "rmse": round(metrics["rmse"], 2),
                    "r2": round(metrics["r2"], 3),
                    "actual": [round(v, 2) for v in metrics["actual"]],
                    "predicted": [round(v, 2) for v in metrics["predicted"]],
                }
                for name, metrics in results.items()
            ],
        }
    )


@app.route("/api/dataset")
def api_dataset():
    return jsonify(
        {
            "columns": ["Year", "Location", "SO2", "NO2", "PM10"],
            "rows": [
                {
                    "year": int(r["Year"]),
                    "location": r["Location"],
                    "so2": float(r["SO2"]),
                    "no2": float(r["NO2"]),
                    "pm10": float(r["PM10"]),
                    "status": pm10_status(float(r["PM10"])),
                }
                for _, r in DATA.iterrows()
            ],
            "locations": LOCATIONS,
            "years": YEARS,
        }
    )


@app.route("/api/download-csv")
def download_csv():
    buffer = io.StringIO()
    DATA.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        io.BytesIO(buffer.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="ghy_pm10_dataset.csv",
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith(("/api/", "/predict")):
        return jsonify({"error": "Endpoint not found."}), 404
    return render_template("error.html", code=404,
                           message="The page you are looking for does not exist."), 404


@app.errorhandler(500)
def server_error(error):
    if request.path.startswith(("/api/", "/predict")):
        return jsonify({"error": "Internal server error."}), 500
    return render_template("error.html", code=500,
                           message="Something went wrong on our side."), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
