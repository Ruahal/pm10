# Machine Learning-Based PM10 Air Pollution Prediction and Monitoring Dashboard — Guwahati

A complete, responsive web dashboard that analyzes Guwahati air-quality data
(2015–2020) from `GHY2020.xlsx`, trains machine-learning models to predict
**PM10**, and visualizes trends, station comparisons and pollutant
relationships. Built with **Python/Flask**, **Scikit-learn** and
**Chart.js** as a B.Tech CSE academic project.

> **Note:** The dataset contains a limited number of observations (36 rows,
> 6 stations × 6 years). Model results should be interpreted as an academic
> demonstration rather than a production-grade air-quality forecasting system.

---

## Features

- **Main dashboard** — summary cards (average / highest / lowest PM10, number
  of monitoring stations), PM10 trend chart with station selector, and average
  PM10 by year. All values are computed live from the Excel dataset.
- **PM10 prediction** — enter Year, Monitoring Station, SO₂ and NO₂; the Flask
  backend (`POST /predict`) returns the model-predicted PM10 (µg/m³) with a
  CPCB-based pollution status indicator. Clearly labelled as a model
  prediction, not a measurement.
- **Pollution trends** — per-station and all-station PM10 trends, yearly
  averages, and SO₂ vs PM10 / NO₂ vs PM10 scatter plots with least-squares
  trendlines (association only — no causal claims).
- **Location analysis** — average PM10 per station, sortable high→low /
  low→high, with full station names in tooltips and a summary table.
- **Model performance** — comparison table and chart of MAE / RMSE / R² for
  Linear Regression, Random Forest and SVR, plus an Actual vs Predicted chart
  for each model. All metrics are calculated from the real dataset on a
  held-out test split.
- **Dataset Explorer** — searchable, sortable, paginated table with year and
  station filters plus CSV download.
- **About page** — objective, dataset, technologies, preprocessing summary and
  limitations.
- **Responsive UI** — navy/white theme, sidebar navigation (hamburger menu on
  mobile), cards, toasts, loading states and accessible labels.

## Pollution status scale

PM10 categories follow the **CPCB (Central Pollution Control Board, India)
National Air Quality Index** break-points for PM10 (24-h average, µg/m³):

| Range      | Status      |
|------------|-------------|
| 0–50       | Good        |
| 51–100     | Satisfactory|
| 101–250    | Moderate    |
| 251–350    | Poor        |
| 351–430    | Very Poor   |
| 431–500    | Severe      |

## Project structure

```text
pm10-dashboard/
├── app.py               # Flask application (pages + JSON APIs + prediction)
├── train_model.py       # Trains LR / Random Forest / SVR, saves model + metrics
├── preprocess.py        # Excel parsing + data validation (documented removals)
├── requirements.txt
├── README.md
├── data/
│   └── GHY2020.xlsx     # Source dataset (2015–2020, 6 stations)
├── models/
│   ├── pm10_model.pkl   # Best model bundle (Joblib)
│   └── metrics.json     # Real MAE / RMSE / R² for all three models
├── templates/           # index, prediction, trends, locations, models,
│                        # dataset, about, error
└── static/
    ├── css/style.css
    └── js/              # charts.js (shared), dashboard.js, prediction.js, dataset.js
```

## Setup (VS Code / local)

Prerequisites: **Python 3.10+**.

```bash
cd pm10-dashboard

# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the models (one-time; app.py also auto-trains if models are missing)
python train_model.py

# 4. Run the dashboard
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

Charts and icons load from CDNs (Chart.js, Bootstrap Icons, Google Fonts), so
an internet connection is needed for full styling.

## Backend API

| Route                   | Method | Description                                    |
|-------------------------|--------|------------------------------------------------|
| `/` , `/dashboard`      | GET    | Main dashboard page                            |
| `/prediction`           | GET    | Prediction page                                |
| `/predict`              | POST   | JSON `{year, location, so2, no2}` → `{predicted_pm10, status, model_used}` |
| `/api/summary`          | GET    | Avg/max/min PM10, station count, preprocessing summary |
| `/api/trends`           | GET    | PM10 by station-year + yearly averages         |
| `/api/location-analysis`| GET    | Average PM10/SO₂/NO₂ per station               |
| `/api/relationships`    | GET    | SO₂ vs PM10 and NO₂ vs PM10 scatter data       |
| `/api/model-performance`| GET    | MAE/RMSE/R² + actual-vs-predicted per model    |
| `/api/dataset`          | GET    | Full dataset rows                              |
| `/api/download-csv`     | GET    | Dataset as CSV download                        |

Invalid or missing prediction inputs return HTTP 400 with a descriptive
`error` message.

## Machine learning details

- **Features:** Year, Location (one-hot encoded), SO₂, NO₂ → **Target:** PM10
- **Models:** Linear Regression, Random Forest (300 trees), SVR (RBF, with
  StandardScaler). Each is wrapped in a scikit-learn `Pipeline`.
- **Split:** `train_test_split(test_size=0.25, random_state=42)` → 27 train /
  9 test observations.
- **Metrics:** MAE, RMSE, R² computed on the test split; the model with the
  lowest test RMSE is saved via Joblib to `models/pm10_model.pkl`.
- **Data validation** (`preprocess.py`): missing values, duplicate rows,
  non-numeric coercion and impossible values are all checked; any removals are
  counted and reported in the preprocessing summary (About page and
  `/api/summary`) — nothing is deleted silently.

To retrain (e.g. after editing the Excel file):

```bash
python train_model.py
```

## Disclaimer

This is an academic project. Predictions come from a model trained on 36
historical observations and are not actual measurements or regulatory
statements. No health advice is implied by the status categories.
