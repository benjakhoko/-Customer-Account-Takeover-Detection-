"""
app.py
Customer Account Takeover Detection - Flask Web Application
Loads the trained pipeline from models/model.pkl and exposes:
  /            -> Home
  /predict     -> Prediction form (GET) + prediction handling (POST)
  /dashboard   -> Aggregated stats from data/predictions.csv
  /about       -> Project information
"""

import os
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request
from flask import Flask, render_template, request, send_from_directory

# -----------------------------------------------------------------
# App setup
# -----------------------------------------------------------------
app = Flask(__name__)

MODEL_PATH = "models/model.pkl"
LOG_PATH   = "data/predictions.csv"
RESULT_DIR = "results"

# Fail fast if the model is missing
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run 'python train_model.py' first."
    )

model = joblib.load(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

# These MUST match the columns used during training (order matters for clarity)
CATEGORICAL_COLS = [
    "Country",
    "Region",
    "City",
    "Browser Name and Version",
    "OS Name and Version",
    "Device Type",
]
NUMERIC_COLS = [
    "Round-Trip Time [ms]",
    "ASN",
    "Login Successful",
    "Is Attack IP",
    "login_hour",
    "login_day_of_week",
]

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
def append_prediction(record: dict):
    """Append one prediction to the CSV log (creates the file if needed)."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    df = pd.DataFrame([record])
    if os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)


def load_history() -> pd.DataFrame:
    """Load the prediction log for the dashboard. Returns an empty frame if none."""
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(LOG_PATH)
    except Exception:
        return pd.DataFrame()


def parse_form(form) -> dict:
    """
    Extract, validate and convert the form fields.
    Raises ValueError with a friendly message on any problem.
    """
    # Required text fields
    required_text = ["country", "region", "city", "browser", "os", "device", "timestamp"]
    for field in required_text:
        if not form.get(field, "").strip():
            raise ValueError(f"Field '{field}' is required.")

    # Numeric fields
    try:
        rtt = float(form["rtt"])
        if rtt < 0:
            raise ValueError("Round-Trip Time must be non-negative.")
    except (KeyError, ValueError):
        raise ValueError("Round-Trip Time must be a positive number.")

    try:
        asn = int(form["asn"])
        if asn < 0:
            raise ValueError("ASN must be non-negative.")
    except (KeyError, ValueError):
        raise ValueError("ASN must be a non-negative integer.")

    # Booleans from radio/select
    login_successful = form.get("login_success") == "1"
    is_attack_ip     = form.get("is_attack_ip") == "1"

    # Timestamp -> derive login_hour and login_day_of_week
    dt = pd.to_datetime(form["timestamp"], errors="coerce", utc=True)
    if pd.isna(dt):
        raise ValueError("Invalid timestamp. Use format: YYYY-MM-DDTHH:MM")

    row = {
        "Round-Trip Time [ms]": rtt,
        "ASN": asn,
        "Login Successful": login_successful,
        "Is Attack IP": is_attack_ip,
        "login_hour": int(dt.hour),
        "login_day_of_week": int(dt.dayofweek),
        "Country": form["country"].strip(),
        "Region": form["region"].strip(),
        "City": form["city"].strip(),
        "Browser Name and Version": form["browser"].strip(),
        "OS Name and Version": form["os"].strip(),
        "Device Type": form["device"].strip(),
    }
    return row


# -----------------------------------------------------------------
# Routes
# -----------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    # POST -> run prediction
    try:
        row = parse_form(request.form)

        # Build a one-row DataFrame in the exact schema the pipeline expects
        X = pd.DataFrame([row])[NUMERIC_COLS + CATEGORICAL_COLS]

        pred  = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])  # P(ATO = 1)

        label = "Potentially Compromised Account" if pred == 1 else "Normal Account"

        # Persist for the dashboard
        append_prediction({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "prediction": pred,
            "label": label,
            "probability": round(proba, 4),
            **row,
        })

        return render_template(
            "result.html",
            label=label,
            prediction=pred,
            probability=round(proba * 100, 2),
            inputs=row,
        )

    except ValueError as ve:
        return render_template("predict.html", error=str(ve))
    except Exception as e:
        return render_template("predict.html", error=f"Unexpected error: {e}")


@app.route("/dashboard")
def dashboard():
    history = load_history()

    total       = len(history)
    compromised = int((history["prediction"] == 1).sum()) if total else 0
    normal      = total - compromised

    # Model performance table
    perf_path = os.path.join(RESULT_DIR, "model_comparison.csv")
    performance = []
    if os.path.exists(perf_path):
        performance = pd.read_csv(perf_path).to_dict(orient="records")

    # Recent predictions (last 10)
    recent = history.tail(10).iloc[::-1].to_dict(orient="records") if total else []

    return render_template(
        "dashboard.html",
        total=total,
        normal=normal,
        compromised=compromised,
        performance=performance,
        recent=recent,
        has_confusion=os.path.exists(os.path.join(RESULT_DIR, "confusion_matrix.png")),
        has_importance=os.path.exists(os.path.join(RESULT_DIR, "feature_importance.png")),
    )


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/results/<path:filename>")
def results_file(filename):
    """Serve generated PNG charts from the results/ folder."""
    return send_from_directory(RESULT_DIR, filename)


# -----------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)