# app.py
from flask import Flask, request, render_template, send_file
import io
import pickle
import pandas as pd
from datetime import datetime

# ===== load models & metadata =====
DELAY_MODEL_PATH = "models/delay_model.pkl"
CANCEL_MODEL_PATH = "models/cancel_model.pkl"
UI_META_PATH = "models/ui_meta.pkl"
CSV_PATH = "Data/Processed_data15.csv"  # used only to fall back for dropdowns if meta missing

with open(DELAY_MODEL_PATH, "rb") as f:
    delay_model = pickle.load(f)
with open(CANCEL_MODEL_PATH, "rb") as f:
    cancel_model = pickle.load(f)

try:
    with open(UI_META_PATH, "rb") as f:
        meta = pickle.load(f)
    carriers = meta["carriers"]
    origins  = meta["origins"]
    dests    = meta["dests"]
except Exception:
    df = pd.read_csv(CSV_PATH, nrows=200000)
    carriers = sorted(df["AIRLINE_CODE"].dropna().unique().tolist())
    origins  = sorted(df["ORIGIN"].dropna().unique().tolist())
    dests    = sorted(df["DEST"].dropna().unique().tolist())

app = Flask(__name__)

def parse_date_any(s):
    """Accept dd/mm/yyyy or yyyy-mm-dd; return (year, month, day_of_week[1..7])."""
    s = s.strip()
    dt = None
    # try dd/mm/yyyy
    try:
        dt = datetime.strptime(s, "%d/%m/%Y")
    except Exception:
        pass
    # try yyyy-mm-dd (from <input type="date">)
    if dt is None:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            pass
    if dt is None:
        # last resort
        dt = pd.to_datetime(s, dayfirst=True, errors="raise").to_pydatetime()
    year = dt.year
    month = dt.month
    # Monday=1..Sunday=7
    day_of_week = (dt.weekday() + 1)
    return year, month, day_of_week

def make_feature_df(year, month, dow, carrier, origin, dest):
    return pd.DataFrame(
        [[year, month, dow, carrier, origin, dest]],
        columns=["year", "month", "day_of_week", "AIRLINE_CODE", "ORIGIN", "DEST"],
    )

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        carriers=carriers, origins=origins, dests=dests,
        prediction_delay=None, prediction_cancel=None,
        delay_proba=None, cancel_proba=None,
        threshold=0.5
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        flight_date = request.form.get("flight_date", "").strip()
        carrier = request.form.get("carrier", "").strip()
        origin = request.form.get("origin", "").strip()
        dest = request.form.get("dest", "").strip()
        threshold = float(request.form.get("threshold", 0.5))

        y, m, dow = parse_date_any(flight_date)
        X = make_feature_df(y, m, dow, carrier, origin, dest)

        # predict + probas
        delay_pred = int(delay_model.predict(X)[0])
        cancel_pred = float(cancel_model.predict_proba(X)[0,1])

        delay_label = "Delayed" if delay_pred == 1 else "On-Time"
        cancel_label = "Cancelled" if cancel_pred >= threshold else "Not Cancelled"

        return render_template(
            "index.html",
            carriers=carriers, origins=origins, dests=dests,
            prediction_delay=delay_label,
            prediction_cancel=cancel_label,
            delay_proba=float(delay_model.predict_proba(X)[0,1]),
            cancel_proba=cancel_pred,
            threshold=threshold,
            form_date=flight_date, form_carrier=carrier, form_origin=origin, form_dest=dest,
        )
    except Exception as e:
        return render_template(
            "index.html",
            carriers=carriers, origins=origins, dests=dests,
            prediction_delay=f"Error: {e}",
            prediction_cancel=None,
            delay_proba=None, cancel_proba=None,
            threshold=0.5
        )

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    """
    Accepts a CSV upload with columns: FL_DATE, AIRLINE_CODE, ORIGIN, DEST
    Returns a CSV with predictions: delay_label, delay_proba, cancel_label, cancel_proba
    """
    file = request.files.get("csvfile")
    if not file:
        return "No file uploaded", 400

    df = pd.read_csv(file)
    needed = ["FL_DATE", "AIRLINE_CODE", "ORIGIN", "DEST"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        return f"Missing columns: {miss}", 400

    # features from date
    # accept both dd/mm/yyyy and yyyy-mm-dd
    dates = pd.to_datetime(df["FL_DATE"], errors="coerce", dayfirst=True)
    # if many NaT, try fallback ISO
    if dates.isna().mean() > 0.5:
        dates = pd.to_datetime(df["FL_DATE"], errors="coerce")

    df_feat = pd.DataFrame({
        "year": dates.dt.year,
        "month": dates.dt.month,
        "day_of_week": dates.dt.dayofweek + 1,
        "AIRLINE_CODE": df["AIRLINE_CODE"],
        "ORIGIN": df["ORIGIN"],
        "DEST": df["DEST"],
    })
    df_feat = df_feat.dropna(subset=["year","month","day_of_week","AIRLINE_CODE","ORIGIN","DEST"])

    # align output with input rows
    preds_delay = delay_model.predict(df_feat)
    probs_delay = delay_model.predict_proba(df_feat)[:,1]
    probs_cancel = cancel_model.predict_proba(df_feat)[:,1]
    labels_delay = np.where(preds_delay==1, "Delayed", "On-Time")
    labels_cancel = np.where(probs_cancel>=0.5, "Cancelled", "Not Cancelled")

    out = df.copy()
    out.loc[df_feat.index, "delay_label"] = labels_delay
    out.loc[df_feat.index, "delay_proba"] = probs_delay
    out.loc[df_feat.index, "cancel_label"] = labels_cancel
    out.loc[df_feat.index, "cancel_proba"] = probs_cancel

    mem = io.BytesIO()
    out.to_csv(mem, index=False)
    mem.seek(0)
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="predictions.csv")

if __name__ == "__main__":
    app.run(debug=True)
