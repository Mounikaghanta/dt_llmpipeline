import pandas as pd
import joblib

from dt_llm.digital_twin.dt_predictor_v2 import DigitalTwinV2
from dt_llm.decision.anomaly_gate import AnomalyGate

# ==========================
# Load data
# ==========================
df = pd.read_excel("data/test/test_sensor_data.xlsx")
df["Date Time"] = pd.to_datetime(df["Date Time"])

# ==========================
# Load models
# ==========================
dt = DigitalTwinV2("models/dt_v2_model_8mo.pkl")

classifier = joblib.load("models/XGBoost_model.pkl")
scaler = joblib.load("models/scaler.pkl")

gate = AnomalyGate()

# ==========================
# Initialize DT state
# ==========================
first = df.iloc[0]

dt.update_state(
    first["T (degC)"],
    first["Tdew (degC)"],
    first["rh (%)"]
)

results = []

# ==========================
# Streaming loop
# ==========================
for i in range(1, len(df)):

    row = df.iloc[i]
    ts = row["Date Time"]

    # DT prediction
    pred = dt.predict(ts)

    # Build classifier features
    res_T = row["T (degC)"] - pred["T_pred"]
    res_Td = row["Tdew (degC)"] - pred["Td_pred"]
    res_RH = row["rh (%)"] - pred["RH_pred"]

    res_mag = (res_T**2 + res_Td**2 + res_RH**2) ** 0.5

    X = [[
        row["T (degC)"],
        row["Tdew (degC)"],
        row["rh (%)"],
        res_T,
        res_Td,
        res_RH,
        res_mag
    ]]

    X_scaled = scaler.transform(X)

    #  Classifier decision
    label = classifier.predict(X_scaled)[0]

    #  Gate update
    gate.update_state(
        dt,
        {
            "T": row["T (degC)"],
            "Td": row["Tdew (degC)"],
            "RH": row["rh (%)"]
        },
        pred,
        label
    )

    results.append({
        "Date Time": ts,
        "T_pred": pred["T_pred"],
        "Td_pred": pred["Td_pred"],
        "RH_pred": pred["RH_pred"],
        "classifier_label": label
    })

# ==========================
# Save results
# ==========================
out = pd.DataFrame(results)
out.to_excel("outputs/streaming_results.xlsx", index=False)

print("Streaming simulation complete.")
