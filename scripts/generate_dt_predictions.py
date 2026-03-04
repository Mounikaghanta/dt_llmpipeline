import pandas as pd
import numpy as np
import math
import joblib

# ==========================
# Load dataset
# ==========================
df = pd.read_excel("data/train/8months_balanced_faults.xlsx")

df["Date Time"] = pd.to_datetime(
    df["Date Time"],
    format="%d.%m.%Y %H:%M:%S"
)

df = df.sort_values("Date Time").reset_index(drop=True)

# ==========================
# Load trained DT model
# ==========================
model = joblib.load("models/dt_v2_model_8mo.pkl")

# ==========================
# Helper functions
# ==========================
def hour_sin_cos(ts):
    h = ts.hour + ts.minute / 60.0
    ang = 2.0 * math.pi * (h / 24.0)
    return math.sin(ang), math.cos(ang)

def doy_sin_cos(ts):
    doy = ts.timetuple().tm_yday
    ang = 2.0 * math.pi * (doy / 365.0)
    return math.sin(ang), math.cos(ang)

def sat_vapor_pressure_hpa(T_c):
    return 6.112 * math.exp((17.62 * T_c) / (243.12 + T_c))

def rh_from_T_Td(T_c, Td_c):
    es = sat_vapor_pressure_hpa(T_c)
    e = sat_vapor_pressure_hpa(Td_c)
    rh = 100.0 * (e / max(es, 1e-6))
    return max(0.0, min(100.0, rh))

# ==========================
# Sequential DT prediction
# ==========================
T_preds = []
Td_preds = []
RH_preds = []

state_T = df.loc[0, "T (degC)"]
state_Td = df.loc[0, "Tdew (degC)"]
state_RH = df.loc[0, "rh (%)"]

for i in range(1, len(df)):

    ts = df.loc[i, "Date Time"]

    hour_sin, hour_cos = hour_sin_cos(ts)
    doy_sin, doy_cos = doy_sin_cos(ts)

    features = np.array([
        state_T,
        state_Td,
        state_RH,
        hour_sin, hour_cos,
        doy_sin, doy_cos,
        state_RH * hour_sin,
        state_RH * hour_cos
    ]).reshape(1, -1)

    T_pred = float(model["T_model"].predict(features)[0])
    Td_pred = float(model["Td_model"].predict(features)[0])

    if Td_pred > T_pred:
        Td_pred = T_pred

    RH_pred = rh_from_T_Td(T_pred, Td_pred)

    T_preds.append(T_pred)
    Td_preds.append(Td_pred)
    RH_preds.append(RH_pred)

    # Update state
    state_T = df.loc[i, "T (degC)"]
    state_Td = df.loc[i, "Tdew (degC)"]
    state_RH = df.loc[i, "rh (%)"]

# ==========================
# Save predictions
# ==========================
pred_df = df.iloc[1:].copy()

pred_df["T_pred"] = T_preds
pred_df["Td_pred"] = Td_preds
pred_df["RH_pred"] = RH_preds

pred_df.to_excel(
    "data/dt_predictions/8months_dt_predictions.xlsx",
    index=False
)

print("DT predictions generated.")
