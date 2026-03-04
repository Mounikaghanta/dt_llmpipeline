import pandas as pd
import numpy as np
import math
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# Load clean dataset

df = pd.read_excel("data/train/8months.xlsx")

df["Date Time"] = pd.to_datetime(
    df["Date Time"],
    format="%d.%m.%Y %H:%M:%S"
)

df = df.sort_values("Date Time").reset_index(drop=True)


# physics

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


# training data

X = []
y_T = []
y_Td = []

for i in range(1, len(df)):

    prev = df.loc[i-1]
    curr = df.loc[i]

    hour_sin, hour_cos = hour_sin_cos(curr["Date Time"])
    doy_sin, doy_cos = doy_sin_cos(curr["Date Time"])

    features = [
        prev["T (degC)"],
        prev["Tdew (degC)"],
        prev["rh (%)"],
        hour_sin, hour_cos,
        doy_sin, doy_cos,
        prev["rh (%)"] * hour_sin,
        prev["rh (%)"] * hour_cos
    ]

    X.append(features)
    y_T.append(curr["T (degC)"])
    y_Td.append(curr["Tdew (degC)"])

X = np.array(X)
y_T = np.array(y_T)
y_Td = np.array(y_Td)


# Train DT models

T_model = GradientBoostingRegressor()
Td_model = GradientBoostingRegressor()

T_model.fit(X, y_T)
Td_model.fit(X, y_Td)


# Save model

os.makedirs("models", exist_ok=True)

joblib.dump(
    {"T_model": T_model, "Td_model": Td_model},
    "models/dt_v2_model_8mo.pkl"
)

print("DT model trained and saved to models/dt_v2_model_8mo.pkl")
