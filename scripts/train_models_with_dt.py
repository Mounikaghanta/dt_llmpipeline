import pandas as pd
import numpy as np
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler


# ==========================
# Load training datasets
# ==========================
sensor_df = pd.read_excel("data/8months_sensor_faults.xlsx")
pred_df   = pd.read_excel("data/8months_dt_predictions.xlsx")

sensor_df["Date Time"] = pd.to_datetime(sensor_df["Date Time"])
pred_df["Date Time"]   = pd.to_datetime(pred_df["Date Time"])

df = sensor_df.merge(pred_df, on="Date Time", how="inner")


# ==========================
# Build residual features
# ==========================
df["res_T"]  = df["T (degC)"] - df["T_pred"]
df["res_Td"] = df["Tdew (degC)"] - df["Td_pred"]
df["res_RH"] = df["rh (%)"] - df["RH_pred"]

df["res_mag"] = np.sqrt(
    df["res_T"]**2 +
    df["res_Td"]**2 +
    df["res_RH"]**2
)


feature_cols = [
    "T (degC)", "Tdew (degC)", "rh (%)",
    "res_T", "res_Td", "res_RH", "res_mag"
]

X = df[feature_cols].values
y = df["binary_label"].values


# ==========================
# Scaling
# ==========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "models/scaler.pkl")


# ==========================
# Define models
# ==========================
models = {

    "MLP": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500),

    "SVM": SVC(kernel="rbf", probability=True),

    "RandomForest": RandomForestClassifier(n_estimators=300),

    "XGBoost": XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss"
    ),

    "LightGBM": lgb.LGBMClassifier(n_estimators=400),

    "KNN": KNeighborsClassifier(n_neighbors=15),

    "LogReg": LogisticRegression(max_iter=2000)
}


# ==========================
# Train and save
# ==========================
for name, model in models.items():

    if name in ["MLP", "SVM", "KNN", "LogReg"]:
        model.fit(X_scaled, y)
    else:
        model.fit(X, y)

    joblib.dump(model, f"models/{name}_model.pkl")

print("All models trained and saved.")
