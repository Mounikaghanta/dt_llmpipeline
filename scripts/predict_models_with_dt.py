import pandas as pd
import numpy as np
import joblib


# ==========================
# Load test datasets
# ==========================
sensor_df = pd.read_excel("data/test/test no labels.xlsx")
pred_df   = pd.read_excel("data/dt_predictions/test_dt_predictions.xlsx")

sensor_df["Date Time"] = pd.to_datetime(sensor_df["Date Time"], dayfirst=True)

pred_df["Date Time"]   = pd.to_datetime(pred_df["Date Time"], dayfirst=True)

df = sensor_df.merge(pred_df, on="Date Time", how="inner")
# Clean merged dataframe
df = df.rename(columns={
    "T (degC)_x": "T (degC)",
    "Tdew (degC)_x": "Tdew (degC)",
    "rh (%)_x": "rh (%)"
})

df = df.drop(columns=[
    "T (degC)_y",
    "Tdew (degC)_y",
    "rh (%)_y",
    "fault_label_x",
    "fault_label_y",
    "binary_label_x",
    "binary_label_y"
], errors="ignore")

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


# ==========================
# Load scaler
# ==========================
scaler = joblib.load("models/scaler.pkl")
X_scaled = scaler.transform(X)


# ==========================
# Models
# ==========================
model_names = [
    "MLP",
    "SVM",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "KNN",
    "LogReg"
]


for name in model_names:

    model = joblib.load(f"models/{name}_model.pkl")

    if name in ["MLP", "SVM", "KNN", "LogReg"]:
        preds = model.predict(X_scaled)
    else:
        preds = model.predict(X)

    df[f"{name}_pred"] = preds


# ==========================
# Save predictions
# ==========================
df.to_excel("outputs/all_model_predictions.xlsx", index=False)

print("Predictions saved.")
