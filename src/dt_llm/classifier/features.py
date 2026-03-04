import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "T (degC)", "Tdew (degC)", "rh (%)",
    "res_T", "res_Td", "res_RH", "res_mag"
]

def build_features(sensor_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    sensor_df = sensor_df.copy()
    pred_df = pred_df.copy()

    sensor_df["Date Time"] = pd.to_datetime(sensor_df["Date Time"])
    pred_df["Date Time"] = pd.to_datetime(pred_df["Date Time"])

    df = sensor_df.merge(pred_df, on="Date Time", how="inner")

    df["res_T"]  = df["T (degC)"]    - df["T_pred"]
    df["res_Td"] = df["Tdew (degC)"] - df["Td_pred"]
    df["res_RH"] = df["rh (%)"]      - df["RH_pred"]

    df["res_mag"] = np.sqrt(
        df["res_T"]**2 +
        df["res_Td"]**2 +
        df["res_RH"]**2
    )

    return df
