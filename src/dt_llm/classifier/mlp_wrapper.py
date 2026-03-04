import joblib
import numpy as np

from .features import FEATURE_COLUMNS, build_features


class MLPClassifierWrapper:

    def __init__(self, model_path: str, scaler_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, sensor_df, pred_df):
        df = build_features(sensor_df, pred_df)
        X = df[FEATURE_COLUMNS].values
        X_scaled = self.scaler.transform(X)

        probs = self.model.predict_proba(X_scaled)[:, 1]
        labels = self.model.predict(X_scaled)

        df["pred_label"] = labels
        df["pred_prob"] = probs

        return df
