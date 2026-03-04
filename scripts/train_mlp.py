import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from dt_llm.classifier.features import build_features, FEATURE_COLUMNS

# Load labeled training data
sensor_df = pd.read_excel("data/processed/8months_balanced_faults.xlsx")
pred_df   = pd.read_excel("data/processed/8monthspredictions_only.xlsx")

df = build_features(sensor_df, pred_df)

X = df[FEATURE_COLUMNS].values
y = df["binary_label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
mlp.fit(X_scaled, y)

joblib.dump(mlp, "models/mlp_model.pkl")
joblib.dump(scaler, "models/mlp_scaler.pkl")

print("MLP trained and saved.")
