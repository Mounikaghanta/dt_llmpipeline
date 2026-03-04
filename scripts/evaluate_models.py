import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score


pred_df = pd.read_excel("outputs/all_model_predictions.xlsx")
true_df = pd.read_excel("data/test_balanced_faults.xlsx")

pred_df["Date Time"] = pd.to_datetime(pred_df["Date Time"])
true_df["Date Time"] = pd.to_datetime(true_df["Date Time"])


df = pred_df.merge(
    true_df[["Date Time", "binary_label"]],
    on="Date Time",
    how="inner"
)


models = [
    "MLP",
    "RandomForest",
    "LightGBM",
    "XGBoost",
    "SVM",
    "KNN",
    "LogReg"
]


print("\n===== MODEL PERFORMANCE =====\n")

for m in models:

    y_true = df["binary_label"]
    y_pred = df[f"{m}_pred"]

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)

    print(m)
    print("Accuracy :", round(acc,4))
    print("Recall   :", round(rec,4))
    print("F1 Score :", round(f1,4))
    print("-"*40)
