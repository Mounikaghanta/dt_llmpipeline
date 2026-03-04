import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

pred_df = pd.read_excel("outputs/mlp_predictions.xlsx")
true_df = pd.read_excel("data/processed/test_balanced_faults.xlsx")

pred_df["Date Time"] = pd.to_datetime(pred_df["Date Time"])
true_df["Date Time"] = pd.to_datetime(true_df["Date Time"])

merged = pred_df.merge(
    true_df[["Date Time", "binary_label"]],
    on="Date Time",
    how="inner"
)

print(confusion_matrix(merged["binary_label"], merged["pred_label"]))
print(classification_report(merged["binary_label"], merged["pred_label"]))
