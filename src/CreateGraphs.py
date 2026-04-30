import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_dir = "./linear_regression_outputs"

prediction_path = os.path.join(output_dir, "prediction_table.csv")
metrics_path = os.path.join(output_dir, "metrics.json")

predictions = pd.read_csv(prediction_path)

y_test = predictions["actual_estimated_owners"].to_numpy()
yhat_test = predictions["predicted_estimated_owners"].to_numpy()
residuals = y_test - yhat_test

# 1. Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, yhat_test, alpha=0.6)

min_val = min(np.min(y_test), np.min(yhat_test))
max_val = max(np.max(y_test), np.max(yhat_test))
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Actual Estimated Owners")
plt.ylabel("Predicted Estimated Owners")
plt.title("Actual vs Predicted Estimated Owners")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=300)
plt.close()

# 2. Residual Plot
plt.figure(figsize=(7, 5))
plt.scatter(yhat_test, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Estimated Owners")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residual_plot.png"), dpi=300)
plt.close()

# 3. Error Histogram
plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=30)

plt.xlabel("Prediction Error: Actual - Predicted")
plt.ylabel("Number of Games")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "error_histogram.png"), dpi=300)
plt.close()

# 4. Coefficient Plot
with open(metrics_path, "r") as f:
    metrics = json.load(f)

# Works for both Lasso and Linear Regression
coefficients = metrics.get("nonzero_coefficients", metrics.get("coefficients", {}))

coef_series = pd.Series(coefficients)

if len(coef_series) > 0:
    coef_series = coef_series.reindex(
        coef_series.abs().sort_values(ascending=False).head(20).index
    )

    plt.figure(figsize=(10, 6))
    coef_series.sort_values().plot(kind="barh")

    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.title("Top 20 Regression Coefficients by Absolute Value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_coefficients.png"), dpi=300)
    plt.close()
else:
    print("No coefficients found in metrics.json")

print("Created:")
print(os.path.join(output_dir, "actual_vs_predicted.png"))
print(os.path.join(output_dir, "residual_plot.png"))
print(os.path.join(output_dir, "error_histogram.png"))
print(os.path.join(output_dir, "lasso_coefficients.png"))