import os
import sys
import time
import random
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Mean Squared Error = (1/n) * SUMMATION (y_i - yhat_i)^2
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# Mean Absolute Error = (1/n) * SUMMATION |y_i - yhat_i|
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# Computes the coefficient of determination R^2.
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # R^2 = 1 - RSS/TSS
    # RSS = SUMMATION (y_i - yhat_i)^2,  TSS = SUMMATION (y_i - y)^2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def soft_threshold(rho: float, alpha: float) -> float:
    # Used by Lasso coordinate descent.
    # This is what pushes small coefficients exactly to 0.
    if rho < -alpha:
        return rho + alpha
    elif rho > alpha:
        return rho - alpha
    else:
        return 0.0


# Lasso regression using coordinate descent
class LassoRegression:
    def __init__(self, alpha: float = 1.0, max_iter: int = 5000, tol: float = 1e-6):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape

        # X is standardized before this method is called.
        # Center y so the intercept is just the mean of y.
        self.intercept_ = float(np.mean(y))
        y_centered = y - self.intercept_

        w = np.zeros(d, dtype=float)
        col_norms = np.sum(X ** 2, axis=0)

        for _ in range(self.max_iter):
            w_old = w.copy()

            for j in range(d):
                if col_norms[j] == 0:
                    continue

                # Partial residual excluding feature j
                residual = y_centered - (X @ w) + X[:, j] * w[j]

                rho = np.dot(X[:, j], residual)

                # Lasso update
                w[j] = soft_threshold(rho, self.alpha) / col_norms[j]

            # Stop when coefficients are barely changing
            if np.max(np.abs(w - w_old)) < self.tol:
                break

        self.coef_ = w

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Error: model not fitted.")
        return self.intercept_ + X @ self.coef_


# Load a dataset from CSV into a pandas DataFrame.
def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# Convert columns to numeric, replace bad/missing values, and remove unusable columns.
def clean_numeric_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the dataset.")

    # Convert every column to numeric when possible.
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where the target is missing.
    df = df.dropna(subset=[target_col])

    # Use every column except the target as a feature.
    feature_cols = [col for col in df.columns if col != target_col]

    # Drop feature columns that are completely empty after numeric conversion.
    feature_cols = [col for col in feature_cols if not df[col].isna().all()]

    # Fill missing feature values with each feature's median.
    for col in feature_cols:
        median_value = df[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        df[col] = df[col].fillna(median_value)

    return df, feature_cols


# Randomly splits row indices into training/validation/test partitions.
def split_dataset(
    total_rows: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_rows = np.arange(total_rows)
    rng.shuffle(all_rows)

    n_train = int(train_fraction * total_rows)
    n_val = int(val_fraction * total_rows)

    train_rows = all_rows[:n_train]
    validate_rows = all_rows[n_train:n_train + n_val]
    test_rows = all_rows[n_train + n_val:]

    return train_rows, validate_rows, test_rows


# Fit standardization parameters on X and return the standardized X.
def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # x' = (x - mean) / std
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma, (X - mu) / sigma


# Applies standardization with mu and sigma to X.
def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


# Compute ROC curve manually from binary labels and prediction scores.
def compute_roc_curve(y_binary: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]

    positives = np.sum(y_binary == 1)
    negatives = np.sum(y_binary == 0)

    fpr_values = []
    tpr_values = []

    for threshold in thresholds:
        pred_binary = (scores >= threshold).astype(int)

        tp = np.sum((pred_binary == 1) & (y_binary == 1))
        fp = np.sum((pred_binary == 1) & (y_binary == 0))

        tpr = tp / positives if positives > 0 else 0.0
        fpr = fp / negatives if negatives > 0 else 0.0

        tpr_values.append(tpr)
        fpr_values.append(fpr)

    fpr_values = np.array(fpr_values)
    tpr_values = np.array(tpr_values)

    # Sort by FPR before calculating area under the curve.
    order = np.argsort(fpr_values)
    fpr_sorted = fpr_values[order]
    tpr_sorted = tpr_values[order]
    auc = float(np.trapz(tpr_sorted, fpr_sorted))

    return fpr_sorted, tpr_sorted, thresholds, auc


# Save ROC curve image.
def save_roc_curve(y_binary: np.ndarray, scores: np.ndarray, output_path: str) -> float:
    fpr, tpr, thresholds, auc = compute_roc_curve(y_binary, scores)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"Lasso ROC curve, AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Predicting High Estimated Owners")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return auc


# Create a truth table / confusion matrix using a threshold.
def create_truth_table(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> pd.DataFrame:
    actual_binary = (y_true >= threshold).astype(int)
    predicted_binary = (y_pred >= threshold).astype(int)

    tn = int(np.sum((actual_binary == 0) & (predicted_binary == 0)))
    fp = int(np.sum((actual_binary == 0) & (predicted_binary == 1)))
    fn = int(np.sum((actual_binary == 1) & (predicted_binary == 0)))
    tp = int(np.sum((actual_binary == 1) & (predicted_binary == 1)))

    truth_table = pd.DataFrame(
        {
            "Predicted Low Owners": [tn, fn],
            "Predicted High Owners": [fp, tp],
        },
        index=["Actual Low Owners", "Actual High Owners"]
    )

    return truth_table


# Train only Lasso regression and compute metrics.
def train_and_eval(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    alpha: float,
    seed: int,
    output_dir: str
) -> Dict[str, Any]:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    # Split data
    training_rows, validation_rows, test_rows = split_dataset(len(df), 0.7, 0.1, seed)

    # Standardize features using only training data statistics
    mu, sigma, X_train = standardize_fit(X[training_rows])
    X_valid = standardize_apply(X[validation_rows], mu, sigma)
    X_test = standardize_apply(X[test_rows], mu, sigma)

    y_train = y[training_rows]
    y_valid = y[validation_rows]
    y_test = y[test_rows]

    # Train Lasso only
    lasso = LassoRegression(alpha=alpha, max_iter=5000, tol=1e-6)
    lasso.fit(X_train, y_train)

    # Predictions
    yhat_train = lasso.predict(X_train)
    yhat_valid = lasso.predict(X_valid)
    yhat_test = lasso.predict(X_test)

    # For ROC/truth table, convert the regression target into a binary label.
    # High owners = estimated_owners >= median estimated_owners in the training split.
    owner_threshold = float(np.median(y_train))

    y_test_binary = (y_test >= owner_threshold).astype(int)
    roc_auc = save_roc_curve(y_test_binary, yhat_test, os.path.join(output_dir, "roc_curve.png"))

    truth_table = create_truth_table(y_test, yhat_test, owner_threshold)
    truth_table.to_csv(os.path.join(output_dir, "truth_table.csv"))

    prediction_table = pd.DataFrame({
        "actual_estimated_owners": y_test,
        "predicted_estimated_owners": yhat_test,
        "actual_class": np.where(y_test >= owner_threshold, "High Owners", "Low Owners"),
        "predicted_class": np.where(yhat_test >= owner_threshold, "High Owners", "Low Owners"),
    })
    prediction_table.to_csv(os.path.join(output_dir, "prediction_table.csv"), index=False)

    results = {
        "model": "Lasso Regression",
        "alpha": alpha,
        "target": target_col,
        "owner_threshold_for_roc_and_truth_table": owner_threshold,
        "roc_auc": roc_auc,
        "splits": {
            "train_n": len(training_rows),
            "val_n": len(validation_rows),
            "test_n": len(test_rows)
        },
        "metrics": {
            "train": {
                "mse": mse(y_train, yhat_train),
                "mae": mae(y_train, yhat_train),
                "r2": r2_score(y_train, yhat_train)
            },
            "validate": {
                "mse": mse(y_valid, yhat_valid),
                "mae": mae(y_valid, yhat_valid),
                "r2": r2_score(y_valid, yhat_valid)
            },
            "test": {
                "mse": mse(y_test, yhat_test),
                "mae": mae(y_test, yhat_test),
                "r2": r2_score(y_test, yhat_test)
            },
        },
        "intercept": lasso.intercept_,
        "coefficients": {
            feature: float(coef)
            for feature, coef in zip(feature_cols, lasso.coef_)
        },
        "nonzero_coefficients": {
            feature: float(coef)
            for feature, coef in zip(feature_cols, lasso.coef_)
            if abs(float(coef)) > 1e-12
        },
        "feature_standardization": {
            "mean": mu.tolist(),
            "std": sigma.tolist(),
            "features": feature_cols
        }
    }

    return results


def main():
    # You can run this file either by passing the CSV path:
    #   python lasso_regression_only.py steam_games_dataset_clean.csv
    #
    # Or by setting the default path below.
    if len(sys.argv) >= 2:
        data_file = sys.argv[1]
    else:
        data_file = r"C:\Users\gregc\OneDrive\Desktop\git\Data_Mining_Steam_Project\data\steam_games_dataset_clean.csv"

    seed = 3245
    # seed = int(time.time()) ^ random.getrandbits(16)  # uncomment for random seed

    alpha = 1.0  # Lasso regularization strength
    target_col = "estimated_owners"

    output_dir = "./lasso_regression_outputs"
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_file)
    df, feature_cols = clean_numeric_data(df, target_col)

    results = train_and_eval(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        alpha=alpha,
        seed=seed,
        output_dir=output_dir
    )

    # Output results to json
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"Model: {results['model']}")
    print(f"Seed: {seed}")
    print(f"Target label: {target_col}")
    print(f"Alpha: {alpha}")
    print(f"Split sizes: train={results['splits']['train_n']}, valid={results['splits']['val_n']}, test={results['splits']['test_n']}")
    print(f"ROC/truth-table threshold: estimated_owners >= {results['owner_threshold_for_roc_and_truth_table']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"Number of features used: {len(feature_cols)}")
    print(f"Number of nonzero Lasso coefficients: {len(results['nonzero_coefficients'])}")

    print("\nLasso Regression:")
    print(f"Intercept: {results['intercept']:.6f}")

    for split in ["train", "validate", "test"]:
        m = results["metrics"][split]
        print(f"  {split.upper()}: MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    print("\nFiles created:")
    print("  metrics.json")
    print("  roc_curve.png")
    print("  truth_table.csv")
    print("  prediction_table.csv")


if __name__ == "__main__":
    main()
