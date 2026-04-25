import os
import sys
import time
import random
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd


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
    if rho < -alpha:
        return rho + alpha
    elif rho > alpha:
        return rho - alpha
    else:
        return 0.0


# Ordinary Least Squares Regression
class LinearRegression:
    def __init__(self):
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    # Fit the model to data using the closed-form solution.
    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape

        Xb = np.hstack([np.ones((n, 1)), X])  # bias column
        XtX = Xb.T @ Xb
        Xty = Xb.T @ y

        # OLS: (XtX)^-1 Xty
        w = np.linalg.pinv(XtX) @ Xty

        self.intercept_ = float(w[0])
        self.coef_ = w[1:].astype(float)

    # Predict targets for the given feature matrix.
    def predict(self, X: np.ndarray) -> np.ndarray:
        # yhat = b + X @ w
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Error: model not fitted.")
        return self.intercept_ + X @ self.coef_


# Lasso regression using coordinate descent
class LassoRegression:
    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-6):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape

        # Since X is already standardized in your pipeline, we only center y
        self.intercept_ = float(np.mean(y))
        y_centered = y - self.intercept_

        w = np.zeros(d, dtype=float)

        # Precompute squared norms of columns
        col_norms = np.sum(X ** 2, axis=0)

        for _ in range(self.max_iter):
            w_old = w.copy()

            for j in range(d):
                if col_norms[j] == 0:
                    continue

                # Partial residual excluding feature j
                residual = y_centered - (X @ w) + X[:, j] * w[j]

                rho = np.dot(X[:, j], residual)

                # Coordinate descent update with soft-thresholding
                w[j] = soft_threshold(rho, self.alpha) / col_norms[j]

            # Convergence check
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


# Train OLS and Lasso models for each target column and compute metrics.
def train_and_eval(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    alpha: float,
    seed: int
) -> Dict[str, Any]:
    X = df[feature_cols].to_numpy(dtype=float)
    Y = df[target_cols].to_numpy(dtype=float)

    # Split data
    training_rows, validation_rows, test_rows = split_dataset(len(df), 0.7, 0.1, seed)

    # Standardize data
    mu, sigma, X_train = standardize_fit(X[training_rows])
    X_valid = standardize_apply(X[validation_rows], mu, sigma)
    X_test = standardize_apply(X[test_rows], mu, sigma)

    results = {
        "splits": {"train_n": len(training_rows), "val_n": len(validation_rows), "test_n": len(test_rows)},
        "targets": {}
    }

    for j, tgt in enumerate(target_cols):
        y_train = Y[training_rows, j]
        y_valid = Y[validation_rows, j]
        y_test = Y[test_rows, j]

        ols = LinearRegression()
        ols.fit(X_train, y_train)

        lasso = LassoRegression(alpha=alpha, max_iter=5000, tol=1e-6)
        lasso.fit(X_train, y_train)

        # Predictions
        yhat_train_ols = ols.predict(X_train)
        yhat_valid_ols = ols.predict(X_valid)
        yhat_test_ols = ols.predict(X_test)

        yhat_train_lasso = lasso.predict(X_train)
        yhat_valid_lasso = lasso.predict(X_valid)
        yhat_test_lasso = lasso.predict(X_test)

        # Metrics
        results["targets"][tgt] = {
            "ols": {
                "intercept": ols.intercept_,
                "coef": ols.coef_.tolist(),
                "metrics": {
                    "train": {"mse": mse(y_train, yhat_train_ols), "mae": mae(y_train, yhat_train_ols), "r2": r2_score(y_train, yhat_train_ols)},
                    "validate": {"mse": mse(y_valid, yhat_valid_ols), "mae": mae(y_valid, yhat_valid_ols), "r2": r2_score(y_valid, yhat_valid_ols)},
                    "test": {"mse": mse(y_test, yhat_test_ols), "mae": mae(y_test, yhat_test_ols), "r2": r2_score(y_test, yhat_test_ols)},
                }
            },
            "lasso": {
                "alpha": alpha,
                "intercept": lasso.intercept_,
                "coef": lasso.coef_.tolist(),
                "metrics": {
                    "train": {"mse": mse(y_train, yhat_train_lasso), "mae": mae(y_train, yhat_train_lasso), "r2": r2_score(y_train, yhat_train_lasso)},
                    "validate": {"mse": mse(y_valid, yhat_valid_lasso), "mae": mae(y_valid, yhat_valid_lasso), "r2": r2_score(y_valid, yhat_valid_lasso)},
                    "test": {"mse": mse(y_test, yhat_test_lasso), "mae": mae(y_test, yhat_test_lasso), "r2": r2_score(y_test, yhat_test_lasso)},
                }
            }
        }

    results["feature_standardization"] = {
        "mean": mu.tolist(),
        "std": sigma.tolist(),
        "features": feature_cols
    }
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python lasso_from_scratch.py <datafile>")
        sys.exit(1)

    data_file = sys.argv[1]

    seed = 3245
    # seed = int(time.time()) ^ random.getrandbits(16)  # uncomment for random seed
    alpha = 1.0  # Lasso regularization strength

    feature_cols = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
    target_cols = ["Y1", "Y2"]

    df = load_data(data_file)

    results = train_and_eval(df, feature_cols, target_cols, alpha=alpha, seed=seed)

    # Output results to json
    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # print summary
    print(f"Seed: {seed}")
    print(f"Split sizes: train={results['splits']['train_n']}, valid={results['splits']['val_n']}, test={results['splits']['test_n']}")
    print(f"Features standardized using TRAIN mean/std: {results['feature_standardization']['features']}")

    for tgt, detail in results["targets"].items():
        print(f"=== Target: {tgt} ===")
        for model_name in ["ols", "lasso"]:
            md = detail[model_name]
            header = "Ordinary Least Squares" if model_name == "ols" else f"Lasso (alpha={md.get('alpha', 0.0)})"
            print(header + ":")
            print(f"Intercept: {md['intercept']:.6f}")
            print(f"Coefficients: {np.array(md['coef'])}")
            for split in ["train", "validate", "test"]:
                m = md["metrics"][split]
                print(f"  {split.upper()}: MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")


if __name__ == "__main__":
    main()