import os
import sys
import json
import sqlite3
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Regression metrics
# -----------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# -----------------------------
# Linear Regression from scratch
# -----------------------------

class LinearRegression:
    def __init__(self):
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]

        # Add a bias/intercept column
        Xb = np.hstack([np.ones((n, 1)), X])

        # Closed-form OLS solution:
        # w = (X^T X)^-1 X^T y
        w = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ y)

        self.intercept_ = float(w[0])
        self.coef_ = w[1:].astype(float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Error: model not fitted.")
        return self.intercept_ + X @ self.coef_


# -----------------------------
# Data loading
# -----------------------------

def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if ext == ".db":
        # Reads the first table from a SQLite database file
        conn = sqlite3.connect(path)
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table';",
            conn
        )["name"].tolist()

        if not tables:
            conn.close()
            raise ValueError("No tables found in the database file.")

        df = pd.read_sql_query(f"SELECT * FROM {tables[0]}", conn)
        conn.close()
        return df

    raise ValueError(f"Unsupported file extension: {ext}")


# -----------------------------
# Splitting and preprocessing
# -----------------------------

def split_dataset(
    total_rows: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.10,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = np.arange(total_rows)
    rng.shuffle(rows)

    n_train = int(train_fraction * total_rows)
    n_val = int(val_fraction * total_rows)

    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]

    return train_rows, val_rows, test_rows


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)

    # Avoid division by zero for constant columns
    std[std == 0] = 1.0

    X_scaled = (X - mean) / std
    return mean, std, X_scaled


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def clean_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the data.")

    # Convert every column to numeric when possible.
    # Non-numeric values become NaN and are filled below.
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where the target is missing
    df = df.dropna(subset=[target_col])

    # Fill missing feature values with 0
    feature_cols = [col for col in df.columns if col != target_col]
    df[feature_cols] = df[feature_cols].fillna(0)

    return df


# -----------------------------
# ROC curve and truth table
# -----------------------------

def make_binary_labels(y_true: np.ndarray, threshold: float) -> np.ndarray:
    # 1 means the game has at least the threshold number of estimated owners
    # 0 means below the threshold
    return (y_true >= threshold).astype(int)


def roc_points(y_binary: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    thresholds = np.r_[np.inf, np.sort(np.unique(y_scores))[::-1], -np.inf]

    fpr_list = []
    tpr_list = []

    positives = np.sum(y_binary == 1)
    negatives = np.sum(y_binary == 0)

    for threshold in thresholds:
        y_pred_binary = (y_scores >= threshold).astype(int)

        tp = np.sum((y_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_binary == 0) & (y_pred_binary == 1))
        tn = np.sum((y_binary == 0) & (y_pred_binary == 0))
        fn = np.sum((y_binary == 1) & (y_pred_binary == 0))

        tpr = tp / positives if positives > 0 else 0.0
        fpr = fp / negatives if negatives > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)

    # Sort by FPR so trapezoid area is correct
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def save_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, output_path: str):
    plt.figure()
    plt.plot(fpr, tpr, label=f"Linear Regression score, AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for estimated_owners")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def confusion_truth_table(y_binary: np.ndarray, y_scores: np.ndarray, threshold: float) -> pd.DataFrame:
    y_pred_binary = (y_scores >= threshold).astype(int)

    tp = int(np.sum((y_binary == 1) & (y_pred_binary == 1)))
    fp = int(np.sum((y_binary == 0) & (y_pred_binary == 1)))
    tn = int(np.sum((y_binary == 0) & (y_pred_binary == 0)))
    fn = int(np.sum((y_binary == 1) & (y_pred_binary == 0)))

    table = pd.DataFrame(
        {
            "Predicted Below Threshold": [tn, fn],
            "Predicted At/Above Threshold": [fp, tp],
        },
        index=["Actual Below Threshold", "Actual At/Above Threshold"]
    )

    return table


# -----------------------------
# Training and evaluation
# -----------------------------

def train_and_eval(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seed: int,
    output_dir: str
) -> Dict[str, Any]:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    train_rows, val_rows, test_rows = split_dataset(len(df), 0.70, 0.10, seed)

    mean, std, X_train = standardize_fit(X[train_rows])
    X_val = standardize_apply(X[val_rows], mean, std)
    X_test = standardize_apply(X[test_rows], mean, std)

    y_train = y[train_rows]
    y_val = y[val_rows]
    y_test = y[test_rows]

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    # For ROC, convert the regression problem into a binary question:
    # Is estimated_owners at or above the median estimated_owners value from training?
    owner_threshold = float(np.median(y_train))
    y_test_binary = make_binary_labels(y_test, owner_threshold)

    fpr, tpr, auc = roc_points(y_test_binary, pred_test)
    roc_path = os.path.join(output_dir, "roc_curve.png")
    save_roc_curve(fpr, tpr, auc, roc_path)

    truth_table = confusion_truth_table(y_test_binary, pred_test, owner_threshold)
    truth_table_path = os.path.join(output_dir, "truth_table.csv")
    truth_table.to_csv(truth_table_path)

    prediction_table = pd.DataFrame({
        "actual_estimated_owners": y_test,
        "predicted_estimated_owners": pred_test,
        "actual_binary_at_or_above_threshold": y_test_binary,
        "predicted_binary_at_or_above_threshold": (pred_test >= owner_threshold).astype(int)
    })
    prediction_table_path = os.path.join(output_dir, "prediction_table.csv")
    prediction_table.to_csv(prediction_table_path, index=False)

    results = {
        "target": target_col,
        "owner_threshold_for_roc_and_truth_table": owner_threshold,
        "splits": {
            "train_n": len(train_rows),
            "val_n": len(val_rows),
            "test_n": len(test_rows)
        },
        "linear_regression": {
            "intercept": model.intercept_,
            "coef": model.coef_.tolist(),
            "metrics": {
                "train": {
                    "mse": mse(y_train, pred_train),
                    "mae": mae(y_train, pred_train),
                    "r2": r2_score(y_train, pred_train)
                },
                "validate": {
                    "mse": mse(y_val, pred_val),
                    "mae": mae(y_val, pred_val),
                    "r2": r2_score(y_val, pred_val)
                },
                "test": {
                    "mse": mse(y_test, pred_test),
                    "mae": mae(y_test, pred_test),
                    "r2": r2_score(y_test, pred_test)
                }
            },
            "roc_auc_on_test": auc
        },
        "output_files": {
            "roc_curve": roc_path,
            "truth_table": truth_table_path,
            "prediction_table": prediction_table_path
        },
        "feature_standardization": {
            "features": feature_cols,
            "mean": mean.tolist(),
            "std": std.tolist()
        }
    }

    return results


def main():
    # Change this path to your CSV or DB file.
    # Example CSV:
    data_file = r"C:\Users\gregc\OneDrive\Desktop\git\Data_Mining_Steam_Project\data\steam_games_dataset_clean.csv"

    # If you want to pass the file path from the terminal instead, run:
    # python Linear_Regression_Only.py path_to_your_file.csv
    if len(sys.argv) >= 2:
        data_file = sys.argv[1]

    seed = 3245
    target_col = "estimated_owners"
    output_dir = "linear_regression_outputs"
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_file)
    df = clean_dataframe(df, target_col)

    # Use all columns except the label as features
    feature_cols = [col for col in df.columns if col != target_col]

    results = train_and_eval(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        seed=seed,
        output_dir=output_dir
    )

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Seed: {seed}")
    print(f"Target label: {target_col}")
    print(f"Number of features used: {len(feature_cols)}")
    print(
        f"Split sizes: train={results['splits']['train_n']}, "
        f"valid={results['splits']['val_n']}, "
        f"test={results['splits']['test_n']}"
    )

    print("\n=== Linear Regression Metrics ===")
    for split, values in results["linear_regression"]["metrics"].items():
        print(
            f"{split.upper()}: "
            f"MSE={values['mse']:.4f}  "
            f"MAE={values['mae']:.4f}  "
            f"R2={values['r2']:.4f}"
        )

    print("\n=== ROC / Truth Table Info ===")
    print(
        "ROC is created by treating games with estimated_owners >= "
        f"{results['owner_threshold_for_roc_and_truth_table']:.4f} as class 1."
    )
    print(f"Test ROC AUC: {results['linear_regression']['roc_auc_on_test']:.4f}")

    print("\nSaved files:")
    print(f"- {metrics_path}")
    print(f"- {results['output_files']['roc_curve']}")
    print(f"- {results['output_files']['truth_table']}")
    print(f"- {results['output_files']['prediction_table']}")


if __name__ == "__main__":
    main()
