import os
import sys
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def clean_numeric_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the dataset.")

    df = df.copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[target_col])

    feature_cols = [col for col in df.columns if col != target_col]
    feature_cols = [col for col in feature_cols if not df[col].isna().all()]

    for col in feature_cols:
        median_value = df[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        df[col] = df[col].fillna(median_value)

    return df, feature_cols


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

    order = np.argsort(fpr_values)
    fpr_sorted = fpr_values[order]
    tpr_sorted = tpr_values[order]

    auc = float(np.trapz(tpr_sorted, fpr_sorted))

    return fpr_sorted, tpr_sorted, thresholds, auc


def save_roc_curve(y_binary: np.ndarray, scores: np.ndarray, output_path: str) -> float:
    fpr, tpr, thresholds, auc = compute_roc_curve(y_binary, scores)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"Random Forest ROC curve, AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Predicting High Estimated Owners")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return auc


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


def save_regression_plots(y_test: np.ndarray, yhat_test: np.ndarray, output_dir: str):
    residuals = y_test - yhat_test

    # Actual vs Predicted
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

    # Residual Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(yhat_test, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--")

    plt.xlabel("Predicted Estimated Owners")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_plot.png"), dpi=300)
    plt.close()

    # Error Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=30)

    plt.xlabel("Prediction Error: Actual - Predicted")
    plt.ylabel("Number of Games")
    plt.title("Distribution of Prediction Errors")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_histogram.png"), dpi=300)
    plt.close()


# Save Random Forest feature importance plot.
def save_feature_importance_plot(
    feature_cols: List[str],
    feature_importances: np.ndarray,
    output_dir: str
):
    importance_series = pd.Series(feature_importances, index=feature_cols)

    importance_series = importance_series.reindex(
        importance_series.abs().sort_values(ascending=False).head(20).index
    )

    plt.figure(figsize=(10, 6))
    importance_series.sort_values().plot(kind="barh")

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importances.png"), dpi=300)
    plt.close()


# Train Random Forest regression and compute metrics.
def train_and_eval(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seed: int,
    output_dir: str,
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
) -> Dict[str, Any]:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    # Split data
    training_rows, validation_rows, test_rows = split_dataset(len(df), 0.7, 0.1, seed)

    X_train = X[training_rows]
    X_valid = X[validation_rows]
    X_test = X[test_rows]

    y_train = y[training_rows]
    y_valid = y[validation_rows]
    y_test = y[test_rows]

    rfr = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )

    rfr.fit(X_train, y_train)

    yhat_train = rfr.predict(X_train)
    yhat_valid = rfr.predict(X_valid)
    yhat_test = rfr.predict(X_test)

    owner_threshold = float(np.median(y_train))

    y_test_binary = (y_test >= owner_threshold).astype(int)
    roc_auc = save_roc_curve(
        y_test_binary,
        yhat_test,
        os.path.join(output_dir, "roc_curve.png")
    )

    truth_table = create_truth_table(y_test, yhat_test, owner_threshold)
    truth_table.to_csv(os.path.join(output_dir, "truth_table.csv"))

    prediction_table = pd.DataFrame({
        "actual_estimated_owners": y_test,
        "predicted_estimated_owners": yhat_test,
        "actual_class": np.where(y_test >= owner_threshold, "High Owners", "Low Owners"),
        "predicted_class": np.where(yhat_test >= owner_threshold, "High Owners", "Low Owners"),
    })

    prediction_table.to_csv(os.path.join(output_dir, "prediction_table.csv"), index=False)

    save_regression_plots(
        y_test=y_test,
        yhat_test=yhat_test,
        output_dir=output_dir
    )

    save_feature_importance_plot(
        feature_cols=feature_cols,
        feature_importances=rfr.feature_importances_,
        output_dir=output_dir
    )

    results = {
        "model": "Random Forest Regression",
        "target": target_col,
        "owner_threshold_for_roc_and_truth_table": owner_threshold,
        "roc_auc": roc_auc,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": seed,
        },
        "splits": {
            "train_n": len(training_rows),
            "val_n": len(validation_rows),
            "test_n": len(test_rows)
        },
        "metrics": {
            "train": {
                "mse": mse(y_train, yhat_train),
                "mae": mae(y_train, yhat_train),
                "r2": r2_score(y_train, yhat_train),
            },
            "validate": {
                "mse": mse(y_valid, yhat_valid),
                "mae": mae(y_valid, yhat_valid),
                "r2": r2_score(y_valid, yhat_valid),
            },
            "test": {
                "mse": mse(y_test, yhat_test),
                "mae": mae(y_test, yhat_test),
                "r2": r2_score(y_test, yhat_test),
            },
        },
        "feature_importances": {
            feature: float(importance)
            for feature, importance in zip(feature_cols, rfr.feature_importances_)
        },
    }

    return results


def main():
    if len(sys.argv) >= 2:
        data_file = sys.argv[1]
    else:
        data_file = r"C:\Users\gregc\OneDrive\Desktop\Data_Mining_Steam_Project\data\steam_games_dataset_clean.csv"

    seed = 3245
    target_col = "estimated_owners"

    output_dir = "./random_forest_regression_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Hyperparameters
    n_estimators = 100
    max_depth = None
    min_samples_split = 2
    min_samples_leaf = 1

    df = load_data(data_file)
    df, feature_cols = clean_numeric_data(df, target_col)

    results = train_and_eval(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        seed=seed,
        output_dir=output_dir,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"Model: {results['model']}")
    print(f"Seed: {seed}")
    print(f"Target label: {target_col}")
    print(f"Split sizes: train={results['splits']['train_n']}, valid={results['splits']['val_n']}, test={results['splits']['test_n']}")
    print(f"ROC/truth-table threshold: estimated_owners >= {results['owner_threshold_for_roc_and_truth_table']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"Number of features used: {len(feature_cols)}")
    print(f"Hyperparameters: {results['hyperparameters']}")

    print("\nRandom Forest Regression:")
    for split in ["train", "validate", "test"]:
        m = results["metrics"][split]
        print(f"  {split.upper()}: MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    print("\nFiles created:")
    print(os.path.join(output_dir, "metrics.json"))
    print(os.path.join(output_dir, "roc_curve.png"))
    print(os.path.join(output_dir, "truth_table.csv"))
    print(os.path.join(output_dir, "prediction_table.csv"))
    print(os.path.join(output_dir, "actual_vs_predicted.png"))
    print(os.path.join(output_dir, "residual_plot.png"))
    print(os.path.join(output_dir, "error_histogram.png"))
    print(os.path.join(output_dir, "feature_importances.png"))


if __name__ == "__main__":
    main()