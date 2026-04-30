import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from clean_data import OWNER_COUNT_RANGES
from data_io import read_data


def get_class_name(label):
    """
    Converts numeric class labels into readable owner count range names.
    Example: 0 -> OWNER_COUNT_RANGES[0]
    """
    try:
        label_int = int(label)
        if 0 <= label_int < len(OWNER_COUNT_RANGES):
            return OWNER_COUNT_RANGES[label_int]
    except Exception:
        pass

    return str(label)


def save_confusion_matrix_plot(y_true, y_pred, labels, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {name}" for name in class_names],
        columns=[f"Predicted {name}" for name in class_names],
    )

    cm_df.to_csv(os.path.join(output_dir, "truth_table.csv"))

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, values_format="d")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    return cm_df


def save_feature_importance_plot(model, feature_cols, output_dir):
    importances = model.feature_importances_
    importance_series = pd.Series(importances, index=feature_cols)

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


def save_class_distribution_plot(y_train, y_test, class_names, labels, output_dir):
    train_counts = pd.Series(y_train).value_counts().reindex(labels, fill_value=0)
    test_counts = pd.Series(y_test).value_counts().reindex(labels, fill_value=0)

    distribution_df = pd.DataFrame({
        "train_count": train_counts.values,
        "test_count": test_counts.values,
    }, index=class_names)

    distribution_df.to_csv(os.path.join(output_dir, "class_distribution.csv"))

    distribution_df.plot(kind="bar", figsize=(10, 6))

    plt.xlabel("Owner Count Class")
    plt.ylabel("Number of Games")
    plt.title("Class Distribution in Train and Test Sets")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
    plt.close()


def save_per_class_f1_plot(report_dict, output_dir):
    rows = []

    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict) and "f1-score" in metrics:
            rows.append({
                "class": class_name,
                "f1_score": metrics["f1-score"],
            })

    f1_df = pd.DataFrame(rows)

    if len(f1_df) == 0:
        return

    f1_df.to_csv(os.path.join(output_dir, "per_class_f1_scores.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.bar(f1_df["class"], f1_df["f1_score"])

    plt.xlabel("Owner Count Class")
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Scores")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_f1_scores.png"), dpi=300)
    plt.close()


def save_multiclass_roc_curve(model, X_test, y_test, labels, class_names, output_dir):
    """
    Creates a one-vs-rest ROC curve for multiclass classification.
    This works because RandomForestClassifier has predict_proba().
    """
    if not hasattr(model, "predict_proba"):
        print("ROC curve skipped because model does not support predict_proba().")
        return None

    y_score = model.predict_proba(X_test)

    # Binarize true labels for one-vs-rest ROC
    y_test_bin = label_binarize(y_test, classes=labels)

    # If there are only two classes, label_binarize returns one column.
    # Convert it into two columns for consistency.
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

    roc_results = {}

    plt.figure(figsize=(8, 6))

    for i, class_name in enumerate(class_names):
        if i >= y_score.shape[1]:
            continue

        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        roc_results[class_name] = float(roc_auc)

        plt.plot(fpr, tpr, label=f"{class_name}, AUC = {roc_auc:.4f}")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest Multiclass ROC Curve")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close()

    with open(os.path.join(output_dir, "roc_auc_scores.json"), "w") as f:
        json.dump(roc_results, f, indent=2)

    return roc_results


if __name__ == "__main__":
    output_dir = "./random_forest_classifier_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load cleaned training and testing data.
    df_train = read_data("steam_games_dataset_clean_training.db")
    df_test = read_data("steam_games_dataset_clean_testing.db")

    print("Data loaded")

    target_col = "estimated_owners"

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # Make sure test columns match training columns.
    X_test = X_test[X_train.columns]

    # Train model.
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=3245,
        class_weight="balanced"
    )

    rf.fit(X_train, y_train)

    # Predictions.
    y_pred = rf.predict(X_test)

    labels = list(rf.classes_)
    class_names = [get_class_name(label) for label in labels]

    # Classification report.
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )

    print(report_text)

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))

    # Prediction table.
    prediction_table = pd.DataFrame({
        "actual_class": y_test,
        "predicted_class": y_pred,
        "actual_class_name": [get_class_name(label) for label in y_test],
        "predicted_class_name": [get_class_name(label) for label in y_pred],
    })

    prediction_table.to_csv(
        os.path.join(output_dir, "prediction_table.csv"),
        index=False
    )

    # Confusion matrix / truth table.
    save_confusion_matrix_plot(
        y_true=y_test,
        y_pred=y_pred,
        labels=labels,
        class_names=class_names,
        output_dir=output_dir
    )

    # Feature importance graph.
    save_feature_importance_plot(
        model=rf,
        feature_cols=X_train.columns,
        output_dir=output_dir
    )

    # Class distribution graph.
    save_class_distribution_plot(
        y_train=y_train,
        y_test=y_test,
        class_names=class_names,
        labels=labels,
        output_dir=output_dir
    )

    # Per-class F1 score graph.
    save_per_class_f1_plot(
        report_dict=report_dict,
        output_dir=output_dir
    )

    # Multiclass ROC curve.
    roc_auc_scores = save_multiclass_roc_curve(
        model=rf,
        X_test=X_test,
        y_test=y_test,
        labels=labels,
        class_names=class_names,
        output_dir=output_dir
    )

    # Save full metrics.
    metrics = {
        "model": "Random Forest Classifier",
        "target": target_col,
        "hyperparameters": {
            "n_estimators": 100,
            "random_state": 3245,
            "class_weight": "balanced"
        },
        "train_n": int(len(df_train)),
        "test_n": int(len(df_test)),
        "classes": {
            str(label): get_class_name(label)
            for label in labels
        },
        "classification_report": report_dict,
        "roc_auc_scores": roc_auc_scores,
        "feature_importances": {
            feature: float(importance)
            for feature, importance in zip(X_train.columns, rf.feature_importances_)
        }
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nFiles created:")
    print(os.path.join(output_dir, "metrics.json"))
    print(os.path.join(output_dir, "classification_report.csv"))
    print(os.path.join(output_dir, "prediction_table.csv"))
    print(os.path.join(output_dir, "truth_table.csv"))
    print(os.path.join(output_dir, "confusion_matrix.png"))
    print(os.path.join(output_dir, "feature_importances.png"))
    print(os.path.join(output_dir, "class_distribution.csv"))
    print(os.path.join(output_dir, "class_distribution.png"))
    print(os.path.join(output_dir, "per_class_f1_scores.csv"))
    print(os.path.join(output_dir, "per_class_f1_scores.png"))
    print(os.path.join(output_dir, "roc_curve.png"))
    print(os.path.join(output_dir, "roc_auc_scores.json"))

    print("\ndone!")