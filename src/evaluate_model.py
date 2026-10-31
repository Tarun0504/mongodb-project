"""
CineScope Model Evaluation Module
Computes classification metrics and generates evaluation artifacts.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORTS_DIR


def evaluate_model(model, X_test, y_test, feature_names=None) -> dict:
    """
    Evaluate a trained classifier on the held-out test set.
    
    Computes:
    - Accuracy
    - Precision (positive class)
    - Recall (positive class)
    - F1-score (positive class)
    - Confusion matrix
    - Classification report
    - Feature importance ranking
    
    Returns a dictionary of all metrics.
    """
    # ─── Generate predictions ────────────────────────────────
    y_pred = model.predict(X_test)

    # Get probabilities if supported
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    # ─── Compute metrics ─────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Macro averages
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report (as dict for storage)
    cls_report = classification_report(y_test, y_pred,
                                        target_names=["Not Liked", "Liked"],
                                        output_dict=True,
                                        zero_division=0)

    # Feature importance
    feature_importance = {}
    if feature_names and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for name, imp in sorted(zip(feature_names, importances),
                                key=lambda x: x[1], reverse=True):
            feature_importance[name] = round(float(imp), 6)

    # ─── Print results ───────────────────────────────────────
    print(f"\n  Classification Metrics (Positive class = 'Liked'):")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"")
    print(f"  Macro Averages:")
    print(f"  Precision: {prec_macro:.4f}")
    print(f"  Recall:    {rec_macro:.4f}")
    print(f"  F1-Score:  {f1_macro:.4f}")
    print(f"")
    print(f"  Confusion Matrix:")
    print(f"  {cm}")
    print(f"")
    print(f"  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Liked", "Liked"],
                                zero_division=0))

    # Top features
    if feature_importance:
        print(f"  Top 10 Feature Importances:")
        for i, (name, imp) in enumerate(list(feature_importance.items())[:10]):
            print(f"    {i+1}. {name}: {imp:.4f}")

    # ─── Save evaluation plots ───────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    _save_confusion_matrix_plot(cm, os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    _save_metrics_bar_plot(acc, prec, rec, f1, os.path.join(REPORTS_DIR, "metrics_chart.png"))
    if feature_importance:
        _save_feature_importance_plot(feature_importance,
                                      os.path.join(REPORTS_DIR, "feature_importance.png"))

    # ─── Build metrics dictionary ────────────────────────────
    metrics = {
        "model_name": "DecisionTreeClassifier",
        "dataset_version": "ml-latest-small",
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "feature_importance": feature_importance,
    }

    return metrics


def _save_confusion_matrix_plot(cm, filepath):
    """Save confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar(im)

    classes = ["Not Liked\n(rating < 4.0)", "Liked\n(rating ≥ 4.0)"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight="bold")

    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix plot saved: {filepath}")


def _save_metrics_bar_plot(acc, prec, rec, f1, filepath):
    """Save a bar chart of the four classification metrics."""
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [acc, prec, rec, f1]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    bars = ax.bar(metrics_names, values, color=colors, width=0.6, edgecolor="white")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classification Metrics", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Metrics bar chart saved: {filepath}")


def _save_feature_importance_plot(feature_importance, filepath):
    """Save a horizontal bar chart of top feature importances."""
    # Take top 15 features
    items = list(feature_importance.items())[:15]
    names = [item[0] for item in items][::-1]
    values = [item[1] for item in items][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, values, color="#2196F3", edgecolor="white")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Top Feature Importances (Decision Tree)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Feature importance plot saved: {filepath}")
