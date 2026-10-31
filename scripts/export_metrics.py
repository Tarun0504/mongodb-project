"""
CineScope Metrics Export Script
Exports model evaluation metrics from MongoDB to JSON and CSV formats.
Usage: python scripts/export_metrics.py
"""
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORTS_DIR
from database import CineScopeDB


def export_metrics():
    """Export stored metrics from MongoDB to reports directory."""
    print("=" * 60)
    print("CineScope — Export Model Metrics")
    print("=" * 60)

    db = CineScopeDB()
    metrics = db.get_latest_metrics()

    if not metrics:
        print("✗ No metrics found in MongoDB. Run train_and_evaluate.py first.")
        db.close()
        return

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Export to JSON
    json_path = os.path.join(REPORTS_DIR, "exported_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics exported to: {json_path}")

    # Export summary table to CSV
    summary_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score",
                   "Macro Precision", "Macro Recall", "Macro F1-Score",
                   "Train Size", "Test Size", "Total Size", "Evaluation Date"],
        "Value": [
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('recall', 0):.4f}",
            f"{metrics.get('f1_score', 0):.4f}",
            f"{metrics.get('precision_macro', 0):.4f}",
            f"{metrics.get('recall_macro', 0):.4f}",
            f"{metrics.get('f1_macro', 0):.4f}",
            f"{metrics.get('train_size', 0):,}",
            f"{metrics.get('test_size', 0):,}",
            f"{metrics.get('total_size', 0):,}",
            str(metrics.get("evaluation_date", "N/A"))
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    csv_path = os.path.join(REPORTS_DIR, "metrics_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"✓ Summary table exported to: {csv_path}")

    print("\nMetrics Summary:")
    print(df_summary.to_string(index=False))

    db.close()


if __name__ == "__main__":
    export_metrics()
