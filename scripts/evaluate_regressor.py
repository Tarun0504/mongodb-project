"""
CineScope Regression Experiment Script
Evaluates a DecisionTreeRegressor predicting continuous ratings (0.5 - 5.0)
using the exact same feature set and train/test split.
Computes R² and MAE as requested by academic assignment rubric.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, REPORTS_DIR
from src.dataset_loader import DatasetLoader
from src.preprocessing import DataPreprocessor


def evaluate_regression():
    print("=" * 60)
    print("CineScope — Secondary Regression Experiment (Continuous Ratings)")
    print("=" * 60)

    loader = DatasetLoader()
    data = loader.load_all()

    preprocessor = DataPreprocessor(random_seed=RANDOM_SEED)
    movies_clean, ratings_clean = preprocessor.clean_data(data["movies"], data["ratings"])
    ratings_with_target = preprocessor.create_target(ratings_clean)

    train_df, test_df = preprocessor.split_data(ratings_with_target)

    X_train = preprocessor.engineer_features(train_df, movies_clean, is_training=True)
    y_train_cont = train_df["rating"].values

    X_test = preprocessor.engineer_features(test_df, movies_clean, is_training=False)
    y_test_cont = test_df["rating"].values

    regressor = DecisionTreeRegressor(
        max_depth=10,
        min_samples_leaf=50,
        random_state=RANDOM_SEED
    )
    regressor.fit(X_train, y_train_cont)
    y_pred_cont = regressor.predict(X_test)

    mae = mean_absolute_error(y_test_cont, y_pred_cont)
    r2 = r2_score(y_test_cont, y_pred_cont)
    rmse = np.sqrt(mean_squared_error(y_test_cont, y_pred_cont))

    print(f"\nRegression Evaluation Results (Held-out Test Set):")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R-Squared (R²): {r2:.4f}")

    results = {
        "model": "DecisionTreeRegressor",
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2), 4),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

    out_path = os.path.join(REPORTS_DIR, "regression_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved regression metrics to {out_path}")
    return results


if __name__ == "__main__":
    evaluate_regression()
