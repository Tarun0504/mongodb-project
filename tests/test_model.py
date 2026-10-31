"""
Unit Tests for Machine Learning Training and Evaluation Pipeline
"""
import os
import pytest
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from config import MODELS_DIR, REPORTS_DIR
from src.preprocessing import DataPreprocessor
from src.evaluate_model import evaluate_model
from src.train_model import train_model


@pytest.fixture
def dummy_dataset():
    """Generates synthetic dataset for quick testing."""
    np_rand = pd.Series(range(200))
    movies_df = pd.DataFrame({
        "movieId": list(range(1, 21)),
        "title": [f"Movie {i} (2000)" for i in range(1, 21)],
        "genres": ["Action|Thriller" if i % 2 == 0 else "Comedy|Romance" for i in range(1, 21)]
    })

    ratings = []
    for uid in range(1, 21):
        for mid in range(1, 21):
            if (uid + mid) % 3 != 0:
                # generate pseudo ratings
                r = 4.5 if (uid + mid) % 2 == 0 else 2.5
                ratings.append({"userId": uid, "movieId": mid, "rating": r, "timestamp": 1000 + uid*10 + mid})

    ratings_df = pd.DataFrame(ratings)
    return movies_df, ratings_df


def test_model_training_and_evaluation_flow(dummy_dataset):
    """Test full training pipeline with synthetic data."""
    movies_df, ratings_df = dummy_dataset

    results = train_model(movies_df, ratings_df, save_artifacts=False)
    model = results["model"]
    metrics = results["metrics"]

    assert isinstance(model, DecisionTreeClassifier)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_saved_artifacts_exist():
    """Verify that official model artifacts and evaluation plots exist on disk."""
    model_path = os.path.join(MODELS_DIR, "decision_tree_model.joblib")
    preproc_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    metrics_path = os.path.join(REPORTS_DIR, "model_metrics.json")
    cm_plot_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    metrics_plot_path = os.path.join(REPORTS_DIR, "metrics_chart.png")

    assert os.path.exists(model_path), "Model joblib must exist"
    assert os.path.exists(preproc_path), "Preprocessor joblib must exist"
    assert os.path.exists(metrics_path), "Metrics JSON must exist"
    assert os.path.exists(cm_plot_path), "Confusion matrix plot must exist"
    assert os.path.exists(metrics_plot_path), "Metrics plot must exist"
