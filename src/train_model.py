"""
CineScope Model Training Module
Trains a DecisionTreeClassifier to predict whether a user will like a movie.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, MODELS_DIR, REPORTS_DIR
from src.preprocessing import DataPreprocessor
from src.evaluate_model import evaluate_model


def train_model(movies_df: pd.DataFrame, ratings_df: pd.DataFrame,
                save_artifacts: bool = True) -> dict:
    """
    Complete model training pipeline.
    
    Steps:
    1. Clean and validate data
    2. Create binary target variable
    3. Split into train/test sets (stratified)
    4. Engineer features (aggregates from training data only)
    5. Train DecisionTreeClassifier
    6. Evaluate on held-out test set
    7. Save model and artifacts
    
    Returns a dictionary with model, metrics, and preprocessor.
    """
    print("=" * 60)
    print("CineScope — Model Training Pipeline")
    print("=" * 60)

    # ─── Step 1: Initialize preprocessor ─────────────────────
    preprocessor = DataPreprocessor()

    # ─── Step 2: Clean data ──────────────────────────────────
    print("\n── Step 1: Data Cleaning ──")
    movies_clean, ratings_clean = preprocessor.clean_data(movies_df, ratings_df)

    # ─── Step 3: Create target variable ──────────────────────
    print("\n── Step 2: Target Variable Creation ──")
    ratings_with_target = preprocessor.create_target(ratings_clean)

    # ─── Step 4: Train/test split ────────────────────────────
    print("\n── Step 3: Train/Test Split ──")
    train_df, test_df = preprocessor.split_data(ratings_with_target)

    # ─── Step 5: Feature engineering ─────────────────────────
    print("\n── Step 4: Feature Engineering ──")
    print("  Computing training features (from training data only)...")
    X_train = preprocessor.engineer_features(train_df, movies_clean, is_training=True)
    y_train = train_df["liked"].values

    print("  Computing test features (using training aggregates)...")
    X_test = preprocessor.engineer_features(test_df, movies_clean, is_training=False)
    y_test = test_df["liked"].values

    feature_names = preprocessor.get_feature_columns()

    # ─── Step 6: Train the model ─────────────────────────────
    print("\n── Step 5: Model Training ──")
    
    # DecisionTreeClassifier with controlled hyperparameters
    # max_depth limits tree complexity for interpretability
    # min_samples_leaf prevents overfitting to small groups
    # class_weight='balanced' adjusts for any class imbalance
    model = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=50,
        min_samples_split=100,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        criterion="gini"
    )

    print(f"  Model: DecisionTreeClassifier")
    print(f"  Parameters:")
    print(f"    max_depth: {model.max_depth}")
    print(f"    min_samples_leaf: {model.min_samples_leaf}")
    print(f"    min_samples_split: {model.min_samples_split}")
    print(f"    class_weight: {model.class_weight}")
    print(f"    random_state: {model.random_state}")
    print(f"    criterion: {model.criterion}")

    model.fit(X_train, y_train)
    print(f"  ✓ Model trained on {len(X_train):,} samples")
    print(f"  Tree depth: {model.get_depth()}")
    print(f"  Number of leaves: {model.get_n_leaves()}")

    # ─── Step 7: Evaluate on test set ────────────────────────
    print("\n── Step 6: Model Evaluation ──")
    metrics = evaluate_model(model, X_test, y_test, feature_names)

    # Add dataset info to metrics
    metrics["train_size"] = len(X_train)
    metrics["test_size"] = len(X_test)
    metrics["total_size"] = len(X_train) + len(X_test)
    metrics["positive_class_threshold"] = preprocessor.threshold
    metrics["random_seed"] = RANDOM_SEED
    metrics["model_params"] = model.get_params()

    # ─── Step 8: Save artifacts ──────────────────────────────
    if save_artifacts:
        print("\n── Step 7: Saving Artifacts ──")
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)

        # Save model
        model_path = os.path.join(MODELS_DIR, "decision_tree_model.joblib")
        joblib.dump(model, model_path)
        print(f"  ✓ Model saved: {model_path}")

        # Save preprocessor (contains fitted aggregates)
        preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        print(f"  ✓ Preprocessor saved: {preprocessor_path}")

        # Save feature names
        features_path = os.path.join(MODELS_DIR, "feature_names.json")
        with open(features_path, "w") as f:
            json.dump(feature_names, f, indent=2)
        print(f"  ✓ Feature names saved: {features_path}")

        # Save metrics to JSON
        metrics_path = os.path.join(REPORTS_DIR, "model_metrics.json")
        # Convert numpy types for JSON
        def convert_np(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        metrics_json = json.loads(json.dumps(metrics, default=convert_np))
        with open(metrics_path, "w") as f:
            json.dump(metrics_json, f, indent=2)
        print(f"  ✓ Metrics saved: {metrics_path}")

        # Save preprocessing stats
        preprocessor.save_stats()

        # Save metrics to MongoDB
        try:
            from database import CineScopeDB
            db = CineScopeDB()
            db.save_model_metrics(metrics_json)
            db.close()
        except Exception as e:
            print(f"  ⚠ Could not save metrics to MongoDB: {e}")

    print("\n" + "=" * 60)
    print("Training Pipeline Complete!")
    print("=" * 60)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "feature_names": feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
