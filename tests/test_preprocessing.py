"""
Unit Tests for CineScope Preprocessing and Feature Engineering
"""
import pandas as pd
import numpy as np
import pytest

from src.preprocessing import DataPreprocessor
from src.schemas import MIN_RATING, MAX_RATING


@pytest.fixture
def sample_data():
    """Create sample movie and rating data for testing."""
    movies_df = pd.DataFrame({
        "movieId": [1, 2, 3, 4],
        "title": ["Toy Story (1995)", "Jumanji (1995)", "Heat (1995)", "Bad Data Movie"],
        "genres": ["Adventure|Animation|Children", "Adventure|Children|Fantasy", "Action|Crime|Thriller", "(no genres listed)"]
    })

    ratings_df = pd.DataFrame({
        "userId": [1, 1, 2, 2, 3, 3, 3],
        "movieId": [1, 2, 1, 3, 1, 2, 4],
        "rating": [4.0, 3.5, 5.0, 2.0, 4.5, 3.0, 1.0],
        "timestamp": [1000, 1001, 1002, 1003, 1004, 1005, 1006]
    })

    return movies_df, ratings_df


def test_clean_data(sample_data):
    """Test data cleaning and year extraction."""
    movies_df, ratings_df = sample_data
    preprocessor = DataPreprocessor()
    
    clean_movies, clean_ratings = preprocessor.clean_data(movies_df, ratings_df)
    
    assert len(clean_movies) == 4
    assert clean_movies.loc[clean_movies["movieId"] == 1, "year"].iloc[0] == 1995
    assert len(clean_ratings) == 7
    assert all(clean_ratings["rating"] >= MIN_RATING)
    assert all(clean_ratings["rating"] <= MAX_RATING)


def test_create_target(sample_data):
    """Test binary classification target creation (rating >= 4.0)."""
    _, ratings_df = sample_data
    preprocessor = DataPreprocessor(threshold=4.0)
    
    df_with_target = preprocessor.create_target(ratings_df)
    assert "liked" in df_with_target.columns
    # Check that ratings >= 4.0 are 1, < 4.0 are 0
    for _, row in df_with_target.iterrows():
        if row["rating"] >= 4.0:
            assert row["liked"] == 1
        else:
            assert row["liked"] == 0


def test_train_test_split(sample_data):
    """Test deterministic stratified train/test split."""
    _, ratings_df = sample_data
    preprocessor = DataPreprocessor(random_seed=42, test_size=0.3)
    df_target = preprocessor.create_target(ratings_df)

    train_df, test_df = preprocessor.split_data(df_target)
    assert len(train_df) + len(test_df) == len(df_target)
    assert abs(len(test_df) / len(df_target) - 0.3) < 0.2


def test_no_data_leakage_in_aggregates(sample_data):
    """Test that aggregates are fitted strictly on the training partition."""
    movies_df, ratings_df = sample_data
    preprocessor = DataPreprocessor(random_seed=42)
    clean_movies, clean_ratings = preprocessor.clean_data(movies_df, ratings_df)
    df_target = preprocessor.create_target(clean_ratings)
    train_df, test_df = preprocessor.split_data(df_target)

    # Fit features on train only
    X_train = preprocessor.engineer_features(train_df, clean_movies, is_training=True)
    assert preprocessor.global_avg_rating == pytest.approx(train_df["rating"].mean(), 0.001)

    # Transform test without re-fitting
    X_test = preprocessor.engineer_features(test_df, clean_movies, is_training=False)
    assert len(X_test) == len(test_df)
    assert len(X_train.columns) == len(X_test.columns)
