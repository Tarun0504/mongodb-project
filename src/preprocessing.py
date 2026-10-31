"""
CineScope Data Preprocessing Module
Handles data cleaning, validation, feature engineering, and train/test splitting.
All preprocessing is designed to prevent data leakage.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, POSITIVE_RATING_THRESHOLD, TEST_SIZE, REPORTS_DIR
from src.schemas import MIN_RATING, MAX_RATING, NO_GENRES_LABEL, ALL_GENRES


class DataPreprocessor:
    """
    Preprocesses MovieLens data for machine learning.
    
    Key responsibilities:
    - Clean and validate raw data
    - Create binary classification target (like/dislike)
    - Engineer features from training data only
    - Prevent target leakage by computing aggregates on train split only
    - Create deterministic train/test splits
    """

    def __init__(self, random_seed: int = RANDOM_SEED,
                 threshold: float = POSITIVE_RATING_THRESHOLD,
                 test_size: float = TEST_SIZE):
        self.random_seed = random_seed
        self.threshold = threshold
        self.test_size = test_size
        self.preprocessing_stats = {}
        # Training-set aggregates (fitted during training only)
        self.global_avg_rating = None
        self.user_stats = None  # DataFrame with user-level aggregates
        self.movie_stats = None  # DataFrame with movie-level aggregates

    def clean_data(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> tuple:
        """
        Validate and clean the raw data.
        Returns cleaned (movies_df, ratings_df) and records preprocessing stats.
        """
        stats = {"before": {}, "after": {}, "removed": {}}
        
        # ─── Movies Cleaning ────────────────────────────────
        stats["before"]["movies"] = len(movies_df)

        # Remove rows with missing required fields
        movies_clean = movies_df.dropna(subset=["movieId", "title"]).copy()

        # Ensure correct data types
        movies_clean["movieId"] = movies_clean["movieId"].astype(int)
        movies_clean["title"] = movies_clean["title"].astype(str)

        # Parse genres into lists
        movies_clean["genre_list"] = movies_clean["genres"].apply(self._parse_genres)

        # Extract year from title
        movies_clean["year"] = movies_clean["title"].apply(self._extract_year)

        # Remove duplicate movieIds (keep first)
        movies_clean = movies_clean.drop_duplicates(subset=["movieId"], keep="first")

        stats["after"]["movies"] = len(movies_clean)
        stats["removed"]["movies"] = stats["before"]["movies"] - stats["after"]["movies"]

        # ─── Ratings Cleaning ────────────────────────────────
        stats["before"]["ratings"] = len(ratings_df)

        # Remove rows with missing required fields
        ratings_clean = ratings_df.dropna(subset=["userId", "movieId", "rating"]).copy()

        # Ensure correct data types
        ratings_clean["userId"] = ratings_clean["userId"].astype(int)
        ratings_clean["movieId"] = ratings_clean["movieId"].astype(int)
        ratings_clean["rating"] = ratings_clean["rating"].astype(float)

        # Remove ratings outside valid range
        valid_range = (ratings_clean["rating"] >= MIN_RATING) & (ratings_clean["rating"] <= MAX_RATING)
        invalid_count = (~valid_range).sum()
        if invalid_count > 0:
            print(f"  ⚠ Removing {invalid_count} ratings outside [{MIN_RATING}, {MAX_RATING}]")
        ratings_clean = ratings_clean[valid_range]

        # Remove duplicate user-movie pairs (keep last = most recent rating)
        dup_count = ratings_clean.duplicated(subset=["userId", "movieId"]).sum()
        if dup_count > 0:
            print(f"  ⚠ Removing {dup_count} duplicate user-movie pairs (keeping last)")
            ratings_clean = ratings_clean.sort_values("timestamp").drop_duplicates(
                subset=["userId", "movieId"], keep="last"
            )

        # Only keep ratings for movies that exist in our movies dataset
        valid_movies = set(movies_clean["movieId"])
        ratings_clean = ratings_clean[ratings_clean["movieId"].isin(valid_movies)]

        stats["after"]["ratings"] = len(ratings_clean)
        stats["removed"]["ratings"] = stats["before"]["ratings"] - stats["after"]["ratings"]

        # ─── Dataset Statistics ──────────────────────────────
        stats["unique_users"] = int(ratings_clean["userId"].nunique())
        stats["unique_movies_rated"] = int(ratings_clean["movieId"].nunique())
        stats["rating_distribution"] = ratings_clean["rating"].value_counts().sort_index().to_dict()

        self.preprocessing_stats["cleaning"] = stats
        
        print(f"\n─── Cleaning Summary ───")
        print(f"  Movies: {stats['before']['movies']:,} → {stats['after']['movies']:,} "
              f"(removed {stats['removed']['movies']})")
        print(f"  Ratings: {stats['before']['ratings']:,} → {stats['after']['ratings']:,} "
              f"(removed {stats['removed']['ratings']})")
        print(f"  Unique users: {stats['unique_users']:,}")
        print(f"  Unique movies with ratings: {stats['unique_movies_rated']:,}")

        return movies_clean, ratings_clean

    def create_target(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary classification target variable.
        Rating >= threshold → 1 (likely to like)
        Rating < threshold  → 0 (unlikely to like)
        """
        df = ratings_df.copy()
        df["liked"] = (df["rating"] >= self.threshold).astype(int)

        class_dist = df["liked"].value_counts()
        total = len(df)
        pos_count = int(class_dist.get(1, 0))
        neg_count = int(class_dist.get(0, 0))
        
        self.preprocessing_stats["target"] = {
            "threshold": self.threshold,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_ratio": round(pos_count / total, 4),
            "negative_ratio": round(neg_count / total, 4)
        }
        
        print(f"\n─── Target Variable ───")
        print(f"  Threshold: rating >= {self.threshold} → liked (positive class)")
        print(f"  Positive (liked): {pos_count:,} ({pos_count/total*100:.1f}%)")
        print(f"  Negative (not liked): {neg_count:,} ({neg_count/total*100:.1f}%)")

        return df

    def split_data(self, df: pd.DataFrame) -> tuple:
        """
        Create a stratified train/test split.
        
        Uses stratified sampling on the target variable to maintain
        class distribution in both splits.
        
        Note: A random stratified split evaluates classification on held-out
        rating records but does not guarantee generalization to entirely
        unseen users or movies. Some users and movies may appear in both
        the training and test sets with different ratings.
        """
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=df["liked"]
        )

        self.preprocessing_stats["split"] = {
            "train_size": len(train_df),
            "test_size": len(test_df),
            "total_size": len(df),
            "test_ratio": round(len(test_df) / len(df), 4),
            "train_positive_ratio": round(train_df["liked"].mean(), 4),
            "test_positive_ratio": round(test_df["liked"].mean(), 4),
            "random_seed": self.random_seed
        }

        print(f"\n─── Train/Test Split ───")
        print(f"  Random seed: {self.random_seed}")
        print(f"  Train: {len(train_df):,} ({(1-self.test_size)*100:.0f}%)")
        print(f"  Test:  {len(test_df):,} ({self.test_size*100:.0f}%)")
        print(f"  Train positive ratio: {train_df['liked'].mean():.4f}")
        print(f"  Test positive ratio:  {test_df['liked'].mean():.4f}")

        return train_df, test_df

    def fit_aggregates(self, train_df: pd.DataFrame):
        """
        Compute user and movie aggregate statistics from TRAINING DATA ONLY.
        This prevents target leakage from test data.
        """
        # Global average rating (from training set)
        self.global_avg_rating = float(train_df["rating"].mean())

        # User-level aggregates from training data
        self.user_stats = train_df.groupby("userId").agg(
            user_avg_rating=("rating", "mean"),
            user_rating_count=("rating", "count"),
            user_liked_ratio=("liked", "mean")
        ).reset_index()
        self.user_stats["user_avg_rating"] = self.user_stats["user_avg_rating"].round(4)
        self.user_stats["user_log_count"] = np.log1p(self.user_stats["user_rating_count"]).round(4)

        # Movie-level aggregates from training data
        self.movie_stats = train_df.groupby("movieId").agg(
            movie_avg_rating=("rating", "mean"),
            movie_rating_count=("rating", "count"),
            movie_liked_ratio=("liked", "mean")
        ).reset_index()
        self.movie_stats["movie_avg_rating"] = self.movie_stats["movie_avg_rating"].round(4)
        self.movie_stats["movie_log_count"] = np.log1p(self.movie_stats["movie_rating_count"]).round(4)

        print(f"\n─── Aggregate Statistics (Training Set Only) ───")
        print(f"  Global average rating: {self.global_avg_rating:.4f}")
        print(f"  Users with stats: {len(self.user_stats):,}")
        print(f"  Movies with stats: {len(self.movie_stats):,}")

    def engineer_features(self, df: pd.DataFrame, movies_df: pd.DataFrame,
                          is_training: bool = True) -> pd.DataFrame:
        """
        Create features for the ML model.
        
        Features:
        - One-hot encoded genres (binary vector for each genre)
        - User's historical average rating (from training data)
        - User's historical number of ratings (log-transformed, from training data)
        - Movie's average rating (from training data)
        - Movie's rating count (log-transformed, from training data)
        
        For unseen users/movies, uses global average as fallback.
        """
        if is_training:
            self.fit_aggregates(df)

        features = df[["userId", "movieId"]].copy()

        # ─── Merge user statistics ───────────────────────────
        features = features.merge(self.user_stats, on="userId", how="left")
        features["user_avg_rating"] = features["user_avg_rating"].fillna(self.global_avg_rating)
        features["user_rating_count"] = features["user_rating_count"].fillna(0)
        features["user_log_count"] = features["user_log_count"].fillna(0.0)

        # ─── Merge movie statistics ──────────────────────────
        features = features.merge(self.movie_stats, on="movieId", how="left")
        features["movie_avg_rating"] = features["movie_avg_rating"].fillna(self.global_avg_rating)
        features["movie_rating_count"] = features["movie_rating_count"].fillna(0)
        features["movie_log_count"] = features["movie_log_count"].fillna(0.0)

        # ─── One-hot encode genres ───────────────────────────
        # Get genres for each movie
        movie_genres = movies_df[["movieId", "genre_list"]].drop_duplicates(subset=["movieId"])
        features = features.merge(movie_genres, on="movieId", how="left")
        features["genre_list"] = features["genre_list"].apply(
            lambda x: x if isinstance(x, list) else []
        )

        # Create binary columns for each genre
        for genre in ALL_GENRES:
            col_name = f"genre_{genre.lower().replace('-', '_')}"
            features[col_name] = features["genre_list"].apply(
                lambda genres: 1 if genre in genres else 0
            )

        # ─── Select final feature columns ────────────────────
        feature_columns = (
            ["user_avg_rating", "user_log_count", "user_rating_count",
             "movie_avg_rating", "movie_log_count", "movie_rating_count"] +
            [f"genre_{g.lower().replace('-', '_')}" for g in ALL_GENRES]
        )

        self.feature_columns = feature_columns

        print(f"  ✓ Engineered {len(feature_columns)} features for {len(features):,} samples")

        return features[feature_columns]

    def get_feature_columns(self) -> list:
        """Return the list of feature column names."""
        return self.feature_columns

    def save_stats(self, filepath: str = None):
        """Save preprocessing statistics to a JSON file."""
        if filepath is None:
            os.makedirs(REPORTS_DIR, exist_ok=True)
            filepath = os.path.join(REPORTS_DIR, "preprocessing_stats.json")

        # Convert any numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        stats = json.loads(json.dumps(self.preprocessing_stats, default=convert))
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Preprocessing stats saved to {filepath}")

    # ─── Private Helpers ─────────────────────────────────────

    @staticmethod
    def _parse_genres(genres_str) -> list:
        """Parse pipe-separated genre string into a list."""
        if pd.isna(genres_str) or genres_str == NO_GENRES_LABEL:
            return []
        return [g.strip() for g in str(genres_str).split("|") if g.strip()]

    @staticmethod
    def _extract_year(title: str):
        """Extract year from movie title like 'Toy Story (1995)'."""
        import re
        match = re.search(r'\((\d{4})\)\s*$', str(title))
        if match:
            return int(match.group(1))
        return None
