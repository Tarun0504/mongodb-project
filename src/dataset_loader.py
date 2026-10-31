"""
CineScope Dataset Loader
Validates CSV files and loads them into pandas DataFrames.
Reports record counts and validates schema.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR
from src.schemas import (
    MOVIES_CSV_COLUMNS, RATINGS_CSV_COLUMNS,
    TAGS_CSV_COLUMNS, LINKS_CSV_COLUMNS,
    MIN_RATING, MAX_RATING, NO_GENRES_LABEL
)


class DatasetLoader:
    """Loads and validates the MovieLens CSV dataset files."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(DATA_DIR, "ml-latest-small")
        self.validation_report = {}

    def _csv_path(self, filename: str) -> str:
        return os.path.join(self.data_dir, filename)

    def _validate_columns(self, df: pd.DataFrame, expected: list, name: str) -> bool:
        """Check that a DataFrame has the expected columns."""
        missing = set(expected) - set(df.columns)
        if missing:
            print(f"  ✗ {name}: Missing columns: {missing}")
            return False
        print(f"  ✓ {name}: All required columns present")
        return True

    def load_movies(self) -> pd.DataFrame:
        """Load and validate movies.csv."""
        path = self._csv_path("movies.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"movies.csv not found at {path}")

        df = pd.read_csv(path)
        self._validate_columns(df, MOVIES_CSV_COLUMNS, "movies.csv")

        # Report statistics
        self.validation_report["movies"] = {
            "total_rows": len(df),
            "unique_movie_ids": df["movieId"].nunique(),
            "null_values": df.isnull().sum().to_dict(),
            "duplicate_ids": df["movieId"].duplicated().sum()
        }
        print(f"  ✓ movies.csv: {len(df):,} rows, {df['movieId'].nunique():,} unique movies")

        return df

    def load_ratings(self) -> pd.DataFrame:
        """Load and validate ratings.csv."""
        path = self._csv_path("ratings.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"ratings.csv not found at {path}")

        df = pd.read_csv(path)
        self._validate_columns(df, RATINGS_CSV_COLUMNS, "ratings.csv")

        # Validate rating range
        out_of_range = ((df["rating"] < MIN_RATING) | (df["rating"] > MAX_RATING)).sum()
        if out_of_range > 0:
            print(f"  ⚠ ratings.csv: {out_of_range} ratings outside [{MIN_RATING}, {MAX_RATING}]")

        # Check for duplicate user-movie pairs
        duplicates = df.duplicated(subset=["userId", "movieId"]).sum()
        if duplicates > 0:
            print(f"  ⚠ ratings.csv: {duplicates} duplicate user-movie pairs")

        self.validation_report["ratings"] = {
            "total_rows": len(df),
            "unique_users": df["userId"].nunique(),
            "unique_movies": df["movieId"].nunique(),
            "rating_range": [float(df["rating"].min()), float(df["rating"].max())],
            "null_values": df.isnull().sum().to_dict(),
            "duplicate_pairs": int(duplicates),
            "out_of_range": int(out_of_range)
        }
        print(f"  ✓ ratings.csv: {len(df):,} rows, {df['userId'].nunique():,} users, "
              f"{df['movieId'].nunique():,} movies")

        return df

    def load_tags(self) -> pd.DataFrame:
        """Load tags.csv (optional)."""
        path = self._csv_path("tags.csv")
        if not os.path.exists(path):
            print("  ⓘ tags.csv not found (optional file)")
            return pd.DataFrame(columns=TAGS_CSV_COLUMNS)
        
        df = pd.read_csv(path)
        self._validate_columns(df, TAGS_CSV_COLUMNS, "tags.csv")
        self.validation_report["tags"] = {
            "total_rows": len(df),
            "unique_tags": df["tag"].nunique() if "tag" in df.columns else 0
        }
        print(f"  ✓ tags.csv: {len(df):,} rows")
        return df

    def load_links(self) -> pd.DataFrame:
        """Load links.csv (optional)."""
        path = self._csv_path("links.csv")
        if not os.path.exists(path):
            print("  ⓘ links.csv not found (optional file)")
            return pd.DataFrame(columns=LINKS_CSV_COLUMNS)
        
        df = pd.read_csv(path)
        self._validate_columns(df, LINKS_CSV_COLUMNS, "links.csv")
        self.validation_report["links"] = {
            "total_rows": len(df),
        }
        print(f"  ✓ links.csv: {len(df):,} rows")
        return df

    def load_all(self) -> dict:
        """Load all dataset files and return as a dictionary of DataFrames."""
        print("\n─── Loading MovieLens Dataset ───")
        data = {
            "movies": self.load_movies(),
            "ratings": self.load_ratings(),
            "tags": self.load_tags(),
            "links": self.load_links()
        }
        print(f"\n✓ Dataset loading complete")
        print(f"  Validation report: {self.validation_report}")
        return data

    def get_validation_report(self) -> dict:
        """Return the validation report from the most recent load."""
        return self.validation_report


if __name__ == "__main__":
    loader = DatasetLoader()
    data = loader.load_all()
    print(f"\nDataset summary:")
    for name, df in data.items():
        print(f"  {name}: {len(df):,} rows, {len(df.columns)} columns")
