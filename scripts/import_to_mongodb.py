"""
CineScope MongoDB Import Script
Imports the MovieLens dataset into MongoDB collections.
Idempotent — safe to run multiple times without duplicating data.
"""
import os
import sys
import re
from datetime import datetime, timezone

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR
from database import CineScopeDB
from src.dataset_loader import DatasetLoader
from src.schemas import NO_GENRES_LABEL


def parse_year_from_title(title: str):
    """Extract release year from movie title, e.g. 'Toy Story (1995)' -> 1995."""
    match = re.search(r'\((\d{4})\)\s*$', title)
    if match:
        return int(match.group(1))
    return None


def import_dataset():
    """Import MovieLens dataset into MongoDB."""
    print("=" * 60)
    print("CineScope — MongoDB Dataset Import")
    print("=" * 60)

    # Initialize database
    db = CineScopeDB()
    db.create_indexes()

    # Load CSV data
    loader = DatasetLoader()
    data = loader.load_all()

    movies_df = data["movies"]
    ratings_df = data["ratings"]
    links_df = data["links"]

    # ─── Compute rating statistics per movie ─────────────────
    print("\n─── Computing Rating Statistics ───")
    rating_stats = ratings_df.groupby("movieId").agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()
    rating_stats["average_rating"] = rating_stats["average_rating"].round(2)
    print(f"  ✓ Computed stats for {len(rating_stats):,} movies")

    # Merge stats into movies
    movies_merged = movies_df.merge(rating_stats, on="movieId", how="left")
    movies_merged["average_rating"] = movies_merged["average_rating"].fillna(0.0)
    movies_merged["rating_count"] = movies_merged["rating_count"].fillna(0).astype(int)

    # Merge external IDs if available
    if not links_df.empty and "movieId" in links_df.columns:
        movies_merged = movies_merged.merge(links_df, on="movieId", how="left")

    # ─── Prepare movie documents ─────────────────────────────
    print("\n─── Preparing Movie Documents ───")
    movie_docs = []
    for _, row in movies_merged.iterrows():
        # Parse genres
        genres_str = row.get("genres", "")
        if genres_str == NO_GENRES_LABEL or pd.isna(genres_str):
            genres = []
        else:
            genres = [g.strip() for g in str(genres_str).split("|") if g.strip()]

        doc = {
            "movieId": int(row["movieId"]),
            "title": str(row["title"]),
            "genres": genres,
            "year": parse_year_from_title(str(row["title"])),
            "average_rating": float(row["average_rating"]),
            "rating_count": int(row["rating_count"]),
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        # Add external IDs if available
        external_ids = {}
        if "imdbId" in row and pd.notna(row["imdbId"]):
            external_ids["imdbId"] = int(row["imdbId"])
        if "tmdbId" in row and pd.notna(row["tmdbId"]):
            external_ids["tmdbId"] = int(row["tmdbId"])
        if external_ids:
            doc["external_ids"] = external_ids

        movie_docs.append(doc)

    print(f"  ✓ Prepared {len(movie_docs):,} movie documents")

    # ─── Import movies ───────────────────────────────────────
    print("\n─── Importing Movies ───")
    existing_movies = db.get_movie_count()
    if existing_movies > 0:
        print(f"  ⓘ {existing_movies:,} movies already in database")

    # Insert in batches of 1000
    batch_size = 1000
    total_inserted = 0
    for i in range(0, len(movie_docs), batch_size):
        batch = movie_docs[i:i + batch_size]
        inserted = db.bulk_insert_movies(batch)
        total_inserted += inserted

    final_movies = db.get_movie_count()
    print(f"  ✓ Movies in database: {final_movies:,} (inserted {total_inserted:,} new)")

    # ─── Prepare rating documents ────────────────────────────
    print("\n─── Preparing Rating Documents ───")
    rating_docs = []
    for _, row in ratings_df.iterrows():
        doc = {
            "userId": int(row["userId"]),
            "movieId": int(row["movieId"]),
            "rating": float(row["rating"]),
            "timestamp": int(row["timestamp"]),
            "source": "movielens"  # Mark as imported data
        }
        rating_docs.append(doc)

    print(f"  ✓ Prepared {len(rating_docs):,} rating documents")

    # ─── Import ratings ──────────────────────────────────────
    print("\n─── Importing Ratings ───")
    existing_ratings = db.get_rating_count()
    if existing_ratings > 0:
        print(f"  ⓘ {existing_ratings:,} ratings already in database")

    total_inserted = 0
    for i in range(0, len(rating_docs), batch_size):
        batch = rating_docs[i:i + batch_size]
        inserted = db.bulk_insert_ratings(batch)
        total_inserted += inserted
        if (i + batch_size) % 10000 == 0 or i + batch_size >= len(rating_docs):
            print(f"  ... processed {min(i + batch_size, len(rating_docs)):,} / {len(rating_docs):,}")

    final_ratings = db.get_rating_count()
    print(f"  ✓ Ratings in database: {final_ratings:,} (inserted {total_inserted:,} new)")

    # ─── Create demo user profiles ───────────────────────────
    print("\n─── Creating Demo User Profiles ───")
    demo_users = [
        {"user_id": "demo_alice", "display_name": "Alice", "preferred_genres": ["Comedy", "Romance", "Drama"]},
        {"user_id": "demo_bob", "display_name": "Bob", "preferred_genres": ["Action", "Sci-Fi", "Thriller"]},
        {"user_id": "demo_carol", "display_name": "Carol", "preferred_genres": ["Documentary", "Drama", "War"]},
        {"user_id": "guest", "display_name": "Guest User", "preferred_genres": []},
    ]
    for user in demo_users:
        created = db.create_user(user["user_id"], user["display_name"], user["preferred_genres"])
        status = "created" if created else "already exists"
        print(f"  ✓ {user['display_name']}: {status}")

    # ─── Final Summary ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Import Complete!")
    print("=" * 60)
    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Report validation summary
    report = loader.get_validation_report()
    print(f"\nDataset validation report:")
    for name, info in report.items():
        print(f"  {name}: {info}")

    db.close()
    return stats


if __name__ == "__main__":
    import_dataset()
