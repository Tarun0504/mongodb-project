"""
CineScope Database Module
Manages MongoDB connection and exposes reusable database operations.
All database interactions go through this module.
"""
import os
import sys
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, BulkWriteError, DuplicateKeyError
from pymongo.collection import Collection

from config import (
    MONGODB_URI, MONGODB_DATABASE,
    MOVIES_COLLECTION, RATINGS_COLLECTION, USERS_COLLECTION,
    PREDICTIONS_COLLECTION, MODEL_METRICS_COLLECTION
)


class CineScopeDB:
    """
    Database manager for the CineScope application.
    Handles connection, collection access, indexing, and all CRUD operations.
    """

    def __init__(self, uri: str = None, db_name: str = None):
        """Initialize database connection."""
        self.uri = uri or MONGODB_URI
        self.db_name = db_name or MONGODB_DATABASE
        self.client = None
        self.db = None
        self._connect()

    def _connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Verify connection
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            print(f"✓ Connected to MongoDB: {self.db_name}")
        except ConnectionFailure as e:
            print(f"\n✗ Could not connect to MongoDB at {self.uri}")
            print(f"  Error: {e}")
            print(f"\n  Setup instructions:")
            print(f"  1. Install MongoDB: brew install mongodb-community")
            print(f"  2. Start MongoDB:   brew services start mongodb-community")
            print(f"  3. Or run:          mongod --dbpath /tmp/mongodb-data")
            print(f"  4. Verify:          mongosh --eval 'db.runCommand({{ping:1}})'")
            raise

    def close(self):
        """Close the database connection."""
        if self.client:
            self.client.close()
            print("✓ MongoDB connection closed")

    # ─── Collection Access ───────────────────────────────────────

    @property
    def movies(self) -> Collection:
        return self.db[MOVIES_COLLECTION]

    @property
    def ratings(self) -> Collection:
        return self.db[RATINGS_COLLECTION]

    @property
    def users(self) -> Collection:
        return self.db[USERS_COLLECTION]

    @property
    def predictions(self) -> Collection:
        return self.db[PREDICTIONS_COLLECTION]

    @property
    def model_metrics(self) -> Collection:
        return self.db[MODEL_METRICS_COLLECTION]

    # ─── Index Creation ──────────────────────────────────────────

    def create_indexes(self):
        """Create all required indexes for optimal query performance."""
        # Movies indexes
        self.movies.create_indexes([
            IndexModel([("movieId", ASCENDING)], unique=True, name="idx_movieId"),
            IndexModel([("title", ASCENDING)], name="idx_title"),
            IndexModel([("genres", ASCENDING)], name="idx_genres"),
            IndexModel([("average_rating", DESCENDING)], name="idx_avg_rating"),
            IndexModel([("rating_count", DESCENDING)], name="idx_rating_count"),
        ])

        # Ratings indexes — compound unique index ensures one rating per user per movie
        self.ratings.create_indexes([
            IndexModel([("userId", ASCENDING), ("movieId", ASCENDING)],
                       unique=True, name="idx_user_movie"),
            IndexModel([("movieId", ASCENDING)], name="idx_rating_movieId"),
            IndexModel([("userId", ASCENDING)], name="idx_rating_userId"),
        ])

        # Users indexes
        self.users.create_indexes([
            IndexModel([("user_id", ASCENDING)], unique=True, name="idx_user_id"),
        ])

        # Predictions indexes
        self.predictions.create_indexes([
            IndexModel([("user_id", ASCENDING), ("movieId", ASCENDING)],
                       name="idx_pred_user_movie"),
        ])

        # Model metrics indexes
        self.model_metrics.create_indexes([
            IndexModel([("model_name", ASCENDING), ("evaluation_date", DESCENDING)],
                       name="idx_model_eval"),
        ])

        print("✓ All database indexes created")

    # ─── Movie Operations ────────────────────────────────────────

    def bulk_insert_movies(self, movie_docs: List[Dict]) -> int:
        """
        Insert movies in bulk. Uses ordered=False for performance.
        Skips duplicates based on unique movieId index.
        Returns the number of successfully inserted documents.
        """
        if not movie_docs:
            return 0
        try:
            result = self.movies.insert_many(movie_docs, ordered=False)
            return len(result.inserted_ids)
        except BulkWriteError as e:
            # Some documents may be duplicates — that's expected for idempotent imports
            inserted = e.details.get("nInserted", 0)
            write_errors = e.details.get("writeErrors", [])
            dup_errors = sum(1 for err in write_errors if err.get("code") == 11000)
            other_errors = len(write_errors) - dup_errors
            if other_errors > 0:
                print(f"  ⚠ {other_errors} non-duplicate write errors occurred")
            return inserted

    def bulk_insert_ratings(self, rating_docs: List[Dict]) -> int:
        """
        Insert ratings in bulk. Skips duplicates based on unique compound index.
        Returns the number of successfully inserted documents.
        """
        if not rating_docs:
            return 0
        try:
            result = self.ratings.insert_many(rating_docs, ordered=False)
            return len(result.inserted_ids)
        except BulkWriteError as e:
            inserted = e.details.get("nInserted", 0)
            write_errors = e.details.get("writeErrors", [])
            dup_errors = sum(1 for err in write_errors if err.get("code") == 11000)
            other_errors = len(write_errors) - dup_errors
            if other_errors > 0:
                print(f"  ⚠ {other_errors} non-duplicate write errors occurred")
            return inserted

    def get_movie_by_id(self, movie_id: int) -> Optional[Dict]:
        """Retrieve a single movie by its movieId."""
        return self.movies.find_one({"movieId": movie_id}, {"_id": 0})

    def search_movies_by_title(self, query: str, limit: int = 20) -> List[Dict]:
        """Search movies by title using case-insensitive regex."""
        import re
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        return list(self.movies.find(
            {"title": {"$regex": pattern}},
            {"_id": 0}
        ).sort("rating_count", DESCENDING).limit(limit))

    def get_movies_by_genre(self, genre: str, limit: int = 20, skip: int = 0) -> List[Dict]:
        """Filter movies that contain the specified genre."""
        return list(self.movies.find(
            {"genres": genre},
            {"_id": 0}
        ).sort("average_rating", DESCENDING).limit(limit).skip(skip))

    def query_movies(self, query: str = None, genre: str = None, sort_by: str = "Popularity",
                     min_ratings: int = 0, limit: int = 24, skip: int = 0) -> List[Dict]:
        """
        Unified movie query combining text search, genre filtering, and flexible sorting.
        Supports:
          - sort_by='Popularity': rating_count DESC, average_rating DESC
          - sort_by='Highest Rated': average_rating DESC, rating_count DESC
          - sort_by='Title A-Z': title ASC with case-insensitive collation
          - sort_by='Title Z-A': title DESC with case-insensitive collation
          - sort_by='Newest (Year)': year DESC, rating_count DESC
        """
        import re
        from pymongo.collation import Collation

        filter_doc = {}

        if query and query.strip():
            pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
            filter_doc["title"] = {"$regex": pattern}

        if genre and genre != "All Genres":
            filter_doc["genres"] = genre

        if min_ratings > 0:
            filter_doc["rating_count"] = {"$gte": min_ratings}

        # Sorting strategy
        collation = None
        if sort_by == "Title A-Z":
            sort_criteria = [("title", ASCENDING)]
            collation = Collation(locale="en", strength=2)
        elif sort_by == "Title Z-A":
            sort_criteria = [("title", DESCENDING)]
            collation = Collation(locale="en", strength=2)
        elif sort_by == "Highest Rated":
            if min_ratings == 0 and not (query and query.strip()):
                filter_doc["rating_count"] = {"$gte": 5}
            sort_criteria = [("average_rating", DESCENDING), ("rating_count", DESCENDING)]
        elif sort_by == "Newest (Year)":
            sort_criteria = [("year", DESCENDING), ("rating_count", DESCENDING)]
        else:  # Default: "Popularity"
            sort_criteria = [("rating_count", DESCENDING), ("average_rating", DESCENDING)]

        cursor = self.movies.find(filter_doc, {"_id": 0})
        if collation:
            cursor = cursor.collation(collation)

        return list(cursor.sort(sort_criteria).skip(skip).limit(limit))

    def count_queried_movies(self, query: str = None, genre: str = None, min_ratings: int = 0) -> int:
        """Count total movies matching given search and genre criteria."""
        import re
        filter_doc = {}
        if query and query.strip():
            pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
            filter_doc["title"] = {"$regex": pattern}
        if genre and genre != "All Genres":
            filter_doc["genres"] = genre
        if min_ratings > 0:
            filter_doc["rating_count"] = {"$gte": min_ratings}
        return self.movies.count_documents(filter_doc)

    def get_top_rated_movies(self, min_ratings: int = 10, limit: int = 20) -> List[Dict]:
        """Retrieve top-rated movies with a minimum number of ratings."""
        return list(self.movies.find(
            {"rating_count": {"$gte": min_ratings}},
            {"_id": 0}
        ).sort("average_rating", DESCENDING).limit(limit))

    def get_popular_movies(self, limit: int = 20) -> List[Dict]:
        """Retrieve most popular movies by rating count."""
        return list(self.movies.find(
            {"rating_count": {"$gte": 1}},
            {"_id": 0}
        ).sort("rating_count", DESCENDING).limit(limit))

    def get_all_movies(self, projection: Dict = None) -> List[Dict]:
        """Retrieve all movies with optional projection."""
        proj = projection or {"_id": 0}
        return list(self.movies.find({}, proj))

    def get_movie_count(self) -> int:
        """Count total movies in the collection."""
        return self.movies.count_documents({})

    def get_all_genres(self) -> List[str]:
        """Retrieve all distinct genres from the movies collection."""
        return sorted(self.movies.distinct("genres"))

    def update_movie_stats(self, movie_id: int, avg_rating: float, rating_count: int):
        """Update a movie's average rating and rating count."""
        self.movies.update_one(
            {"movieId": movie_id},
            {"$set": {
                "average_rating": round(avg_rating, 2),
                "rating_count": rating_count
            }}
        )

    # ─── Rating Operations ───────────────────────────────────────

    def get_movie_ratings(self, movie_id: int) -> List[Dict]:
        """Retrieve all ratings for a specific movie."""
        return list(self.ratings.find(
            {"movieId": movie_id},
            {"_id": 0}
        ))

    def get_movie_rating_stats(self, movie_id: int) -> Dict:
        """Get aggregated rating statistics for a movie using aggregation pipeline."""
        pipeline = [
            {"$match": {"movieId": movie_id}},
            {"$group": {
                "_id": "$movieId",
                "average_rating": {"$avg": "$rating"},
                "rating_count": {"$sum": 1},
                "min_rating": {"$min": "$rating"},
                "max_rating": {"$max": "$rating"}
            }}
        ]
        results = list(self.ratings.aggregate(pipeline))
        if results:
            stats = results[0]
            stats["average_rating"] = round(stats["average_rating"], 2)
            return stats
        return {"average_rating": 0, "rating_count": 0, "min_rating": 0, "max_rating": 0}

    def get_user_ratings(self, user_id) -> List[Dict]:
        """Retrieve all ratings submitted by a specific user."""
        return list(self.ratings.find(
            {"userId": user_id},
            {"_id": 0}
        ).sort("timestamp", DESCENDING))

    def get_user_rating_for_movie(self, user_id, movie_id: int) -> Optional[Dict]:
        """Get a specific user's rating for a specific movie."""
        return self.ratings.find_one(
            {"userId": user_id, "movieId": movie_id},
            {"_id": 0}
        )

    def upsert_rating(self, user_id, movie_id: int, rating: float, source: str = "app") -> str:
        """
        Insert or update a user rating. Uses upsert to handle both cases.
        Returns 'inserted' or 'updated'.
        """
        now = datetime.now(timezone.utc)
        result = self.ratings.update_one(
            {"userId": user_id, "movieId": movie_id},
            {"$set": {
                "rating": rating,
                "timestamp": int(now.timestamp()),
                "rated_at": now.isoformat(),
                "source": source
            }},
            upsert=True
        )
        # After upserting, recalculate movie stats
        stats = self.get_movie_rating_stats(movie_id)
        self.update_movie_stats(movie_id, stats["average_rating"], stats["rating_count"])

        if result.upserted_id:
            return "inserted"
        return "updated"

    def get_rating_count(self) -> int:
        """Count total ratings in the collection."""
        return self.ratings.count_documents({})

    # ─── User/Profile Operations ─────────────────────────────────

    def create_user(self, user_id: str, display_name: str,
                    preferred_genres: List[str] = None) -> bool:
        """Create a new user profile. Returns True if created, False if exists."""
        try:
            self.users.insert_one({
                "user_id": user_id,
                "display_name": display_name,
                "preferred_genres": preferred_genres or [],
                "created_at": datetime.now(timezone.utc).isoformat()
            })
            return True
        except DuplicateKeyError:
            return False

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Retrieve a user profile."""
        return self.users.find_one({"user_id": user_id}, {"_id": 0})

    def get_all_users(self) -> List[Dict]:
        """Retrieve all user profiles."""
        return list(self.users.find({}, {"_id": 0}))

    def update_user_genres(self, user_id: str, genres: List[str]):
        """Update a user's preferred genres."""
        self.users.update_one(
            {"user_id": user_id},
            {"$set": {"preferred_genres": genres}}
        )

    def get_user_count(self) -> int:
        """Count registered user profiles."""
        return self.users.count_documents({})

    # ─── Prediction Operations ───────────────────────────────────

    def save_prediction(self, user_id: str, movie_id: int, predicted_class: int,
                        predicted_probability: float = None, model_version: str = "v1"):
        """Save a prediction result."""
        self.predictions.update_one(
            {"user_id": user_id, "movieId": movie_id},
            {"$set": {
                "predicted_class": predicted_class,
                "predicted_probability": round(predicted_probability, 4) if predicted_probability else None,
                "model_version": model_version,
                "created_at": datetime.now(timezone.utc).isoformat()
            }},
            upsert=True
        )

    def get_predictions_for_user(self, user_id: str) -> List[Dict]:
        """Retrieve all predictions for a user."""
        return list(self.predictions.find(
            {"user_id": user_id},
            {"_id": 0}
        ))

    def get_prediction_count(self) -> int:
        """Count total predictions."""
        return self.predictions.count_documents({})

    def save_bulk_predictions(self, prediction_docs: List[Dict]):
        """Save multiple predictions using bulk operations."""
        if not prediction_docs:
            return
        from pymongo import UpdateOne
        operations = []
        for pred in prediction_docs:
            operations.append(UpdateOne(
                {"user_id": pred["user_id"], "movieId": pred["movieId"]},
                {"$set": pred},
                upsert=True
            ))
        if operations:
            self.predictions.bulk_write(operations, ordered=False)

    # ─── Model Metrics Operations ────────────────────────────────

    def save_model_metrics(self, metrics: Dict):
        """Save model evaluation metrics to MongoDB."""
        metrics["evaluation_date"] = datetime.now(timezone.utc).isoformat()
        self.model_metrics.insert_one(metrics)
        print("✓ Model metrics saved to MongoDB")

    def get_latest_metrics(self, model_name: str = "DecisionTreeClassifier") -> Optional[Dict]:
        """Retrieve the most recent model evaluation metrics."""
        return self.model_metrics.find_one(
            {"model_name": model_name},
            {"_id": 0},
            sort=[("evaluation_date", DESCENDING)]
        )

    def get_all_metrics(self) -> List[Dict]:
        """Retrieve all stored model metrics."""
        return list(self.model_metrics.find({}, {"_id": 0}).sort("evaluation_date", DESCENDING))

    # ─── Database Statistics ─────────────────────────────────────

    def get_database_stats(self) -> Dict:
        """Get overview statistics for the database."""
        return {
            "movies": self.get_movie_count(),
            "ratings": self.get_rating_count(),
            "users": self.get_user_count(),
            "predictions": self.get_prediction_count(),
            "database": self.db_name,
            "connection": self.uri
        }

    def is_connected(self) -> bool:
        """Check if the database connection is active."""
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    def drop_database(self):
        """Drop the entire database. Use with caution — for testing only."""
        self.client.drop_database(self.db_name)
        print(f"✗ Database '{self.db_name}' dropped")


# Module-level convenience function
_db_instance = None

def get_db() -> CineScopeDB:
    """Get or create a singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = CineScopeDB()
    return _db_instance
