"""
CineScope Recommendation Engine
Implements a hybrid recommendation workflow combining:
A) Content-based genre similarity (cosine similarity)
B) ML-predicted preference (Decision Tree Classifier)
C) New-user popularity fallback
D) Existing-user personalization
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, TOP_N_RECOMMENDATIONS, POSITIVE_RATING_THRESHOLD
from src.schemas import ALL_GENRES


class RecommendationEngine:
    """
    Hybrid recommendation engine for CineScope.
    
    Components:
    A) Content-based: Uses cosine similarity on genre vectors
    B) ML Prediction: Uses trained DecisionTreeClassifier 
    C) Popularity fallback: For new users without rating history
    D) Personalization: Incorporates user's own ratings
    """

    def __init__(self, db=None):
        """
        Initialize the recommendation engine.
        
        Args:
            db: CineScopeDB instance for database access
        """
        self.db = db
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self._genre_matrix = None
        self._movie_ids = None
        self._movie_data = None
        self._load_model()

    def _load_model(self):
        """Load the trained model and preprocessor from disk."""
        model_path = os.path.join(MODELS_DIR, "decision_tree_model.joblib")
        preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        features_path = os.path.join(MODELS_DIR, "feature_names.json")

        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
            if os.path.exists(features_path):
                with open(features_path, "r") as f:
                    self.feature_names = json.load(f)
        except Exception as e:
            print(f"  ⚠ Could not load model artifacts: {e}")

    def _build_genre_matrix(self, movies: list):
        """Build a genre binary matrix for all movies."""
        self._movie_data = {m["movieId"]: m for m in movies}
        self._movie_ids = [m["movieId"] for m in movies]
        
        # Build binary genre vectors
        genre_vectors = []
        for movie in movies:
            genres = movie.get("genres", [])
            vector = [1 if g in genres else 0 for g in ALL_GENRES]
            genre_vectors.append(vector)

        self._genre_matrix = np.array(genre_vectors, dtype=np.float64)

    def get_similar_movies(self, movie_id: int, movies: list = None,
                           top_n: int = TOP_N_RECOMMENDATIONS,
                           genre_filter: str = None) -> list:
        """
        Component A: Content-based genre similarity.
        
        Finds movies similar to the selected movie based on genre overlap
        using cosine similarity on binary genre vectors.
        
        Args:
            movie_id: The reference movie ID
            movies: List of movie documents (from MongoDB)
            top_n: Number of recommendations to return
            genre_filter: Optional genre to filter results
            
        Returns:
            List of similar movie dicts with similarity scores
        """
        if movies is None and self.db:
            movies = self.db.get_all_movies()
        
        if not movies:
            return []

        # Build genre matrix if needed
        if self._genre_matrix is None or self._movie_ids is None:
            self._build_genre_matrix(movies)

        # Find the index of the target movie
        if movie_id not in self._movie_ids:
            return []
        
        target_idx = self._movie_ids.index(movie_id)
        target_vector = self._genre_matrix[target_idx].reshape(1, -1)

        # Compute cosine similarity between target and all movies
        similarities = cosine_similarity(target_vector, self._genre_matrix)[0]

        # Create results, excluding the target movie
        results = []
        for idx, sim_score in enumerate(similarities):
            mid = self._movie_ids[idx]
            if mid == movie_id:
                continue  # Exclude the selected movie from its own recommendations
            
            movie = self._movie_data[mid]

            # Apply genre filter if specified
            if genre_filter and genre_filter not in movie.get("genres", []):
                continue

            results.append({
                **movie,
                "similarity_score": round(float(sim_score), 4),
                "recommendation_type": "content_similarity"
            })

        # Sort by similarity (descending), then by rating_count as tiebreaker
        results.sort(key=lambda x: (
            -x["similarity_score"],
            -x.get("rating_count", 0)
        ))

        # Remove duplicates (by movieId)
        seen = set()
        unique_results = []
        for r in results:
            if r["movieId"] not in seen:
                seen.add(r["movieId"])
                unique_results.append(r)

        return unique_results[:top_n]

    def predict_user_preference(self, user_id, movie_ids: list,
                                 user_ratings: list = None) -> list:
        """
        Component B: ML-predicted preference.
        
        Uses the trained DecisionTreeClassifier to predict whether
        the user is likely to like each candidate movie.
        
        Args:
            user_id: User identifier
            movie_ids: List of candidate movie IDs
            user_ratings: User's rating history (list of dicts)
            
        Returns:
            List of predictions with probabilities
        """
        if self.model is None or self.preprocessor is None:
            return []

        if user_ratings is None and self.db:
            user_ratings = self.db.get_user_ratings(user_id)

        # Compute user statistics from their ratings
        if user_ratings and len(user_ratings) > 0:
            ratings_values = [r["rating"] for r in user_ratings]
            user_avg = np.mean(ratings_values)
            user_count = len(ratings_values)
            user_log_count = np.log1p(user_count)
            has_history = True
        else:
            # Fallback for new users
            user_avg = self.preprocessor.global_avg_rating or 3.5
            user_count = 0
            user_log_count = 0.0
            has_history = False

        if not movie_ids or self.feature_names is None:
            return []

        # Vectorized batch feature extraction
        feature_dicts = []
        for mid in movie_ids:
            # Get movie stats from preprocessor's training aggregates
            movie_row = None
            if self.preprocessor.movie_stats is not None:
                mask = self.preprocessor.movie_stats["movieId"] == mid
                if mask.any():
                    movie_row = self.preprocessor.movie_stats[mask].iloc[0]

            if movie_row is not None:
                movie_avg = float(movie_row["movie_avg_rating"])
                movie_count = int(movie_row["movie_rating_count"])
                movie_log_count = float(movie_row["movie_log_count"])
            else:
                # Unseen movie fallback
                movie_avg = self.preprocessor.global_avg_rating or 3.5
                movie_count = 0
                movie_log_count = 0.0

            # Get genre features
            movie_data = self._movie_data.get(mid, {}) if self._movie_data else {}
            if not movie_data and self.db:
                movie_data = self.db.get_movie_by_id(mid) or {}
            genres = movie_data.get("genres", [])

            feature_dict = {
                "user_avg_rating": user_avg,
                "user_log_count": user_log_count,
                "user_rating_count": user_count,
                "movie_avg_rating": movie_avg,
                "movie_log_count": movie_log_count,
                "movie_rating_count": movie_count,
            }

            for genre in ALL_GENRES:
                col_name = f"genre_{genre.lower().replace('-', '_')}"
                feature_dict[col_name] = 1 if genre in genres else 0

            feature_dicts.append(feature_dict)

        # Batch predict with DataFrame to preserve feature names
        X_df = pd.DataFrame(feature_dicts)[self.feature_names]
        pred_classes = self.model.predict(X_df)
        pred_probs = self.model.predict_proba(X_df)[:, 1] if hasattr(self.model, "predict_proba") else [None] * len(movie_ids)

        predictions = []
        for i, mid in enumerate(movie_ids):
            p_class = int(pred_classes[i])
            p_prob = float(pred_probs[i]) if pred_probs[i] is not None else None
            predictions.append({
                "movieId": mid,
                "predicted_class": p_class,
                "predicted_probability": round(p_prob, 4) if p_prob is not None else None,
                "predicted_label": "Likely to like" if p_class == 1 else "Unlikely to like",
                "has_user_history": has_history,
                "recommendation_type": "ml_prediction"
            })

        # Sort by predicted probability (descending)
        predictions.sort(key=lambda x: -(x["predicted_probability"] or 0))

        return predictions

    def get_genre_recommendations(self, genres: list, movies: list = None,
                                   top_n: int = TOP_N_RECOMMENDATIONS,
                                   min_ratings: int = 5) -> list:
        """
        Component C: Genre-based popularity fallback for new users.
        
        Recommends popular, well-rated movies within the selected genres.
        Clearly labelled as genre-based/popularity-based recommendations.
        
        Args:
            genres: List of preferred genres
            movies: List of movie documents
            top_n: Number of recommendations
            min_ratings: Minimum number of ratings required
            
        Returns:
            List of recommended movies
        """
        if movies is None and self.db:
            movies = self.db.get_all_movies()

        if not movies:
            return []

        # Filter movies that match any of the preferred genres
        matching = []
        for movie in movies:
            movie_genres = movie.get("genres", [])
            if not movie_genres:
                continue
            overlap = set(genres) & set(movie_genres)
            if overlap and movie.get("rating_count", 0) >= min_ratings:
                matching.append({
                    **movie,
                    "genre_overlap": len(overlap),
                    "recommendation_type": "genre_popularity",
                    "recommendation_reason": f"Popular in: {', '.join(sorted(overlap))}"
                })

        # Sort by genre overlap, then average rating, then rating count
        matching.sort(key=lambda x: (
            -x["genre_overlap"],
            -x.get("average_rating", 0),
            -x.get("rating_count", 0)
        ))

        return matching[:top_n]

    def get_personalized_recommendations(self, user_id: str,
                                          movies: list = None,
                                          top_n: int = TOP_N_RECOMMENDATIONS) -> list:
        """
        Component D: Personalized recommendations for existing users.
        
        Combines user's genre preferences from their ratings with
        ML predictions to create a personalized recommendation list.
        
        Args:
            user_id: User identifier
            movies: List of movie documents
            top_n: Number of recommendations
            
        Returns:
            List of recommended movies with explanations
        """
        if movies is None and self.db:
            movies = self.db.get_all_movies()

        if not movies or not self.db:
            return []

        # Get user's ratings
        user_ratings = self.db.get_user_ratings(user_id)
        rated_movie_ids = {r["movieId"] for r in user_ratings}

        if not user_ratings:
            # No ratings — check if user has preferred genres
            user = self.db.get_user(user_id)
            if user and user.get("preferred_genres"):
                genre_recs = self.get_genre_recommendations(
                    user["preferred_genres"], movies, top_n
                )
                for rec in genre_recs:
                    rec["combined_score"] = round(rec.get("average_rating", 3.0) / 5.0, 4)
                    rec["recommendation_reasons"] = [
                        rec.get("recommendation_reason", "Based on your preferred genres"),
                        f"Average rating: {rec.get('average_rating', 0):.1f}/5.0"
                    ]
                return genre_recs
            return []

        # Determine user's preferred genres from their highly rated movies
        liked_genres = {}
        for r in user_ratings:
            if r["rating"] >= POSITIVE_RATING_THRESHOLD:
                movie = self.db.get_movie_by_id(r["movieId"])
                if movie:
                    for g in movie.get("genres", []):
                        liked_genres[g] = liked_genres.get(g, 0) + 1

        # Get candidate movies (not already rated)
        candidates = [m for m in movies if m["movieId"] not in rated_movie_ids]

        if not candidates:
            return []

        # Score candidates using ML prediction
        candidate_ids = [m["movieId"] for m in candidates[:500]]  # Limit for performance
        predictions = self.predict_user_preference(user_id, candidate_ids, user_ratings)

        # Build recommendation list with explanations
        pred_map = {p["movieId"]: p for p in predictions}
        results = []

        for movie in candidates[:500]:
            mid = movie["movieId"]
            pred = pred_map.get(mid, {})
            movie_genres = set(movie.get("genres", []))
            genre_overlap = movie_genres & set(liked_genres.keys())

            # Combine signals
            ml_score = pred.get("predicted_probability", 0.5) or 0.5
            genre_score = len(genre_overlap) / max(len(liked_genres), 1)
            combined_score = 0.6 * ml_score + 0.4 * genre_score

            # Build explanation
            reasons = []
            if pred.get("predicted_class") == 1:
                reasons.append(f"ML model predicts you'll like this ({ml_score:.0%} confidence)")
            if genre_overlap:
                reasons.append(f"Matches your preferred genres: {', '.join(sorted(genre_overlap))}")
            if movie.get("average_rating", 0) >= 4.0:
                reasons.append(f"Highly rated ({movie['average_rating']:.1f}/5)")

            if not reasons:
                reasons.append("Based on overall popularity")

            results.append({
                **movie,
                "combined_score": round(combined_score, 4),
                "ml_probability": pred.get("predicted_probability"),
                "predicted_label": pred.get("predicted_label", "Unknown"),
                "genre_match_count": len(genre_overlap),
                "recommendation_type": "personalized",
                "recommendation_reasons": reasons
            })

        # Sort by combined score
        results.sort(key=lambda x: -x["combined_score"])

        return results[:top_n]

    def get_hybrid_recommendations(self, movie_id: int = None,
                                    user_id: str = None,
                                    genres: list = None,
                                    movies: list = None,
                                    top_n: int = TOP_N_RECOMMENDATIONS) -> dict:
        """
        Master recommendation method that combines all components.
        
        Returns a dict with categorized recommendations:
        - similar: Content-based similar movies
        - predicted: ML-predicted preferences
        - popular: Genre-based popular movies
        - personalized: Combined personalized picks
        """
        if movies is None and self.db:
            movies = self.db.get_all_movies()

        results = {
            "similar": [],
            "predicted": [],
            "popular": [],
            "personalized": []
        }

        # Content-based similarity (if a movie is selected)
        if movie_id and movies:
            results["similar"] = self.get_similar_movies(movie_id, movies, top_n)

        # Genre-based popularity
        if genres:
            results["popular"] = self.get_genre_recommendations(genres, movies, top_n)

        # ML predictions and personalization (if user exists)
        if user_id and self.db:
            user_ratings = self.db.get_user_ratings(user_id)
            if user_ratings:
                results["personalized"] = self.get_personalized_recommendations(
                    user_id, movies, top_n
                )
            elif genres:
                results["personalized"] = self.get_genre_recommendations(genres, movies, top_n)

        return results
