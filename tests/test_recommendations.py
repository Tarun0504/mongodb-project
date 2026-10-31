"""
Unit Tests for Recommendation Engine Components
"""
import pytest
from src.recommendation_engine import RecommendationEngine


@pytest.fixture
def mock_movies():
    """Mock movies list for recommendation testing."""
    return [
        {"movieId": 1, "title": "Toy Story (1995)", "genres": ["Adventure", "Animation", "Children", "Comedy"], "average_rating": 3.92, "rating_count": 215},
        {"movieId": 2, "title": "Jumanji (1995)", "genres": ["Adventure", "Children", "Fantasy"], "average_rating": 3.43, "rating_count": 110},
        {"movieId": 3, "title": "Grumpier Old Men (1995)", "genres": ["Comedy", "Romance"], "average_rating": 3.25, "rating_count": 52},
        {"movieId": 4, "title": "Toy Story 2 (1999)", "genres": ["Adventure", "Animation", "Children", "Comedy"], "average_rating": 3.87, "rating_count": 107},
        {"movieId": 5, "title": "Heat (1995)", "genres": ["Action", "Crime", "Thriller"], "average_rating": 3.94, "rating_count": 102},
        {"movieId": 6, "title": "Casino (1995)", "genres": ["Crime", "Drama"], "average_rating": 3.92, "rating_count": 84},
    ]


def test_content_similarity_excludes_self(mock_movies):
    """Test that content similarity excludes the reference movie and ranks high-overlap movies first."""
    engine = RecommendationEngine(db=None)
    similar = engine.get_similar_movies(movie_id=1, movies=mock_movies, top_n=3)

    assert len(similar) > 0
    # Selected movie must NOT appear in its own recommendations
    rec_ids = [m["movieId"] for m in similar]
    assert 1 not in rec_ids
    # Toy Story 2 (movieId 4) shares identical genres and should be top ranked
    assert similar[0]["movieId"] == 4
    assert similar[0]["similarity_score"] == pytest.approx(1.0, 0.001)


def test_genre_recommendations_fallback(mock_movies):
    """Test genre-based popularity fallback for new users."""
    engine = RecommendationEngine(db=None)
    recs = engine.get_genre_recommendations(genres=["Animation", "Children"], movies=mock_movies, top_n=2, min_ratings=1)

    assert len(recs) == 2
    rec_ids = {m["movieId"] for m in recs}
    assert 1 in rec_ids or 4 in rec_ids
    assert recs[0]["recommendation_type"] == "genre_popularity"


def test_genre_filter_in_similarity(mock_movies):
    """Test filtering similar movies by a specific genre."""
    engine = RecommendationEngine(db=None)
    similar = engine.get_similar_movies(movie_id=1, movies=mock_movies, top_n=5, genre_filter="Fantasy")

    # Only Jumanji has Fantasy among the mock movies
    assert len(similar) == 1
    assert similar[0]["movieId"] == 2
