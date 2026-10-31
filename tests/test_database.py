"""
Unit and Integration Tests for CineScope MongoDB Database Operations
"""
import pytest
from database import CineScopeDB


@pytest.fixture(scope="module")
def db():
    """Database fixture for testing."""
    test_db = CineScopeDB(db_name="cinescope_test_db")
    test_db.create_indexes()
    yield test_db
    # Cleanup after tests
    test_db.drop_database()
    test_db.close()


def test_db_connection(db):
    """Test that connection to MongoDB is alive."""
    assert db.is_connected() is True


def test_insert_and_get_movie(db):
    """Test inserting and retrieving a movie."""
    movie_doc = {
        "movieId": 99901,
        "title": "Test Movie 1 (2024)",
        "genres": ["Action", "Sci-Fi"],
        "year": 2024,
        "average_rating": 4.5,
        "rating_count": 10
    }
    inserted = db.bulk_insert_movies([movie_doc])
    assert inserted == 1

    fetched = db.get_movie_by_id(99901)
    assert fetched is not None
    assert fetched["title"] == "Test Movie 1 (2024)"
    assert "Action" in fetched["genres"]


def test_movie_idempotent_insert(db):
    """Test that duplicate movieIds are safely rejected without crashing."""
    duplicate_doc = {
        "movieId": 99901,
        "title": "Duplicate Movie",
        "genres": ["Drama"],
        "average_rating": 3.0,
        "rating_count": 1
    }
    inserted = db.bulk_insert_movies([duplicate_doc])
    assert inserted == 0


def test_search_and_genre_filtering(db):
    """Test title search and genre filtering."""
    movies = [
        {"movieId": 99902, "title": "Inception Test", "genres": ["Action", "Sci-Fi"], "average_rating": 4.8, "rating_count": 100},
        {"movieId": 99903, "title": "Interstellar Test", "genres": ["Adventure", "Sci-Fi"], "average_rating": 4.7, "rating_count": 80},
        {"movieId": 99904, "title": "Comedy Club Test", "genres": ["Comedy"], "average_rating": 3.5, "rating_count": 20}
    ]
    db.bulk_insert_movies(movies)

    # Search by title
    search_res = db.search_movies_by_title("inception")
    assert any(m["movieId"] == 99902 for m in search_res)

    # Filter by genre
    sci_fi = db.get_movies_by_genre("Sci-Fi")
    sci_fi_ids = {m["movieId"] for m in sci_fi}
    assert 99902 in sci_fi_ids
    assert 99903 in sci_fi_ids
    assert 99904 not in sci_fi_ids


def test_query_movies_genre_and_sorting(db):
    """Test unified query_movies method with genre filtering and Title A-Z / Popularity sorting."""
    # Test Genre + Title A-Z
    res_a_z = db.query_movies(genre="Sci-Fi", sort_by="Title A-Z")
    titles = [m["title"] for m in res_a_z if m["movieId"] in [99902, 99903]]
    assert titles == ["Inception Test", "Interstellar Test"]

    # Test Genre + Popularity
    res_pop = db.query_movies(genre="Sci-Fi", sort_by="Popularity")
    m_ids = [m["movieId"] for m in res_pop if m["movieId"] in [99902, 99903]]
    assert m_ids[0] == 99902  # Inception (100 ratings) before Interstellar (80 ratings)

    # Test Count
    count_scifi = db.count_queried_movies(genre="Sci-Fi")
    assert count_scifi >= 2


def test_rating_upsert_and_compound_unique_index(db):
    """Test that ratings can be inserted, updated, and enforce uniqueness per user-movie."""
    # First rating
    res1 = db.upsert_rating(user_id="test_user_1", movie_id=99902, rating=4.0)
    assert res1 in ["inserted", "updated"]

    user_rating = db.get_user_rating_for_movie("test_user_1", 99902)
    assert user_rating is not None
    assert user_rating["rating"] == 4.0

    # Update rating (should update existing document, not insert new one)
    res2 = db.upsert_rating(user_id="test_user_1", movie_id=99902, rating=5.0)
    assert res2 == "updated"

    updated_rating = db.get_user_rating_for_movie("test_user_1", 99902)
    assert updated_rating["rating"] == 5.0

    # Check that movie average rating was updated
    stats = db.get_movie_rating_stats(99902)
    assert stats["average_rating"] == 5.0
    assert stats["rating_count"] == 1


def test_user_profile_crud(db):
    """Test user profile creation and retrieval."""
    created = db.create_user("test_profile", "Test Profile", ["Action", "Drama"])
    assert created is True

    # Duplicate creation returns False
    created_dup = db.create_user("test_profile", "Duplicate", [])
    assert created_dup is False

    user = db.get_user("test_profile")
    assert user["display_name"] == "Test Profile"
    assert "Action" in user["preferred_genres"]


def test_model_metrics_storage(db):
    """Test saving and retrieving model metrics."""
    metrics_data = {
        "model_name": "DecisionTreeClassifier",
        "accuracy": 0.7008,
        "precision": 0.6838,
        "recall": 0.7046,
        "f1_score": 0.6941
    }
    db.save_model_metrics(metrics_data)

    latest = db.get_latest_metrics("DecisionTreeClassifier")
    assert latest is not None
    assert latest["accuracy"] == 0.7008
    assert latest["f1_score"] == 0.6941
