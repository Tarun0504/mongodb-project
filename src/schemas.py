"""
CineScope Database Schemas
Defines the expected structure for MongoDB documents and CSV validation.
"""

# Expected CSV columns for validation
MOVIES_CSV_COLUMNS = ["movieId", "title", "genres"]
RATINGS_CSV_COLUMNS = ["userId", "movieId", "rating", "timestamp"]
TAGS_CSV_COLUMNS = ["userId", "movieId", "tag", "timestamp"]
LINKS_CSV_COLUMNS = ["movieId", "imdbId", "tmdbId"]

# Valid rating range
MIN_RATING = 0.5
MAX_RATING = 5.0
RATING_STEP = 0.5

# Genre that indicates no genre
NO_GENRES_LABEL = "(no genres listed)"

# All possible MovieLens genres
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "IMAX", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]
