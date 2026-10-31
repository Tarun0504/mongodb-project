"""
CineScope Configuration Module
Loads environment variables and provides project-wide constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "cinescope_db")

# Dataset Configuration
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# ML Configuration
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
POSITIVE_RATING_THRESHOLD = float(os.getenv("POSITIVE_RATING_THRESHOLD", "4.0"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

# Application Configuration
APP_NAME = "CineScope"
APP_TAGLINE = "Movie Recommendation & Rating Prediction System"
TOP_N_RECOMMENDATIONS = 10

# Collection Names
MOVIES_COLLECTION = "movies"
RATINGS_COLLECTION = "ratings"
USERS_COLLECTION = "users"
PREDICTIONS_COLLECTION = "predictions"
MODEL_METRICS_COLLECTION = "model_metrics"
