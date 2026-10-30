# CineScope — Movie Recommendation & Rating Prediction System

> **An End-to-End Academic Implementation Using MongoDB, PyMongo, Scikit-Learn, and Streamlit**

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/MongoDB-7.0%20Community-green.svg)](https://www.mongodb.com/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-Academic%20Use-lightgrey.svg)]()

---

## 📌 Executive Summary

**CineScope** is a comprehensive, production-ready movie discovery and recommendation platform designed for academic demonstration and viva defense. It pairs **MongoDB Community Server 7.0** and **PyMongo** with **Scikit-learn** and a cinematic dark-themed **Streamlit** user interface.

Built upon the official **MovieLens Latest Small** dataset (100,836 ratings, 9,742 movies, 610 users), CineScope features a transparent hybrid recommendation engine combining:
1. **Content-Based Cosine Similarity:** Multi-hot encoded genre vectors to discover thematically similar titles.
2. **Supervised Rating Prediction:** A constrained `DecisionTreeClassifier` predicting user preference ($\text{rating} \ge 4.0$).
3. **Cold-Start Fallbacks:** Popularity and genre-based onboarding for new profiles.
4. **Secondary Continuous Regressor:** A companion `DecisionTreeRegressor` evaluating continuous star predictions ($R^2 = 0.2632$, $\text{MAE} = 0.6787$).

---

## 🏗️ System Architecture

```
+-------------------------------------------------------------+
|                  Presentation Layer                         |
|            Streamlit (Dark Cinematic UI)                    |
|           Plotly Express & Graph Objects                    |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                  Application & Logic Layer                  |
|    src/recommendation_engine.py  |  src/train_model.py      |
|    src/preprocessing.py          |  src/dataset_loader.py   |
|            Scikit-learn, Pandas, NumPy, Joblib              |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                  Database Access Layer                      |
|                  database.py (CineScopeDB)                  |
|                 PyMongo 4.6+ Native Driver                  |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                  Data Persistence Layer                     |
|           MongoDB Community Server 7.0 (Local)              |
|           Database: cinescope_db (5 Collections)            |
+-------------------------------------------------------------+
```

---

## 📊 Empirical Machine Learning Results

All models were evaluated strictly against the **held-out test partition of 20,168 records (20%)** using aggregates fitted strictly on training data (zero target leakage).

### Primary Supervised Classifier (`DecisionTreeClassifier`)
- **Accuracy:** **70.08%**
- **Precision (Liked $\ge 4.0$):** **68.38%**
- **Recall (Liked $\ge 4.0$):** **70.46%**
- **F1-Score (Liked $\ge 4.0$):** **69.41%**
- **Macro F1-Score:** **70.06%**

### Confusion Matrix ($N = 20,168$)
```
                 Predicted Not Liked    Predicted Liked
Actual Not Liked        7,287 (TN)          3,165 (FP)
Actual Liked            2,870 (FN)          6,846 (TP)
```

### Secondary Regressor (`DecisionTreeRegressor`)
- **Mean Absolute Error (MAE):** **0.6787 stars**
- **Root Mean Squared Error (RMSE):** **0.8935 stars**
- **$R^2$ Score:** **0.2632**

---

## 🗄️ MongoDB Database Design (`cinescope_db`)

| Collection | Key Attributes | Indexes | Description |
|---|---|---|---|
| `movies` | `movieId`, `title`, `genres`, `year`, `average_rating`, `rating_count`, `external_ids` | `movieId` (Unique), `title`, `genres`, `average_rating`, `rating_count` | 9,742 movie documents with embedded rating statistics |
| `ratings` | `userId`, `movieId`, `rating`, `timestamp`, `rated_at`, `source` | `(userId, movieId)` (Compound Unique), `movieId`, `userId` | 100,836 rating records with idempotent upsert support |
| `users` | `user_id`, `display_name`, `preferred_genres`, `created_at` | `user_id` (Unique) | Demo user profiles for cold-start onboarding |
| `predictions` | `user_id`, `movieId`, `predicted_class`, `predicted_probability`, `model_version` | `(user_id, movieId)` | Cached ML inference records |
| `model_metrics` | `model_name`, `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix` | `(model_name, evaluation_date)` | Model evaluation history stored in database |

---

## 🚀 macOS Setup & Execution Instructions

### 1. Prerequisites
- **Python 3.11+** installed (verified compatible with Python 3.11 and 3.13)
- **MongoDB Community Server 7.0** installed via Homebrew:
  ```bash
  brew tap mongodb/brew
  brew install mongodb-community
  ```

### 2. Start MongoDB Server
```bash
# Option A: Homebrew service (recommended)
brew services start mongodb-community

# Option B: Run directly in background
mkdir -p /tmp/mongodb-data
mongod --dbpath /tmp/mongodb-data &
```

### 3. Clone / Navigate to Project Directory
```bash
cd /Users/leg/Desktop/mongodb-project
```

### 4. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

### 6. Environment Configuration
Ensure `.env` exists (copied from `.env.example`):
```bash
cp .env.example .env
```

### 7. Run Automated Data Pipeline
Execute the scripts in order:

```bash
# A. Download MovieLens Latest Small dataset (if not already downloaded)
python scripts/download_dataset.py

# B. Import dataset into MongoDB (Idempotent: safe to run multiple times)
python scripts/import_to_mongodb.py

# C. Train Decision Tree Classifier and generate evaluation artifacts
python scripts/train_and_evaluate.py

# D. Run secondary continuous regression experiment (MAE & R²)
python scripts/evaluate_regressor.py

# E. Export evaluation metrics to reports/ directory
python scripts/export_metrics.py
```

### 8. Run Automated Test Suite
```bash
pytest tests/ -v
```
*(All 16 unit and integration tests will execute and pass)*

### 9. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 🖥️ Streamlit Application Pages

1. **🏠 Discover:** Dynamic catalog exploration, title search with regex, genre filtering, and popularity rankings.
2. **🎬 Movie Details:** Detailed metadata, release year extraction, external IMDb links, interactive rating submission slider, and content-based similar movies.
3. **✨ Recommendations:** Three view modes:
   - *Personalized Profile Picks:* Confidence rationales combining ML probability and genre affinity.
   - *Movie-to-Movie Similarity:* Cosine similarity over 19 genre vectors.
   - *Cold-Start Discovery:* Top-rated and popular picks for new users.
4. **⭐ Rate Movies:** Searchable movie rating interface with instant PyMongo upsert and history logs.
5. **📊 Model Analytics:** Interactive Plotly confusion matrix, metric comparisons, feature importance bar charts, and experiment metadata read live from MongoDB.
6. **💾 Database & Project Info:** Real-time collection counters, index schemas, and academic citations.

---

## 📂 Project Directory Structure

```
mongodb-project/
├── app.py                      # Main Streamlit web application
├── database.py                 # CineScopeDB PyMongo manager class
├── config.py                   # Centralized configuration and environment loader
├── requirements.txt            # Python dependencies
├── .env.example                # Template environment variables
├── .env                        # Local environment variables
├── .gitignore                  # Git exclusions
├── README.md                   # Project documentation and quickstart
├── data/                       # MovieLens dataset CSV files & ZIP
│   └── ml-latest-small/
│       ├── movies.csv
│       ├── ratings.csv
│       ├── tags.csv
│       └── links.csv
├── models/                     # Serialized machine learning models
│   ├── decision_tree_model.joblib
│   ├── preprocessor.joblib
│   └── feature_names.json
├── reports/                    # Evaluation artifacts and charts
│   ├── confusion_matrix.png
│   ├── metrics_chart.png
│   ├── feature_importance.png
│   ├── model_metrics.json
│   ├── preprocessing_stats.json
│   └── regression_metrics.json
├── src/                        # Core application modules
│   ├── __init__.py
│   ├── schemas.py              # CSV and document schemas
│   ├── dataset_loader.py       # Dataset loading and validation
│   ├── preprocessing.py        # Data hygiene and feature engineering
│   ├── train_model.py          # Decision Tree training pipeline
│   ├── evaluate_model.py       # Evaluation metrics and plotting
│   └── recommendation_engine.py# Hybrid recommendation engine
├── scripts/                    # Command-line automation scripts
│   ├── download_dataset.py     # Dataset downloader
│   ├── import_to_mongodb.py    # Idempotent bulk MongoDB importer
│   ├── train_and_evaluate.py   # Training script
│   ├── evaluate_regressor.py   # Continuous regression evaluation
│   ├── export_metrics.py       # Metrics exporter
│   └── generate_docx_report.py # Academic Word report generator
├── tests/                      # Automated Pytest suite
│   ├── conftest.py
│   ├── test_database.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_recommendations.py
└── docs/                       # Academic project documentation
    ├── project_report.md       # Complete project report in Markdown
    └── project_report.docx     # Academic report in Word DOCX format
```

---

## 🎓 Academic Viva Voce Quick Reference

| Question | Core Model Answer |
|---|---|
| **Why MongoDB instead of MySQL?** | Media metadata has variable-length arrays (genres) and optional nested objects. In relational databases, this requires junction tables and costly joins. MongoDB's BSON format stores arrays natively and supports multi-key indexing with zero joins. |
| **How did you prevent target leakage?** | The dataset was split into train (80%) and test (20%) partitions *before* computing summary statistics. Aggregates (user mean, movie mean, counts) were calculated strictly on the training partition. |
| **Why cosine similarity + Decision Tree?** | Cosine similarity evaluates taxonomic item resemblance (independent of user taste). The Decision Tree is a supervised classifier that evaluates whether an individual with specific historical preferences will enjoy a candidate film. |
| **Why are $R^2$ and MAE not classification metrics?** | $R^2$ and MAE quantify continuous error residuals. Our primary model is a binary classifier ($\ge 4.0$). We ran an explicit secondary regression experiment to report MAE (0.6787) and $R^2$ (0.2632) for continuous ratings. |
| **How does the compound index work?** | In `ratings`, a unique compound index on `(userId, movieId)` prevents duplicate ratings at the database engine level and supports atomic upserts via `update_one(..., upsert=True)`. |

---

## 📜 Dataset Attribution & Citations

- **Dataset:** MovieLens Latest Small, GroupLens Research Project, University of Minnesota
- **Citation:** F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.
- **License:** Granted freely for academic and educational non-commercial research use.
# mongodb-project
