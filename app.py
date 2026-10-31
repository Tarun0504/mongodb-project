"""
CineScope — Movie Recommendation and Rating Prediction System
Streamlit Application
"""
import os
import sys
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    APP_NAME, APP_TAGLINE, REPORTS_DIR, MODELS_DIR,
    POSITIVE_RATING_THRESHOLD
)
from database import CineScopeDB
from src.recommendation_engine import RecommendationEngine
from src.schemas import ALL_GENRES

# ─── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title=f"{APP_NAME} — Movie Discovery & Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Cinematic Dark Theme Styling ────────────────────────────────
st.markdown("""
<style>
    /* Dark cinematic theme */
    .stApp {
        background-color: #0d0f12;
        color: #e0e0e0;
    }
    
    /* Header branding */
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin-bottom: 0px;
    }
    .brand-accent {
        color: #e50914;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #8c8c8c;
        margin-bottom: 1.5rem;
    }
    
    /* Movie Card Styling */
    .movie-card {
        background-color: #161a20;
        border: 1px solid #242933;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .movie-card:hover {
        border-color: #3b4252;
    }
    .movie-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .movie-card-meta {
        font-size: 0.85rem;
        color: #9aa5b1;
        margin-bottom: 8px;
    }
    .genre-badge {
        display: inline-block;
        background-color: #202632;
        color: #cbd2d9;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        margin-right: 5px;
        margin-bottom: 5px;
        border: 1px solid #2e3846;
    }
    .rating-badge {
        display: inline-flex;
        align-items: center;
        background-color: #1a2218;
        color: #48bb78;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        border: 1px solid #22543d;
    }
    .rating-count {
        font-size: 0.8rem;
        color: #718096;
        margin-left: 6px;
    }
    
    /* Stat boxes */
    .stat-box {
        background-color: #161a20;
        border: 1px solid #242933;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #9aa5b1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Recommendation badges */
    .rec-badge-ml {
        background-color: #2b1f14;
        color: #ed8936;
        border: 1px solid #7b341e;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .rec-badge-similarity {
        background-color: #142129;
        color: #4299e1;
        border: 1px solid #2b4365;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .rec-badge-popularity {
        background-color: #1e1b2e;
        color: #9f7aea;
        border: 1px solid #44337a;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─── Database and Engine Singleton ───────────────────────────────
@st.cache_resource
def get_backend():
    """Initialize database and recommendation engine once."""
    try:
        db = CineScopeDB()
        engine = RecommendationEngine(db)
        return db, engine, None
    except Exception as e:
        return None, None, str(e)


db, engine, db_error = get_backend()

if db_error or not db or not db.is_connected():
    st.error("🚨 **Cannot connect to MongoDB**")
    st.markdown("""
    CineScope requires a running MongoDB instance.
    
    **To start MongoDB on macOS:**
    ```bash
    # Option 1: Using Homebrew services
    brew services start mongodb-community
    
    # Option 2: Running directly in background
    mongod --dbpath /tmp/mongodb-data &
    ```
    After starting MongoDB, click **Rerun** in the top right.
    """)
    st.stop()


# ─── Sidebar Navigation ──────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0;">
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 800; color: #fff;">
            Cine<span style="color: #e50914;">Scope</span> 🎬
        </h2>
        <p style="margin: 0; font-size: 0.8rem; color: #888;">Movie Recommendation & ML Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    nav_options = [
        "🏠 Discover",
        "🎬 Movie Details",
        "✨ Recommendations",
        "⭐ Rate Movies",
        "📊 Model Analytics",
        "💾 Database & Project Info"
    ]

    # Check if a page redirect was requested by an action button
    if "nav_override" in st.session_state and st.session_state["nav_override"] in nav_options:
        target_page = st.session_state.pop("nav_override")
        default_idx = nav_options.index(target_page)
    elif "nav_selection" in st.session_state and st.session_state["nav_selection"] in nav_options:
        default_idx = nav_options.index(st.session_state["nav_selection"])
    else:
        default_idx = 0

    page = st.radio(
        "Navigation",
        nav_options,
        index=default_idx,
        key="nav_radio"
    )
    st.session_state["nav_selection"] = page

    st.markdown("---")
    
    # User Profile Switcher
    st.markdown("##### 👤 Active Profile")
    users = db.get_all_users()
    user_options = {u["user_id"]: f"{u['display_name']} ({u['user_id']})" for u in users}
    
    if not user_options:
        user_options = {"guest": "Guest User"}
        
    selected_user_id = st.selectbox(
        "Select User",
        options=list(user_options.keys()),
        format_func=lambda uid: user_options[uid]
    )
    
    # Show active user details
    active_user = db.get_user(selected_user_id)
    if active_user:
        genres_str = ", ".join(active_user.get("preferred_genres", [])) or "None set"
        st.caption(f"**Preferred genres:** {genres_str}")
        user_ratings_count = len(db.get_user_ratings(selected_user_id))
        st.caption(f"**Ratings given:** {user_ratings_count}")

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.75rem; color: #666;'>"
        "CineScope v1.0<br>MongoDB 7.0 + PyMongo + Scikit-Learn"
        "</div>",
        unsafe_allow_html=True
    )


# ─── Helper Functions ────────────────────────────────────────────
def render_movie_card(movie, col_obj, show_action=True):
    """Render a clean cinematic movie card."""
    with col_obj:
        title = movie.get("title", "Unknown Title")
        year = movie.get("year")
        year_str = f"({year})" if year else ""
        avg_rating = movie.get("average_rating", 0.0)
        rating_count = movie.get("rating_count", 0)
        genres = movie.get("genres", [])
        movie_id = movie.get("movieId")

        genre_html = "".join([f'<span class="genre-badge">{g}</span>' for g in genres[:3]])
        if len(genres) > 3:
            genre_html += f'<span class="genre-badge">+{len(genres)-3}</span>'

        rating_color = "#48bb78" if avg_rating >= 4.0 else ("#ecc94b" if avg_rating >= 3.0 else "#cbd5e0")
        
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-card-title">{title}</div>
            <div class="movie-card-meta">
                <span class="rating-badge" style="color: {rating_color};">★ {avg_rating:.2f}</span>
                <span class="rating-count">({rating_count:,} ratings)</span>
            </div>
            <div>{genre_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if show_action:
            col_a, col_b = st.columns([1, 1])
            with col_a:
                if st.button("Details", key=f"det_{movie_id}", use_container_width=True):
                    st.session_state["selected_movie_id"] = movie_id
                    st.session_state["nav_override"] = "🎬 Movie Details"
                    st.rerun()
            with col_b:
                if st.button("Similar", key=f"sim_{movie_id}", use_container_width=True):
                    st.session_state["selected_movie_id"] = movie_id
                    st.session_state["rec_mode"] = "similar"
                    st.session_state["nav_override"] = "✨ Recommendations"
                    st.rerun()


# =================================================================
# PAGE 1: DISCOVER / HOME
# =================================================================
if page == "🏠 Discover":
    st.markdown('<div class="brand-title">Cine<span class="brand-accent">Scope</span> Discover</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Explore 9,700+ movies from the MovieLens dataset powered by MongoDB & PyMongo</div>', unsafe_allow_html=True)

    # Search and Filter Bar
    col1, col2, col3 = st.columns([3, 2, 1.3])
    with col1:
        search_query = st.text_input("🔍 Search Movies by Title", placeholder="e.g. Inception, Matrix, Toy Story...", key="search_input")
    with col2:
        selected_genre = st.selectbox("🎭 Filter by Genre", options=["All Genres"] + ALL_GENRES, key="genre_select")
    with col3:
        sort_by = st.selectbox("Sort By", ["Popularity", "Highest Rated", "Title A-Z", "Newest (Year)"], key="sort_select")

    st.markdown("<br>", unsafe_allow_html=True)

    # Dynamic query execution (combining search, genre filter, and sort_by)
    min_r = 5 if (sort_by == "Highest Rated" and not (search_query and search_query.strip())) else 0
    total_matching = db.count_queried_movies(
        query=search_query,
        genre=selected_genre,
        min_ratings=min_r
    )
    max_pages = max(1, (total_matching + 23) // 24)

    # Build descriptive title
    if selected_genre != "All Genres" and search_query.strip():
        header_title = f'"{search_query.strip()}" in {selected_genre} — Sorted by {sort_by}'
    elif selected_genre != "All Genres":
        header_title = f"{selected_genre} Movies — Sorted by {sort_by}"
    elif search_query.strip():
        header_title = f'Search Results for "{search_query.strip()}" — Sorted by {sort_by}'
    else:
        header_title = f"Movies Catalogue — Sorted by {sort_by}"

    col_h1, col_h2 = st.columns([4, 1.2])
    with col_h1:
        st.markdown(f"### {header_title}")
        st.caption(f"Showing up to 24 of {total_matching:,} movies found in MongoDB")
    with col_h2:
        page_num = st.number_input("Page", min_value=1, max_value=max_pages, value=1, step=1, key="page_selector")

    skip = (page_num - 1) * 24
    movies = db.query_movies(
        query=search_query,
        genre=selected_genre,
        sort_by=sort_by,
        min_ratings=min_r,
        limit=24,
        skip=skip
    )

    if not movies:
        st.info("No movies found matching your criteria. Try adjusting your search query or filters.")
    else:
        cols = st.columns(3)
        for i, movie in enumerate(movies):
            render_movie_card(movie, cols[i % 3])


# =================================================================
# PAGE 2: MOVIE DETAILS
# =================================================================
elif page == "🎬 Movie Details":
    st.markdown('<div class="brand-title">Movie <span class="brand-accent">Details</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Inspect metadata, rating statistics, and discover similar films</div>', unsafe_allow_html=True)

    # Movie selector
    all_movies = db.get_all_movies({"movieId": 1, "title": 1, "_id": 0})
    movie_dict = {m["movieId"]: m["title"] for m in all_movies}

    default_id = st.session_state.get("selected_movie_id", 1)
    if default_id not in movie_dict:
        default_id = 1

    selected_mid = st.selectbox(
        "Select or Search a Movie",
        options=list(movie_dict.keys()),
        index=list(movie_dict.keys()).index(default_id),
        format_func=lambda mid: f"{movie_dict[mid]} (ID: {mid})"
    )

    movie = db.get_movie_by_id(selected_mid)

    if movie:
        title = movie.get("title", "")
        year = movie.get("year", "N/A")
        genres = movie.get("genres", [])
        avg_rating = movie.get("average_rating", 0.0)
        rating_count = movie.get("rating_count", 0)
        external_ids = movie.get("external_ids", {})

        # Details Header
        st.markdown(f"## {title}")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Release Year", str(year) if year else "N/A")
        with col_m2:
            st.metric("Average Rating", f"★ {avg_rating:.2f} / 5.0")
        with col_m3:
            st.metric("Total Ratings", f"{rating_count:,}")
        with col_m4:
            imdb_link = f"https://www.imdb.com/title/tt{external_ids.get('imdbId'):07d}" if external_ids.get("imdbId") else None
            if imdb_link:
                st.markdown(f"**External Links**<br>[IMDb Page]({imdb_link})", unsafe_allow_html=True)
            else:
                st.metric("IMDb", "Not linked")

        st.markdown(f"**Genres:** " + " ".join([f"`{g}`" for g in genres]))

        # Rating interaction
        st.markdown("---")
        st.markdown("#### ⭐ Rate This Movie as Active User")
        col_r1, col_r2 = st.columns([2, 3])
        with col_r1:
            existing_user_rating = db.get_user_rating_for_movie(selected_user_id, selected_mid)
            default_val = float(existing_user_rating["rating"]) if existing_user_rating else 3.5
            
            new_rating = st.select_slider(
                "Your Rating",
                options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                value=default_val,
                key=f"slider_rate_{selected_mid}"
            )
            if st.button("Submit / Update Rating", key=f"btn_rate_{selected_mid}", type="primary"):
                action = db.upsert_rating(selected_user_id, selected_mid, new_rating, source="app_ui")
                st.success(f"✓ Rating successfully {action}! ({new_rating} ★ for '{title}')")
                st.rerun()

        with col_r2:
            if existing_user_rating:
                st.info(f"You previously rated this movie **{existing_user_rating['rating']} ★** (Source: {existing_user_rating.get('source', 'app')})")
            else:
                st.caption("You have not rated this movie yet.")

        # Similar movies section
        st.markdown("---")
        st.markdown("#### 🎯 Content-Based Similar Movies")
        st.caption("Computed using cosine similarity on binary genre vectors (excluding the selected movie):")
        
        similar_movies = engine.get_similar_movies(selected_mid, top_n=6)
        if similar_movies:
            cols = st.columns(3)
            for i, sim in enumerate(similar_movies):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-card-title">{sim['title']}</div>
                        <div class="movie-card-meta">
                            <span class="rec-badge-similarity">Similarity: {sim['similarity_score']:.1%}</span>
                            <span class="rating-badge" style="margin-left: 8px;">★ {sim.get('average_rating', 0):.2f}</span>
                        </div>
                        <div style="margin-top: 6px;">
                            {"".join([f'<span class="genre-badge">{g}</span>' for g in sim.get("genres", [])[:3]])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("View This Movie", key=f"view_sim_{sim['movieId']}", use_container_width=True):
                        st.session_state["selected_movie_id"] = sim["movieId"]
                        st.rerun()
        else:
            st.info("No similar movies found.")


# =================================================================
# PAGE 3: RECOMMENDATIONS
# =================================================================
elif page == "✨ Recommendations":
    st.markdown('<div class="brand-title">Hybrid <span class="brand-accent">Recommendations</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Transparent recommendation engine combining Content Similarity, ML Classifier, and Popularity Fallback</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "👤 Personalized for Active Profile",
        "🎬 Similar to Movie (Content-Based)",
        "🎭 Genre & Popularity Discovery"
    ])

    # ─── Tab 1: Personalized Profile Recommendations ───
    with tab1:
        st.markdown(f"### Recommendations for `{active_user.get('display_name', selected_user_id)}`")
        
        user_ratings = db.get_user_ratings(selected_user_id)
        
        if not user_ratings:
            st.warning(f"Profile **{selected_user_id}** has no ratings yet. Using **New-User Fallback**.")
            st.markdown("""
            **How New-User Fallback Works:**
            Since there is no interaction history to predict personalization, we recommend top-rated and popular titles matching the user's preferred genres.
            """)
            pref_genres = active_user.get("preferred_genres", [])
            if not pref_genres:
                pref_genres = st.multiselect("Select Preferred Genres", ALL_GENRES, default=["Action", "Sci-Fi"])
                if st.button("Save Preferences to Profile"):
                    db.update_user_genres(selected_user_id, pref_genres)
                    st.success("Preferences updated!")
                    st.rerun()

            recs = engine.get_genre_recommendations(pref_genres, top_n=10)
        else:
            st.success(f"Personalizing based on **{len(user_ratings)} ratings** in profile.")
            recs = engine.get_personalized_recommendations(selected_user_id, top_n=10)

        if recs:
            for i, r in enumerate(recs):
                with st.container():
                    col_r1, col_r2, col_r3 = st.columns([4, 2, 1])
                    with col_r1:
                        st.markdown(f"**{i+1}. {r.get('title', 'Unknown')}**")
                        genres_str = " • ".join(r.get("genres", []))
                        st.caption(genres_str)
                    with col_r2:
                        reasons = r.get("recommendation_reasons", [])
                        reason_text = " | ".join(reasons) if reasons else r.get("recommendation_reason", "Recommended for you")
                        st.markdown(f"<span class='rec-badge-ml'>{reason_text}</span>", unsafe_allow_html=True)
                    with col_r3:
                        avg_r = r.get("average_rating", 0)
                        st.markdown(f"★ **{avg_r:.2f}** ({r.get('rating_count', 0)} ratings)")
                    st.divider()

    # ─── Tab 2: Movie-to-Movie Content Similarity ───
    with tab2:
        st.markdown("### Find Movies Similar to a Selected Title")
        all_movies = db.get_all_movies({"movieId": 1, "title": 1, "_id": 0})
        m_opts = {m["movieId"]: m["title"] for m in all_movies}
        
        def_id = st.session_state.get("selected_movie_id", 1)
        sim_target_id = st.selectbox(
            "Choose Target Movie",
            options=list(m_opts.keys()),
            index=list(m_opts.keys()).index(def_id) if def_id in m_opts else 0,
            format_func=lambda x: m_opts[x],
            key="sim_tab_select"
        )
        
        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            genre_filter = st.selectbox("Optional Genre Filter", ["Any"] + ALL_GENRES)
            g_filter = None if genre_filter == "Any" else genre_filter
        with col_g2:
            top_k = st.slider("Number of Recommendations", 5, 20, 10)

        similar = engine.get_similar_movies(sim_target_id, top_n=top_k, genre_filter=g_filter)

        st.markdown(f"#### Top {len(similar)} Similar Films:")
        cols = st.columns(2)
        for i, s in enumerate(similar):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-card-title">{i+1}. {s['title']}</div>
                    <div class="movie-card-meta">
                        <span class="rec-badge-similarity">Similarity: {s['similarity_score']:.1%}</span>
                        <span class="rating-badge" style="margin-left: 8px;">★ {s.get('average_rating', 0):.2f}</span>
                        <span class="rating-count">({s.get('rating_count', 0)} ratings)</span>
                    </div>
                    <div>{"".join([f'<span class="genre-badge">{g}</span>' for g in s.get("genres", [])])}</div>
                </div>
                """, unsafe_allow_html=True)

    # ─── Tab 3: Genre & Popularity Discovery ───
    with tab3:
        st.markdown("### Discover by Genre Preference")
        sel_genres = st.multiselect("Pick one or more genres", ALL_GENRES, default=["Adventure", "Animation"])
        min_rat = st.slider("Minimum Ratings Count", 1, 50, 10)
        
        if sel_genres:
            genre_picks = engine.get_genre_recommendations(sel_genres, top_n=10, min_ratings=min_rat)
            for i, p in enumerate(genre_picks):
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-card-title">{i+1}. {p['title']}</div>
                    <div class="movie-card-meta">
                        <span class="rec-badge-popularity">{p.get('recommendation_reason', 'Genre match')}</span>
                        <span class="rating-badge" style="margin-left: 8px;">★ {p.get('average_rating', 0):.2f}</span>
                        <span class="rating-count">({p.get('rating_count', 0)} ratings)</span>
                    </div>
                    <div>{"".join([f'<span class="genre-badge">{g}</span>' for g in p.get("genres", [])])}</div>
                </div>
                """, unsafe_allow_html=True)


# =================================================================
# PAGE 4: RATE MOVIES
# =================================================================
elif page == "⭐ Rate Movies":
    st.markdown('<div class="brand-title">Rate <span class="brand-accent">Movies</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="brand-subtitle">Submit ratings for profile: <b>{active_user.get("display_name", selected_user_id)}</b></div>', unsafe_allow_html=True)

    st.markdown("#### Search and Rate")
    search_term = st.text_input("Find a movie to rate", placeholder="Type title name...")
    
    if search_term.strip():
        candidates = db.search_movies_by_title(search_term.strip(), limit=10)
    else:
        candidates = db.get_popular_movies(limit=8)

    st.caption("Showing matches from MongoDB:")
    
    for c in candidates:
        mid = c["movieId"]
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.markdown(f"**{c['title']}**")
                st.caption(f"Genres: {', '.join(c.get('genres', []))}")
            with col2:
                current_rate = db.get_user_rating_for_movie(selected_user_id, mid)
                val = float(current_rate["rating"]) if current_rate else 3.5
                rating_val = st.select_slider(
                    f"Rating for {c['title']}",
                    options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                    value=val,
                    key=f"slider_page_rate_{mid}",
                    label_visibility="collapsed"
                )
            with col3:
                btn_label = "Update Rating" if current_rate else "Submit Rating"
                if st.button(btn_label, key=f"btn_page_rate_{mid}", type="primary" if not current_rate else "secondary"):
                    res = db.upsert_rating(selected_user_id, mid, rating_val, source="app_rate_page")
                    st.success(f"✓ Rating {res}: {rating_val} ★ for '{c['title']}'!")
                    st.rerun()

            if current_rate:
                st.caption(f"Current stored rating: {current_rate['rating']} ★ (Recorded: {current_rate.get('rated_at', 'MovieLens dataset')})")
            st.divider()

    # User's recent ratings
    st.markdown("---")
    st.markdown("#### 📜 Your Rating History")
    my_ratings = db.get_user_ratings(selected_user_id)
    if my_ratings:
        st.write(f"Total ratings submitted by this profile: **{len(my_ratings)}**")
        history_data = []
        for r in my_ratings[:15]:
            m = db.get_movie_by_id(r["movieId"])
            history_data.append({
                "Movie Title": m.get("title", f"ID {r['movieId']}") if m else f"ID {r['movieId']}",
                "Your Rating": f"{r['rating']} ★",
                "Source": r.get("source", "app"),
                "Timestamp": datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M") if "timestamp" in r else "N/A"
            })
        st.dataframe(pd.DataFrame(history_data), use_container_width=True)
    else:
        st.info("You haven't submitted any ratings yet. Rate a movie above to start!")


# =================================================================
# PAGE 5: MODEL ANALYTICS
# =================================================================
elif page == "📊 Model Analytics":
    st.markdown('<div class="brand-title">Machine Learning <span class="brand-accent">Analytics</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Evaluation metrics for DecisionTreeClassifier loaded from MongoDB & disk</div>', unsafe_allow_html=True)

    metrics = db.get_latest_metrics("DecisionTreeClassifier")
    
    if not metrics:
        st.warning("No evaluation metrics found in MongoDB. Running evaluation script will populate this page.")
        if st.button("Train & Evaluate Model Now"):
            with st.spinner("Training model on MovieLens dataset..."):
                from src.dataset_loader import DatasetLoader
                from src.train_model import train_model
                loader = DatasetLoader()
                data = loader.load_all()
                train_model(data["movies"], data["ratings"], save_artifacts=True)
                st.success("Model trained and metrics saved!")
                st.rerun()
    else:
        # Four Primary Metrics
        acc = metrics.get("accuracy", 0.0)
        prec = metrics.get("precision", 0.0)
        rec = metrics.get("recall", 0.0)
        f1 = metrics.get("f1_score", 0.0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color: #4299e1;">{acc:.2%}</div>
                <div class="stat-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color: #48bb78;">{prec:.4f}</div>
                <div class="stat-label">Precision (Liked)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color: #ed8936;">{rec:.4f}</div>
                <div class="stat-label">Recall (Liked)</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color: #e53e3e;">{f1:.4f}</div>
                <div class="stat-label">F1-Score (Liked)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualizations Row: Confusion Matrix & Metrics Chart
        col_v1, col_v2 = st.columns([1, 1])

        with col_v1:
            st.markdown("#### 🎯 Confusion Matrix (Test Split: 20,168 samples)")
            cm = metrics.get("confusion_matrix")
            if cm:
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual: Not Liked (<4.0)", "Actual: Liked (≥4.0)"],
                    columns=["Pred: Not Liked", "Pred: Liked"]
                )
                fig_cm = px.imshow(
                    cm_df,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted Class", y="Actual Class", color="Count")
                )
                fig_cm.update_layout(
                    paper_bgcolor="#161a20",
                    plot_bgcolor="#161a20",
                    font=dict(color="#ffffff"),
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        with col_v2:
            st.markdown("#### 📊 Evaluation Metrics Overview")
            metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
            metric_vals = [acc, prec, rec, f1]
            fig_bar = go.Figure(go.Bar(
                x=metric_names,
                y=metric_vals,
                marker_color=["#4299e1", "#48bb78", "#ed8936", "#e53e3e"],
                text=[f"{v:.4f}" for v in metric_vals],
                textposition="auto"
            ))
            fig_bar.update_layout(
                paper_bgcolor="#161a20",
                plot_bgcolor="#161a20",
                font=dict(color="#ffffff"),
                yaxis=dict(range=[0, 1.1], gridcolor="#242933"),
                xaxis=dict(gridcolor="#242933"),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Feature Importance
        feat_imp = metrics.get("feature_importance", {})
        if feat_imp:
            st.markdown("#### 🔍 Feature Importances (DecisionTreeClassifier)")
            top_features = list(feat_imp.items())[:10]
            feat_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
            fig_imp = px.bar(
                feat_df[::-1],
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis"
            )
            fig_imp.update_layout(
                paper_bgcolor="#161a20",
                plot_bgcolor="#161a20",
                font=dict(color="#ffffff"),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        # Model Parameters & Metadata
        st.markdown("#### ⚙️ Experiment & Methodology Details")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"""
            - **Model Type:** `DecisionTreeClassifier`
            - **Dataset Version:** `{metrics.get('dataset_version', 'ml-latest-small')}`
            - **Positive Class Threshold:** `Rating >= {metrics.get('positive_class_threshold', 4.0)}`
            - **Random Seed:** `{metrics.get('random_seed', 42)}`
            - **Evaluation Date:** `{metrics.get('evaluation_date', 'N/A')}`
            """)
        with col_d2:
            st.markdown(f"""
            - **Training Set Size:** `{metrics.get('train_size', 80668):,} samples (80%)`
            - **Test Set Size:** `{metrics.get('test_size', 20168):,} samples (20%)`
            - **Total Evaluated Records:** `{metrics.get('total_size', 100836):,}`
            - **Data Leakage Prevention:** Aggregates fitted strictly on training partition
            """)


# =================================================================
# PAGE 6: DATABASE & PROJECT INFO
# =================================================================
elif page == "💾 Database & Project Info":
    st.markdown('<div class="brand-title">Database & <span class="brand-accent">Project Info</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Live MongoDB statistics and project architecture overview</div>', unsafe_allow_html=True)

    # Live Database Statistics from MongoDB
    stats = db.get_database_stats()
    
    st.markdown("#### 🗄️ MongoDB Collections Status")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Movies Collection", f"{stats['movies']:,}")
    with col2:
        st.metric("Ratings Collection", f"{stats['ratings']:,}")
    with col3:
        st.metric("Users Collection", f"{stats['users']:,}")
    with col4:
        st.metric("Predictions Saved", f"{stats['predictions']:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### 🏗️ Architecture & Component Overview")
    st.markdown("""
    | Component | Technology | Purpose |
    |---|---|---|
    | **Database** | MongoDB 7.0 Community | Primary document store for movies, ratings, users, predictions |
    | **Driver** | PyMongo 4.6+ | Native Python interface with compound unique indexing & bulk operations |
    | **Supervised Classifier** | Scikit-learn `DecisionTreeClassifier` | Predicts user rating preference (liked ≥ 4.0) |
    | **Content-Based Engine** | Cosine Similarity on Multi-Hot Genres | Recommends similar films based on movie metadata |
    | **UI Framework** | Streamlit | Cinematic dark theme user interface |
    | **Evaluation** | Scikit-learn metrics | Accuracy, Precision, Recall, F1, Confusion Matrix |
    """)

    st.markdown("---")
    st.markdown("#### 📚 Dataset Attribution & Terms of Use")
    st.markdown("""
    - **Dataset:** MovieLens Latest Small (ml-latest-small)
    - **Curated By:** GroupLens Research Group, Department of Computer Science and Engineering, University of Minnesota
    - **Citation:** F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
    - **URL:** [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
    - **Dataset License:** Non-commercial educational research use.
    """)

    st.markdown("---")
    st.markdown("#### 🔍 PyMongo Indexing Structure")
    st.code("""
# Verified Indexes in cinescope_db:
movies:
  - idx_movieId: {"movieId": 1} (UNIQUE)
  - idx_title: {"title": 1}
  - idx_genres: {"genres": 1}
  - idx_avg_rating: {"average_rating": -1}
  - idx_rating_count: {"rating_count": -1}

ratings:
  - idx_user_movie: {"userId": 1, "movieId": 1} (UNIQUE COMPOUND)
  - idx_rating_movieId: {"movieId": 1}
  - idx_rating_userId: {"userId": 1}

users:
  - idx_user_id: {"user_id": 1} (UNIQUE)
    """, language="python")

# Handle any navigation overrides
if "nav_override" in st.session_state:
    del st.session_state["nav_override"]
