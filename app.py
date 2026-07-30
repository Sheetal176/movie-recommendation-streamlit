"""
AI Playground: Movie Recommendation System
Built with Streamlit, TF-IDF, and Cosine Similarity.
Extracted from Project 4 of AI Playground notebook.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page Configuration ───────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Playground: Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #fafafa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    .rec-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 15px;
    }
    .badge {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# ── Title & Intro ─────────────────────────────────────────────────────────
st.title("🎬 AI Playground: Movie Recommendation System")
st.write(
    "A **Content-Based Recommendation System** that compares text descriptions of movies using "
    "**TF-IDF** and **Cosine Similarity** to suggest movies mathematically similar to your favorites."
)
st.divider()

# ── Default Dataset ───────────────────────────────────────────────────────
DEFAULT_MOVIES = pd.DataFrame({
    "title": [
        "Interstellar", "Inception", "The Martian", "Arrival",
        "The Matrix", "Avatar", "Titanic", "The Notebook",
        "Avengers: Endgame", "Iron Man", "Jurassic Park", "The Dark Knight"
    ],
    "description": [
        "space science fiction astronauts future adventure",
        "science fiction dreams technology thriller mind bending",
        "space science fiction astronaut survival mars adventure",
        "science fiction aliens language space mystery",
        "science fiction technology artificial intelligence action",
        "science fiction space aliens adventure fantasy",
        "romance drama ship ocean historical tragedy",
        "romance relationship love drama emotional",
        "superhero action marvel time travel adventure",
        "superhero action technology marvel engineering",
        "dinosaurs science adventure action island",
        "superhero action crime batman thriller"
    ]
})

# Initialize dataset in session state
if "movies_df" not in st.session_state:
    st.session_state["movies_df"] = DEFAULT_MOVIES.copy()

movies = st.session_state["movies_df"]

# ── Recommendation Engine Helper ──────────────────────────────────────────
def get_recommendations(movie_title, top_n=5):
    if movie_title not in movies["title"].values:
        return None, None
        
    vectorizer = TfidfVectorizer(stop_words="english")
    movie_matrix = vectorizer.fit_transform(movies["description"])
    similarity_matrix = cosine_similarity(movie_matrix)
    
    movie_idx = movies.index[movies["title"] == movie_title][0]
    scores = list(enumerate(similarity_matrix[movie_idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the movie itself
    scores = [item for item in scores if item[0] != movie_idx]
    
    recs = []
    for idx, score in scores[:top_n]:
        recs.append({
            "title": movies.iloc[idx]["title"],
            "description": movies.iloc[idx]["description"],
            "similarity": score,
            "match_pct": round(score * 100, 1)
        })
    return pd.DataFrame(recs), similarity_matrix

# ── Custom Text Search Helper ──────────────────────────────────────────────
def get_custom_recommendations(user_text, top_n=5):
    if not user_text.strip():
        return None
        
    vectorizer = TfidfVectorizer(stop_words="english")
    # Combine user text + dataset to ensure unified vocabulary
    all_texts = list(movies["description"]) + [user_text]
    matrix = vectorizer.fit_transform(all_texts)
    
    # User vector is the last row
    user_vector = matrix[-1]
    dataset_matrix = matrix[:-1]
    
    sim_scores = cosine_similarity(user_vector, dataset_matrix)[0]
    indexed_scores = list(enumerate(sim_scores))
    indexed_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    
    recs = []
    for idx, score in indexed_scores[:top_n]:
        recs.append({
            "title": movies.iloc[idx]["title"],
            "description": movies.iloc[idx]["description"],
            "similarity": score,
            "match_pct": round(score * 100, 1)
        })
    return pd.DataFrame(recs)

# ── Tabs Navigation ───────────────────────────────────────────────────────
tab_select, tab_search, tab_heatmap, tab_add, tab_edu = st.tabs([
    "🎯 Movie Selector",
    "🔍 Custom Prompt Search",
    "📊 Similarity Matrix Heatmap",
    "➕ Add New Movie",
    "📖 Learning Corner"
])

# --- Tab 1: Movie Selector ---
with tab_select:
    st.subheader("Select a Movie You Like")
    
    col_input, col_slider = st.columns([2, 1])
    with col_input:
        selected_movie = st.selectbox(
            "Choose a movie from the dataset:",
            options=movies["title"].values,
            index=0
        )
    with col_slider:
        num_recs = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=10,
            value=4
        )
        
    if selected_movie:
        # Display selected movie details
        sel_desc = movies.loc[movies["title"] == selected_movie, "description"].values[0]
        st.info(f"**Selected Movie:** {selected_movie}  \n**Keywords:** *{sel_desc}*")
        
        recs_df, _ = get_recommendations(selected_movie, top_n=num_recs)
        
        st.subheader("🍿 Recommended Movies")
        if recs_df is not None and not recs_df.empty():
            for idx, row in recs_df.iterrows():
                col_title, col_bar = st.columns([2, 3])
                with col_title:
                    st.markdown(f"### #{idx + 1} {row['title']}")
                    st.markdown(f"**Keywords:** *{row['description']}*")
                with col_bar:
                    st.write(f"**Match Score:** {row['match_pct']}%")
                    st.progress(float(row['similarity']))
                st.divider()

# --- Tab 2: Custom Prompt Search ---
with tab_search:
    st.subheader("Search Movies by Custom Description")
    st.write("Type any plot summary or genre keywords to find matching movies in the database:")
    
    custom_query = st.text_input(
        "Enter plot ideas or keywords:",
        placeholder="e.g. A romantic story about lovers on a tragic voyage...",
        key="custom_query"
    )
    
    custom_top_n = st.slider("Max results:", min_value=1, max_value=8, value=4, key="custom_slider")
    
    if st.button("🔍 Find Matching Movies", type="primary"):
        if not custom_query.strip():
            st.warning("Please enter some keywords first!")
        else:
            custom_recs = get_custom_recommendations(custom_query, top_n=custom_top_n)
            if custom_recs is not None and not custom_recs.empty():
                st.success(f"Top matches for: *\"{custom_query}\"*")
                for idx, row in custom_recs.iterrows():
                    col_t, col_b = st.columns([2, 3])
                    with col_t:
                        st.markdown(f"### #{idx + 1} {row['title']}")
                        st.markdown(f"**Keywords:** *{row['description']}*")
                    with col_b:
                        st.write(f"**Match Score:** {row['match_pct']}%")
                        st.progress(float(row['similarity']))
                    st.divider()

# --- Tab 3: Heatmap ---
with tab_heatmap:
    st.subheader("📊 Pairwise Cosine Similarity Heatmap")
    st.write("Visualizing how mathematically similar every movie description is to every other title:")
    
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(movies["description"])
    sim_matrix = cosine_similarity(matrix)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=movies["title"],
        yticklabels=movies["title"],
        ax=ax
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Movie Description Similarity Matrix")
    st.pyplot(fig)

# --- Tab 4: Add New Movie ---
with tab_add:
    st.subheader("➕ Add a Custom Movie to the Dataset")
    
    with st.form("add_movie_form"):
        new_title = st.text_input("Movie Title:", placeholder="e.g. Star Wars")
        new_desc = st.text_area(
            "Movie Description / Keywords:",
            placeholder="e.g. space science fiction jedi alien adventure rebellion"
        )
        submit = st.form_submit_button("Add Movie")
        
        if submit:
            if not new_title.strip() or not new_desc.strip():
                st.error("Please provide both a title and keywords!")
            elif new_title in movies["title"].values:
                st.warning(f"'{new_title}' is already in the dataset!")
            else:
                new_row = pd.DataFrame({"title": [new_title.strip()], "description": [new_desc.strip()]})
                st.session_state["movies_df"] = pd.concat([st.session_state["movies_df"], new_row], ignore_index=True)
                st.success(f"✅ Added '{new_title}' to the dataset!")
                st.rerun()
                
    st.subheader("Current Dataset")
    st.dataframe(st.session_state["movies_df"], use_container_width=True)

# --- Tab 5: Learning Corner ---
with tab_edu:
    st.subheader("📖 Learning & Explanation Corner")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        ### How Content-Based Filtering Works
        1. **TF-IDF Vectorization**: Converts raw text descriptions into numeric vectors. Unique, distinctive words score higher.
        2. **Cosine Similarity**: Measures the angle between two numeric vectors in multidimensional space.
           $$\\text{Cosine Similarity} = \\cos(\\theta) = \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|}$$
           - **Score = 1.0**: Perfect match (identical descriptions).
           - **Score = 0.0**: Completely unrelated descriptions.
        """)
        
    with col_b:
        with st.expander("💼 Interview Q&A Corner"):
            st.write("""
            **Q: What is the difference between content-based and collaborative filtering?**
            * **Content-Based**: Compares item features/descriptions directly (e.g. genres, keywords). Works well for new items.
            * **Collaborative Filtering**: Compares patterns across user behavior ("Users who liked X also liked Y").
            
            **Q: Why use Cosine Similarity over Euclidean Distance for text?**
            * Cosine similarity measures vector orientation (angle) rather than length, making it invariant to document length.
            
            **Q: What is the "Cold Start Problem"?**
            * Occurs when a new user or new item enters the system without historical interaction data. Content-based filtering handles new items much better than collaborative filtering.
            """)
