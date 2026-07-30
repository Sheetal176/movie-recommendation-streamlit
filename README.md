# 🎬 Movie Recommendation System

An interactive, content-based recommendation web application built with **Streamlit**, **scikit-learn**, and **Pandas**. This application implements the recommendation engine from **Project 4** of `AI_Playground_4_Real_World_AI_Projects_v4.ipynb`.

## 🚀 Key Features

1. **Title Recommendations**: Select any movie from the dataset to find the top $N$ most similar movies based on plot descriptions.
2. **Custom Text Search**: Type any arbitrary description (e.g. *"space battle with alien monsters"*) to get instant real-time recommendations.
3. **Similarity Heatmap**: Visualize the pairwise Cosine Similarity Matrix between all titles.
4. **Dynamic Add Movie**: Add custom titles and descriptions to expand the dataset in real-time.
5. **Educational Q&A**: Explanations of TF-IDF, Cosine Similarity, and interview questions.

---

## 🛠️ Local Setup Instructions

1. Clone or download the repository:
   ```bash
   cd c:/Users/abhineetsinha/Documents/movie-recommendation-streamlit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the app:
   ```bash
   streamlit run app.py
   ```
