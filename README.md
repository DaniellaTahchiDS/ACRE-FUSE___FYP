# 🎬 ACRE-FUSE: Advanced Movie Recommendation & Discovery Platform

**ACRE-FUSE** is a sophisticated movie recommendation system developed as a Final Year Project. It explores the intersection of high-dimensional clustering and semantic feature unification to provide users with both precise matches and intelligent "out-of-bubble" discovery.

![Project Preview](https://img.shields.io/badge/Status-Complete-green)
![Tech Stack](https://img.shields.io/badge/Tech-Flask_|_Scikit--learn_|_Gemini_API-blue)

---

## 🚀 Overview

The platform is built around two primary recommendation engines, each addressing a different aspect of the "filter bubble" problem in modern Recommender Systems (RecSys):

### 1. **ACRE Engine** (Adaptive Cross-Cluster Recommendation Engine)
*   **Philosophy**: Discovery through cluster proximity.
*   **How it works**: Groups movies using **PCA-reduced K-Means clustering** across a 170+ dimensional feature space. Instead of just suggesting movies from your own cluster, ACRE identifies neighboring preference groups and interleaves recommendations across them.
*   **Benefit**: Facilitates serendipity by bridging the gap between what you love and what you *might* love in adjacent thematic territories.

### 2. **FUSE Engine** (Feature Unification for Semantic Exploration)
*   **Philosophy**: Semantic hybrid profiling.
*   **How it works**: Engineered to create "hybrid profiles" by unifying segments (TF-IDF, LDA Topics, Sentence Embeddings, Sentiment, and Genres) from multiple input movies. It uses a **combinatorial assignment** strategy to find real movies that match these synthetic profiles.
*   **Benefit**: Provides highly granular and explainable recommendations that span traditional genre boundaries.

---

## ✨ Key Features

*   **Dual-Engine Recommendations**: Input 3-5 movies and get side-by-side results from ACRE and FUSE.
*   **AI Chatbot**: Integrated **Google Gemini-powered** movie assistant that understands natural language queries and leverages the FUSE embedding matrix for search.
*   **Rich Explanations**: Every recommendation comes with a human-readable explanation of *why* it was chosen (e.g., "Combines the emotional tone from Interstellar with the thematic depth of Inception").
*   **Analytics Dashboard**: A comprehensive stats page visualizing **Intra-list Diversity**, **Semantic Novelty**, and **Genre Distribution** across algorithms.
*   **Personalized Experience**:
    *   Custom watchlists and "Watched" history.
    *   Folder-based organization for movie collections.
    *   User accounts with preference tracking.
*   **Feedback System**: Built-in mechanism for users to rate and compare recommendation sets, fueling data for RecSys evaluation.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python, Flask, SQLAlchemy (SQLite), Flask-Login |
| **Data Science** | Scikit-learn, Pandas, NumPy, UMAP/PCA, Joblib |
| **NLP** | NLTK, Sentence-Transformers, LDA (Latent Dirichlet Allocation) |
| **AI** | Google Gemini 1.5 (via `google-genai`) |
| **Frontend** | HTML5, Vanilla CSS3 (Modern Glassmorphism Design), JavaScript |

---

## 📂 Project Structure

```text
ACRE-FUSE/
├── Artifacts/              # Pre-trained models, feature matrices, and clusters
├── Dataset/                # Raw and processed CSV data
├── Notebooks-ACRE/         # Development notebooks for clustering & PCA
├── Notebooks-FUSE/         # Development notebooks for NLP & Feature Engineering
├── static/                 # CSS, JS, and Images
├── templates/              # HTML Flask templates
├── acre_engine.py          # ACRE logic & artifact loading
├── fuse_engine.py          # FUSE combinatorial engine class
├── chatbot_engine.py       # Gemini API integration
├── app.py                  # Main Flask application & routes
└── requirements.txt        # Project dependencies
```

---

## ⚙️ Installation & Setup

### Prerequisites
*   Python 3.10+
*   Google Gemini API Key (Optional, for chatbot features)

### Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/acre-fuse.git
    cd acre-fuse
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    venv\Scripts\activate  # Windows
    source venv/bin/activate # Linux/Mac
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY=your_actual_key_here
    ```

5.  **Initialize Database**:
    ```bash
    # Run in python console
    python
    >>> from app import app, db
    >>> with app.app_context():
    ...     db.create_all()
    >>> exit()
    ```

6.  **Run the application**:
    ```bash
    python app.py
    ```
    Access the app at `http://127.0.0.1:5000`

---

## 📊 Evaluation Metrics

The project implements several quantitative metrics to evaluate the quality of recommendations:
*   **Intra-List Diversity (ILD)**: Measures how different the recommended items are from each other.
*   **Semantic Novelty**: Measures the distance between the user's input profile and the recommended items (rewarding non-obvious matches).
*   **User Satisfaction**: Tracked via the feedback module to measure the "perceived" quality of each engine.

---

## 👩‍💻 Author

**Daniella Tahchi**  
Final Year Project - Computer Science  

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
