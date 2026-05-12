from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import csv
import json
import random
from flask import jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from acre_engine import get_acre_recommendations, get_acre_explanation
from fuse_engine import FUSEEngine
from chatbot_engine import ChatbotEngine
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'acre-secret-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=True)
    full_name = db.Column(db.String(200), nullable=True)
    favorite_genres = db.Column(db.Text, nullable=True)
    favorite_directors = db.Column(db.Text, nullable=True)
    bio = db.Column(db.Text, nullable=True)
    movies_per_week = db.Column(db.Integer, nullable=True, default=0)
    created_at = db.Column(db.DateTime, default=db.func.now())

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    selected_set = db.Column(db.String(50), nullable=False)
    familiarity_score = db.Column(db.Integer, nullable=False)
    match_score = db.Column(db.Integer, nullable=False)
    comments = db.Column(db.Text, nullable=True)
    
    input_movie_ids = db.Column(db.Text, nullable=True)
    acre_movie_ids = db.Column(db.Text, nullable=True)
    fuse_movie_ids = db.Column(db.Text, nullable=True)
    overlap_count = db.Column(db.Integer, nullable=True)
    
    acre_intra_list_diversity = db.Column(db.Float, nullable=True)
    fuse_intra_list_diversity = db.Column(db.Float, nullable=True)
    acre_mean_cosine_sim = db.Column(db.Float, nullable=True)
    fuse_mean_cosine_sim = db.Column(db.Float, nullable=True)
    acre_semantic_novelty = db.Column(db.Float, nullable=True)
    fuse_semantic_novelty = db.Column(db.Float, nullable=True)
    acre_genre_dist = db.Column(db.Text, nullable=True)
    fuse_genre_dist = db.Column(db.Text, nullable=True)

class UserMovie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default='watchlist') # 'watchlist' or 'watched'
    recommended_by = db.Column(db.String(50), nullable=True) # algorithm name
    user_rating = db.Column(db.Integer, nullable=True)
    user_review = db.Column(db.Text, nullable=True)
    folder_id = db.Column(db.Integer, db.ForeignKey('watchlist_folder.id'), nullable=True)
    added_at = db.Column(db.DateTime, default=db.func.now())

class RecommendationSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_movie_ids = db.Column(db.Text, nullable=False)
    acre_movie_ids = db.Column(db.Text, nullable=True)
    fuse_movie_ids = db.Column(db.Text, nullable=True)
    chosen_set = db.Column(db.String(20), nullable=True)
    viewed_movie_ids = db.Column(db.Text, nullable=True) # JSON list of IDs actually seen
    created_at = db.Column(db.DateTime, default=db.func.now())

class WatchlistFolder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.now())

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

MOVIES_DATA = []
MOVIES_BY_ID = {}

def load_movies():
    global MOVIES_DATA, MOVIES_BY_ID
    if not MOVIES_DATA:
        csv_path = os.path.join(os.path.dirname(__file__), 'Artifacts', 'preprocessing', 'cleaned_movies.csv')
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    movie = {
                        'id': row['id'],
                        'title': row['title'],
                        'poster_path': row['poster_path'],
                        'release_year': row.get('release_year', ''),
                        'genres': row.get('genres', ''),
                        'vote_average': row.get('vote_average', ''),
                        'overview': row.get('overview', ''),
                        'runtime': row.get('runtime', ''),
                        'tagline': row.get('tagline', ''),
                        'original_language': row.get('original_language', ''),
                        'production_companies': row.get('production_companies', ''),
                        'production_countries': row.get('production_countries', '')
                    }
                    MOVIES_DATA.append(movie)
                    MOVIES_BY_ID[int(movie['id'])] = movie
            print(f"Loaded {len(MOVIES_DATA)} movies.")
        except Exception as e:
            print(f"Error loading movies: {e}")

load_movies()

fuse_engine_instance = None
chatbot_instance = None

def initialize_engines():
    global fuse_engine_instance, chatbot_instance
    print("Initializing Engines...")
    
    # Pre-load ACRE (calls _load_artifacts internally once)
    from acre_engine import _load_artifacts
    _load_artifacts()
    
    # Pre-load FUSE
    fuse_pkl_path = os.path.join(os.path.dirname(__file__), 'Artifacts', 'models', 'fuse_engine.pkl')
    if os.path.exists(fuse_pkl_path):
        try:
            fuse_engine_instance = FUSEEngine.from_pkl(fuse_pkl_path)
            print("FUSE Engine loaded.")
        except Exception as e:
            print(f"Error loading FUSE engine setup: {e}")
    else:
        print("FUSE model pkl not found at:", fuse_pkl_path)

    # Initialize Chatbot
    try:
        df_for_chat = pd.DataFrame(MOVIES_DATA)
        chatbot_instance = ChatbotEngine(fuse_engine_instance, df_for_chat)
        print("Chatbot initialized.")
    except Exception as e:
        print(f"Error initializing Chatbot: {e}")

initialize_engines()

def compute_recsys_metrics(input_titles, rec_titles):
    if fuse_engine_instance is None or not rec_titles:
        return {
            'intra_list_diversity': 0.0,
            'mean_cosine_sim': 0.0,
            'semantic_novelty': 0.0,
            'genre_dist': '{}'
        }
        
    input_idx = []
    for t in input_titles:
        try:
            input_idx.append(fuse_engine_instance.get_movie_index(t))
        except:
            pass
            
    rec_idx = []
    for t in rec_titles:
        try:
            rec_idx.append(fuse_engine_instance.get_movie_index(t))
        except:
            pass
            
    ild = 0.0
    if len(rec_idx) > 1:
        rec_embs = fuse_engine_instance.embedding_matrix[rec_idx]
        sim_matrix = cosine_similarity(rec_embs)
        n = len(rec_idx)
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        ild = np.mean(1.0 - upper_tri) if len(upper_tri) > 0 else 0.0
        
    mean_cosine_sim = 0.0
    semantic_novelty = 0.0
    if input_idx and rec_idx:
        input_embs = fuse_engine_instance.embedding_matrix[input_idx]
        rec_embs = fuse_engine_instance.embedding_matrix[rec_idx]
        
        user_profile = np.mean(input_embs, axis=0).reshape(1, -1)
        sim_scores = cosine_similarity(rec_embs, user_profile).flatten()
        mean_cosine_sim = np.float64(np.mean(sim_scores))
        
        cross_sims = cosine_similarity(rec_embs, input_embs)
        max_sims = np.max(cross_sims, axis=1)
        semantic_novelty = np.float64(np.mean(1.0 - max_sims))
        
    genre_counts = {}
    for idx in rec_idx:
        row = fuse_engine_instance.df.iloc[idx]
        genres = str(row['genres']).split(',')
        for g in genres:
            g = g.strip()
            if g:
                genre_counts[g] = genre_counts.get(g, 0) + 1
                
    return {
        'intra_list_diversity': float(ild),
        'mean_cosine_sim': float(mean_cosine_sim),
        'semantic_novelty': float(semantic_novelty),
        'genre_dist': json.dumps(genre_counts)
    }

def get_fuse_recommendations(user_choices, exclude_ids, top_n=15):
    global fuse_engine_instance
    if fuse_engine_instance is None:
        return [], {}
            
    fuse_ids = []
    fuse_explanations = {}
    try:
        input_titles = [m['title'] for m in user_choices]
        # fuse_output returns (result_dict, combo_dict, raw_explanation_str)
        fuse_output = fuse_engine_instance.recommend(input_titles, top_n=top_n*2, verbose=False)
        
        title_to_id = {m['title'].lower(): int(m['id']) for m in MOVIES_DATA}
        
        for res, combo, _ in fuse_output:
            t = res['title'].lower()
            if t in title_to_id:
                m_id = title_to_id[t]
                if m_id not in exclude_ids and m_id not in fuse_ids:
                    fuse_ids.append(m_id)
                    
                    # Generate a cleaner, concise web explanation
                    clean_exp = fuse_engine_instance.get_web_explanation(combo)
                    fuse_explanations[m_id] = clean_exp
                    
            if len(fuse_ids) >= top_n:
                break
    except Exception as e:
        print(f"Error running FUSE recommend: {e}")
        
    return fuse_ids, fuse_explanations


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    if request.method == 'POST':
        selected_ids_json = request.form.get('selected_movies')
        
        selected_ids = []
        if selected_ids_json:
            try:
                selected_ids = json.loads(selected_ids_json)
            except Exception:
                pass
                
        user_choices = [m for m in MOVIES_DATA if str(m['id']) in map(str, selected_ids)]
        
        tmdb_ids = [int(i) for i in selected_ids]
        
        # Exclude movies the user has marked as watched
        exclude_ids = [um.movie_id for um in UserMovie.query.filter_by(user_id=current_user.id, status='watched').all()]
        
        acre_ids, acre_exps = get_acre_recommendations(tmdb_ids, top_n=15, exclude_ids=exclude_ids)
        fuse_ids, fuse_exps = get_fuse_recommendations(user_choices, exclude_ids=exclude_ids, top_n=15)

        # Build sets using cached MOVIES_BY_ID (Optimization)
        acre_movies = []
        for tid in acre_ids:
            if tid in MOVIES_BY_ID:
                m_copy = MOVIES_BY_ID[tid].copy()
                # Generate specific explanation for ACRE
                m_copy['explanation'] = get_acre_explanation(tid, tmdb_ids)
                acre_movies.append(m_copy)
        
        fuse_movies = []
        for tid in fuse_ids:
            if tid in MOVIES_BY_ID:
                m_copy = MOVIES_BY_ID[tid].copy()
                if tid in fuse_exps:
                    m_copy['explanation'] = fuse_exps[tid]
                fuse_movies.append(m_copy)

        # Explicitly label ACRE and FUSE (No longer blind)
        set_a_movies = acre_movies
        set_type_a = "ACRE"
        set_b_movies = fuse_movies
        set_type_b = "FUSE"
        
        input_titles = [m['title'] for m in user_choices]
        acre_titles = [m['title'] for m in acre_movies]
        fuse_titles = [m['title'] for m in fuse_movies]
        
        acre_metrics = compute_recsys_metrics(input_titles, acre_titles)
        fuse_metrics = compute_recsys_metrics(input_titles, fuse_titles)
        
        overlap_count = len(set(acre_ids).intersection(set(fuse_ids)))
        
        metrics_data = {
            'input_movie_ids': json.dumps(tmdb_ids),
            'acre_movie_ids': json.dumps(acre_ids),
            'fuse_movie_ids': json.dumps(fuse_ids),
            'overlap_count': overlap_count,
            'acre_intra_list_diversity': acre_metrics['intra_list_diversity'],
            'fuse_intra_list_diversity': fuse_metrics['intra_list_diversity'],
            'acre_mean_cosine_sim': acre_metrics['mean_cosine_sim'],
            'fuse_mean_cosine_sim': fuse_metrics['mean_cosine_sim'],
            'acre_semantic_novelty': acre_metrics['semantic_novelty'],
            'fuse_semantic_novelty': fuse_metrics['semantic_novelty'],
            'acre_genre_dist': acre_metrics['genre_dist'],
            'fuse_genre_dist': fuse_metrics['genre_dist']
        }
        
        return render_template('recommend.html', 
                               user_choices=user_choices, 
                               set_a=set_a_movies, 
                               set_b=set_b_movies,
                               set_type_a=set_type_a,
                               set_type_b=set_type_b,
                               metrics_data=metrics_data)
        
    movies = MOVIES_DATA[:60]
    return render_template('input.html', movies=movies)

@app.route('/dashboard')
@login_required
def dashboard():
    feedbacks = Feedback.query.all()
    
    # Aggregated Stats
    total_reviews = len(feedbacks)
    acre_wins = len([f for f in feedbacks if f.selected_set == 'ACRE'])
    fuse_wins = len([f for f in feedbacks if f.selected_set == 'FUSE'])
    
    # Calculate Averages
    def get_avg(attr):
        vals = [getattr(f, attr) for f in feedbacks if getattr(f, attr) is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0

    stats = {
        'total_reviews': total_reviews,
        'win_ratio': {
            'ACRE': round(acre_wins / total_reviews * 100, 1) if total_reviews else 0,
            'FUSE': round(fuse_wins / total_reviews * 100, 1) if total_reviews else 0
        },
        'avg_match': get_avg('match_score'),
        'avg_familiarity': get_avg('familiarity_score'),
        'metrics': {
            'ACRE': {
                'diversity': get_avg('acre_intra_list_diversity'),
                'similarity': get_avg('acre_mean_cosine_sim'),
                'novelty': get_avg('acre_semantic_novelty')
            },
            'FUSE': {
                'diversity': get_avg('fuse_intra_list_diversity'),
                'similarity': get_avg('fuse_mean_cosine_sim'),
                'novelty': get_avg('fuse_semantic_novelty')
            }
        }
    }
    
    return render_template('dashboard.html', stats=stats)


@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    if not chatbot_instance:
        return jsonify({'response': 'Chatbot is initializing or unavailable. Please try again later.'}), 503
        
    data = request.json
    message = data.get('message', '')
    session_id = f"user_{current_user.id}"
    
    try:
        response = chatbot_instance.process(message, session_id=session_id)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Sorry, something went wrong with the chatbot: {str(e)}'}), 500

@app.route('/api/search')
@login_required
def search_movies():
    query = request.args.get('q', '').lower()
    
    if not query:
        return jsonify({'movies': MOVIES_DATA[:60]})
        
    results = []
    for m in MOVIES_DATA:
        if query in m['title'].lower():
            results.append(m)
            if len(results) >= 60:
                break
                
    return jsonify({'movies': results})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.', 'error')
            return redirect(url_for('login'))
            
        login_user(user)
        # Smart welcome: check if user has any recommendation sessions
        has_sessions = RecommendationSession.query.filter_by(user_id=user.id).first()
        if has_sessions:
            flash(f'Welcome back, {user.full_name or user.username}! 🎬', 'welcome')
        else:
            flash(f'Welcome to ACRE-FUSE, {user.full_name or user.username}! 🌟 Start by getting your first recommendations.', 'welcome')
        return redirect(url_for('home'))
        
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email', '')
        full_name = request.form.get('full_name', '')
        favorite_genres = request.form.get('favorite_genres', '')
        favorite_directors = request.form.get('favorite_directors', '')
        movies_per_week = request.form.get('movies_per_week', 0)
        
        user = User.query.filter_by(username=username).first()
        
        if user:
            flash('Username already exists. Please login.', 'error')
            return redirect(url_for('signup'))
        
        if email:
            email_exists = User.query.filter_by(email=email).first()
            if email_exists:
                flash('Email already registered. Please login.', 'error')
                return redirect(url_for('signup'))
            
        new_user = User(
            username=username,
            password=generate_password_hash(password, method='pbkdf2:sha256'),
            email=email,
            full_name=full_name,
            favorite_genres=favorite_genres,
            favorite_directors=favorite_directors,
            movies_per_week=int(movies_per_week) if movies_per_week else 0
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/submit_feedback', methods=['POST'])
@login_required
def submit_feedback():
    preferred_set = request.form.get('preferred_set') # 'A' or 'B'
    set_a_alg = request.form.get('set_a_alg')
    set_b_alg = request.form.get('set_b_alg')
    
    selected_alg = set_a_alg if preferred_set == 'A' else set_b_alg
    
    familiarity = request.form.get('familiarity')
    match_score = request.form.get('match_score')
    comments = request.form.get('comments')
    
    feedback = Feedback(
        user_id=current_user.id,
        selected_set=selected_alg,
        familiarity_score=int(familiarity) if familiarity else 0,
        match_score=int(match_score) if match_score else 0,
        comments=comments,
        input_movie_ids=request.form.get('input_movie_ids'),
        acre_movie_ids=request.form.get('acre_movie_ids'),
        fuse_movie_ids=request.form.get('fuse_movie_ids'),
        overlap_count=int(request.form.get('overlap_count', 0)),
        acre_intra_list_diversity=float(request.form.get('acre_intra_list_diversity', 0.0)),
        fuse_intra_list_diversity=float(request.form.get('fuse_intra_list_diversity', 0.0)),
        acre_mean_cosine_sim=float(request.form.get('acre_mean_cosine_sim', 0.0)),
        fuse_mean_cosine_sim=float(request.form.get('fuse_mean_cosine_sim', 0.0)),
        acre_semantic_novelty=float(request.form.get('acre_semantic_novelty', 0.0)),
        fuse_semantic_novelty=float(request.form.get('fuse_semantic_novelty', 0.0)),
        acre_genre_dist=request.form.get('acre_genre_dist'),
        fuse_genre_dist=request.form.get('fuse_genre_dist')
    )
    db.session.add(feedback)
    
    # Save recommendation session for history tracking
    session_record = RecommendationSession(
        user_id=current_user.id,
        input_movie_ids=request.form.get('input_movie_ids', '[]'),
        acre_movie_ids=request.form.get('acre_movie_ids', '[]'),
        fuse_movie_ids=request.form.get('fuse_movie_ids', '[]'),
        chosen_set=selected_alg,
        viewed_movie_ids=request.form.get('viewed_movie_ids', '[]')
    )
    db.session.add(session_record)
    db.session.commit()
    return redirect(url_for('thank_you'))

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

@app.route('/account')
@login_required
def account():
    # Stats
    watchlist_count = UserMovie.query.filter_by(user_id=current_user.id, status='watchlist').count()
    watched_count = UserMovie.query.filter_by(user_id=current_user.id, status='watched').count()
    feedback_count = Feedback.query.filter_by(user_id=current_user.id).count()
    sessions_count = RecommendationSession.query.filter_by(user_id=current_user.id).count()
    # Total unique movies recommended across all sessions
    all_sessions = RecommendationSession.query.filter_by(user_id=current_user.id).all()
    all_rec_ids = set()
    for s in all_sessions:
        if s.acre_movie_ids:
            try: all_rec_ids.update(json.loads(s.acre_movie_ids))
            except: pass
        if s.fuse_movie_ids:
            try: all_rec_ids.update(json.loads(s.fuse_movie_ids))
            except: pass
    total_recommended = len(all_rec_ids)
    
    
    # Folders
    folders = WatchlistFolder.query.filter_by(user_id=current_user.id).all()
    
    # First 6 items for preview
    watchlist_preview = []
    watched_preview = []
    watchlist_ids = set()
    watched_ids = set()
    user_movies = UserMovie.query.filter_by(user_id=current_user.id).order_by(UserMovie.added_at.desc()).all()
    for um in user_movies:
        if um.movie_id in MOVIES_BY_ID:
            movie_detail = MOVIES_BY_ID[um.movie_id].copy()
            movie_detail['db_id'] = um.id
            movie_detail['recommended_by'] = um.recommended_by
            movie_detail['user_rating'] = um.user_rating
            movie_detail['user_review'] = um.user_review
            movie_detail['folder_id'] = um.folder_id
            if um.status == 'watchlist':
                watchlist_ids.add(um.movie_id)
                if len(watchlist_preview) < 20:
                    watchlist_preview.append(movie_detail)
            elif um.status == 'watched':
                watched_ids.add(um.movie_id)
                if len(watched_preview) < 20:
                    watched_preview.append(movie_detail)
                    
    # Last recommendation session
    last_session = RecommendationSession.query.filter_by(user_id=current_user.id).order_by(RecommendationSession.created_at.desc()).first()
    last_session_movies = []
    last_session_total_count = 0
    if last_session:
        chosen_ids_str = last_session.viewed_movie_ids if last_session.viewed_movie_ids else (last_session.acre_movie_ids if last_session.chosen_set == 'ACRE' else last_session.fuse_movie_ids)
        if chosen_ids_str:
            try:
                chosen_ids = json.loads(chosen_ids_str)
                last_session_total_count = len(chosen_ids)
                
                all_session_movies = [MOVIES_BY_ID[mid] for mid in chosen_ids if mid in MOVIES_BY_ID]
                
                # Filter to show unwatched first, then limit to 5
                unwatched = [m for m in all_session_movies if int(m['id']) not in watched_ids]
                already_watched = [m for m in all_session_movies if int(m['id']) in watched_ids]
                
                last_session_movies = (unwatched + already_watched)[:5]
            except: 
                all_session_movies = []
                last_session_movies = []
    else:
        all_session_movies = []
                
    stats = {
        'watchlist': watchlist_count,
        'watched': watched_count,
        'feedback': feedback_count,
        'sessions': sessions_count,
        'total_recommended': total_recommended
    }
    
    return render_template('account.html', user=current_user, 
                           watchlist=watchlist_preview, watched_history=watched_preview,
                           stats=stats, folders=folders,
                           last_session_movies=last_session_movies,
                           all_session_movies=all_session_movies,
                           last_session=last_session,
                           last_session_total_count=last_session_total_count,
                           watchlist_total=watchlist_count, watched_total=watched_count,
                           watchlist_ids=watchlist_ids, watched_ids=watched_ids)

# --- Lazy Loading APIs ---
@app.route('/api/watchlist/items')
@login_required
def api_watchlist_items():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    folder_id = request.args.get('folder_id', None, type=int)
    
    query = UserMovie.query.filter_by(user_id=current_user.id, status='watchlist')
    if folder_id:
        query = query.filter_by(folder_id=folder_id)
    
    total = query.count()
    items = query.order_by(UserMovie.added_at.desc()).offset((page-1)*per_page).limit(per_page).all()
    
    movies = []
    for um in items:
        if um.movie_id in MOVIES_BY_ID:
            m = MOVIES_BY_ID[um.movie_id].copy()
            m['db_id'] = um.id
            m['recommended_by'] = um.recommended_by
            m['folder_id'] = um.folder_id
            movies.append(m)
    
    return jsonify({'movies': movies, 'total': total, 'page': page, 'has_more': page * per_page < total})

@app.route('/api/watched/items')
@login_required
def api_watched_items():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    query = UserMovie.query.filter_by(user_id=current_user.id, status='watched')
    total = query.count()
    items = query.order_by(UserMovie.added_at.desc()).offset((page-1)*per_page).limit(per_page).all()
    
    movies = []
    for um in items:
        if um.movie_id in MOVIES_BY_ID:
            m = MOVIES_BY_ID[um.movie_id].copy()
            m['db_id'] = um.id
            m['recommended_by'] = um.recommended_by
            m['user_rating'] = um.user_rating
            m['user_review'] = um.user_review
            movies.append(m)
    
    return jsonify({'movies': movies, 'total': total, 'page': page, 'has_more': page * per_page < total})

# --- Watchlist CRUD ---
@app.route('/api/watchlist/add', methods=['POST'])
@login_required
def watchlist_add():
    data = request.json
    movie_id = data.get('movie_id')
    recommended_by = data.get('recommended_by', '')
    folder_id = data.get('folder_id', None)
    
    if not movie_id:
        return jsonify({'error': 'No movie_id provided'}), 400
        
    try:
        movie_id = int(movie_id)
        existing = UserMovie.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
        if existing:
            return jsonify({'message': 'Already in lists', 'already': True}), 200
            
        new_um = UserMovie(user_id=current_user.id, movie_id=movie_id, status='watchlist', 
                          recommended_by=recommended_by, folder_id=folder_id)
        db.session.add(new_um)
        db.session.commit()
        return jsonify({'message': 'Added to watchlist!', 'already': False, 'db_id': new_um.id}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/watchlist/remove', methods=['POST'])
@login_required
def watchlist_remove():
    data = request.json
    db_id = data.get('db_id')
    if not db_id:
        return jsonify({'error': 'No standard input'}), 400
        
    um = UserMovie.query.get(db_id)
    if um and um.user_id == current_user.id:
        db.session.delete(um)
        db.session.commit()
        return jsonify({'success': True}), 200
    return jsonify({'error': 'Not found or not authorized'}), 404

@app.route('/api/watchlist/watched', methods=['POST'])
@login_required
def watchlist_watched():
    data = request.json
    db_id = data.get('db_id')
    rating = data.get('rating')
    review = data.get('review')
    
    if not db_id:
        return jsonify({'error': 'No ID provided'}), 400
        
    um = UserMovie.query.get(db_id)
    if um and um.user_id == current_user.id:
        um.status = 'watched'
        if rating is not None:
            um.user_rating = int(rating)
        if review is not None:
            um.user_review = review
        db.session.commit()
        return jsonify({'success': True}), 200
    return jsonify({'error': 'Not found or not authorized'}), 404

@app.route('/api/watched/add_direct', methods=['POST'])
@login_required
def watched_add_direct():
    data = request.json
    movie_id = data.get('movie_id')
    recommended_by = data.get('recommended_by', '')
    
    if not movie_id:
        return jsonify({'error': 'No movie_id provided'}), 400
        
    try:
        movie_id = int(movie_id)
        existing = UserMovie.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
        if existing:
            existing.status = 'watched'
        else:
            new_um = UserMovie(user_id=current_user.id, movie_id=movie_id, status='watched', recommended_by=recommended_by)
            db.session.add(new_um)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Added to watched list!', 'db_id': existing.id if existing else new_um.id}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/watchlist/move', methods=['POST'])
@login_required
def watchlist_move_to_folder():
    data = request.json
    db_id = data.get('db_id')
    folder_id = data.get('folder_id')  # None to remove from folder
    
    um = UserMovie.query.get(db_id)
    if um and um.user_id == current_user.id:
        um.folder_id = folder_id
        db.session.commit()
        return jsonify({'success': True}), 200
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/folders', methods=['GET'])
@login_required
def get_folders():
    folders = WatchlistFolder.query.filter_by(user_id=current_user.id).all()
    return jsonify({'folders': [{'id': f.id, 'name': f.name} for f in folders]})

@app.route('/api/folders/full', methods=['GET'])
@login_required
def get_folders_full():
    folders = WatchlistFolder.query.filter_by(user_id=current_user.id).all()
    results = []
    for f in folders:
        movies_um = UserMovie.query.filter_by(user_id=current_user.id, folder_id=f.id).all()
        folder_movies = []
        for um in movies_um:
            if um.movie_id in MOVIES_BY_ID:
                m = MOVIES_BY_ID[um.movie_id].copy()
                m['db_id'] = um.id
                m['status'] = um.status
                m['user_rating'] = um.user_rating
                folder_movies.append(m)
        results.append({
            'id': f.id,
            'name': f.name,
            'movies': folder_movies
        })
    return jsonify({'folders': results})

@app.route('/api/folders', methods=['POST'])
@login_required
def create_folder():
    data = request.json
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Folder name required'}), 400
    existing = WatchlistFolder.query.filter_by(user_id=current_user.id, name=name).first()
    if existing:
        return jsonify({'error': 'Folder already exists'}), 400
    folder = WatchlistFolder(user_id=current_user.id, name=name)
    db.session.add(folder)
    db.session.commit()
    return jsonify({'id': folder.id, 'name': folder.name}), 201

@app.route('/api/folders/<int:folder_id>', methods=['DELETE'])
@login_required
def delete_folder(folder_id):
    folder = WatchlistFolder.query.get(folder_id)
    if folder and folder.user_id == current_user.id:
        # Remove folder assignment from movies (don't delete the movies)
        UserMovie.query.filter_by(folder_id=folder_id).update({'folder_id': None})
        db.session.delete(folder)
        db.session.commit()
        return jsonify({'success': True}), 200
    return jsonify({'error': 'Not found'}), 404

# --- Profile Update ---
@app.route('/api/profile/update', methods=['POST'])
@login_required
def update_profile():
    data = request.json
    if 'full_name' in data:
        current_user.full_name = data['full_name']
    if 'email' in data:
        current_user.email = data['email']
    if 'favorite_genres' in data:
        current_user.favorite_genres = data['favorite_genres']
    if 'favorite_directors' in data:
        current_user.favorite_directors = data['favorite_directors']
    if 'bio' in data:
        current_user.bio = data['bio']
    if 'movies_per_week' in data:
        current_user.movies_per_week = int(data['movies_per_week'])
    db.session.commit()
    return jsonify({'success': True}), 200

# --- Dedicated Pages ---
@app.route('/watchlist')
@login_required
def watchlist_page():
    folders = WatchlistFolder.query.filter_by(user_id=current_user.id).all()
    return render_template('watchlist.html', folders=folders)

@app.route('/watched')
@login_required
def watched_page():
    return render_template('watched.html')

@app.route('/session/<int:session_id>')
@login_required
def session_detail(session_id):
    session = RecommendationSession.query.get_or_404(session_id)
    if session.user_id != current_user.id:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('account'))
        
    viewed_ids = json.loads(session.viewed_movie_ids or '[]')
    if not viewed_ids:
        # Fallback to chosen set if viewed_ids is empty
        viewed_ids = json.loads(session.acre_movie_ids if session.chosen_set == 'ACRE' else session.fuse_movie_ids or '[]')
        
    movies = [MOVIES_BY_ID[mid] for mid in viewed_ids if mid in MOVIES_BY_ID]
    
    # Get user's current lists for the UI buttons
    watchlist_ids = [um.movie_id for um in UserMovie.query.filter_by(user_id=current_user.id, status='watchlist').all()]
    watched_ids = [um.movie_id for um in UserMovie.query.filter_by(user_id=current_user.id, status='watched').all()]
    
    return render_template('session_detail.html', movies=movies, session=session,
                           watchlist_ids=set(watchlist_ids), watched_ids=set(watched_ids))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
