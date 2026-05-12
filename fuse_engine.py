"""
FUSE Engine — Feature Unification for Semantic Exploration

A content-based movie recommendation engine that creates hybrid semantic 
profiles by combining NLP features from multiple input movies, then finds 
the best matching real movies using weighted multi-segment cosine similarity.

Usage:
    # Load from pkl
    engine = FUSEEngine.from_pkl('Artifacts/models/fuse_engine.pkl')

    # Or load from raw feature files
    engine = FUSEEngine.from_artifacts('Artifacts/features')

    # Get recommendations
    results = engine.recommend(['Inception', 'Interstellar', 'The Dark Knight'])

    # Get baseline comparison
    baseline = engine.baseline_recommend(['Inception', 'Interstellar', 'The Dark Knight'])
"""

import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from itertools import product as itertools_product
from sklearn.metrics.pairwise import cosine_similarity


class FUSEEngine:
    """
    FUSE Recommendation Engine.

    Builds combinatorial hybrid profiles from multiple input movies,
    where each feature segment (TF-IDF, topics, embeddings, sentiment, genre)
    can come from a different input movie. This produces diverse, explainable
    recommendations that span genre boundaries.

    Attributes:
        df (pd.DataFrame): Movie metadata with titles, genres, ratings, etc.
        tfidf_matrix (sp.spmatrix): Sparse TF-IDF feature matrix.
        lda_matrix (np.ndarray): LDA topic distribution matrix.
        embedding_matrix (np.ndarray): Dense sentence embedding matrix.
        sentiment_matrix (np.ndarray): Sentiment (polarity + subjectivity) matrix.
        genre_matrix (np.ndarray): One-hot genre encoding matrix.
        feature_segments (dict): Segment names → weights and descriptions.
        genre_names (list): List of genre category names.
    """

    def __init__(self, df, tfidf_matrix, lda_matrix, embedding_matrix,
                 sentiment_matrix, genre_matrix, feature_segments, genre_names):
        """
        Initialize the FUSE engine with pre-computed feature data.

        Args:
            df: DataFrame with movie metadata (must include 'title', 'popularity',
                'vote_average', 'genres', 'release_year', 'overview' columns).
            tfidf_matrix: Sparse TF-IDF matrix (n_movies × vocab_size).
            lda_matrix: LDA topic distributions (n_movies × n_topics).
            embedding_matrix: Sentence embeddings (n_movies × embed_dim).
            sentiment_matrix: Sentiment features (n_movies × 2).
            genre_matrix: One-hot genre encoding (n_movies × n_genres).
            feature_segments: Dict mapping segment names to {'weight': float, 'description': str}.
            genre_names: List of genre names corresponding to genre_matrix columns.
        """
        self.df = df
        self.tfidf_matrix = tfidf_matrix
        self.lda_matrix = lda_matrix
        self.embedding_matrix = embedding_matrix
        self.sentiment_matrix = sentiment_matrix
        self.genre_matrix = genre_matrix
        self.feature_segments = feature_segments
        self.genre_names = genre_names

        # Map segment names to their matrices for fast lookup
        self._matrices = {
            'tfidf': self._to_csr(self.tfidf_matrix),
            'topics': self.lda_matrix,
            'embedding': self.embedding_matrix,
            'sentiment': self.sentiment_matrix,
            'genre': self.genre_matrix,
        }
        
        # Pre-calculate lowercase titles for fast filtering
        self._lowercase_titles = self.df['title'].str.lower().values
    def _to_csr(self, matrix):
        """Converts matrix to CSR format for faster arithmetic if needed."""
        if sp.issparse(matrix) and not sp.isspmatrix_csr(matrix):
            return matrix.tocsr()
        return matrix

    # ── Factory Methods ─────────────────────────────────────────────────

    @classmethod
    def from_pkl(cls, pkl_path):
        """
        Load a FUSE engine from a saved pkl file.

        Args:
            pkl_path: Path to the fuse_engine.pkl file.

        Returns:
            FUSEEngine instance.
        """
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        return cls(
            df=data['df'],
            tfidf_matrix=data['tfidf_matrix'],
            lda_matrix=data['lda_matrix'],
            embedding_matrix=data['embedding_matrix'],
            sentiment_matrix=data['sentiment_matrix'],
            genre_matrix=data['genre_matrix'],
            feature_segments=data['feature_segments'],
            genre_names=data['genre_names'],
        )

    @classmethod
    def from_artifacts(cls, artifacts_dir):
        """
        Load a FUSE engine from individual artifact files.

        Args:
            artifacts_dir: Path to the features directory containing
                           movies_with_features.csv, tfidf_matrix.npz, etc.

        Returns:
            FUSEEngine instance.
        """
        df = pd.read_csv(os.path.join(artifacts_dir, 'movies_with_features.csv'))
        tfidf_matrix = sp.load_npz(os.path.join(artifacts_dir, 'tfidf_matrix.npz'))
        lda_matrix = np.load(os.path.join(artifacts_dir, 'lda_matrix.npy'))
        embedding_matrix = np.load(os.path.join(artifacts_dir, 'embedding_matrix.npy'))
        sentiment_matrix = np.load(os.path.join(artifacts_dir, 'sentiment_matrix.npy'))
        genre_matrix = np.load(os.path.join(artifacts_dir, 'genre_matrix.npy'))

        with open(os.path.join(artifacts_dir, 'genre_names.pkl'), 'rb') as f:
            genre_names = pickle.load(f)

        feature_segments = {
            'tfidf':     {'weight': 0.25, 'description': 'Writing style & vocabulary'},
            'topics':    {'weight': 0.20, 'description': 'Thematic content'},
            'embedding': {'weight': 0.25, 'description': 'Semantic meaning'},
            'sentiment': {'weight': 0.10, 'description': 'Emotional tone'},
            'genre':     {'weight': 0.20, 'description': 'Genre categories'},
        }

        return cls(
            df=df,
            tfidf_matrix=tfidf_matrix,
            lda_matrix=lda_matrix,
            embedding_matrix=embedding_matrix,
            sentiment_matrix=sentiment_matrix,
            genre_matrix=genre_matrix,
            feature_segments=feature_segments,
            genre_names=genre_names,
        )

    def save_pkl(self, pkl_path):
        """
        Save the engine state to a pkl file.

        Args:
            pkl_path: Output file path.
        """
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

        data = {
            'df': self.df,
            'tfidf_matrix': self.tfidf_matrix,
            'lda_matrix': self.lda_matrix,
            'embedding_matrix': self.embedding_matrix,
            'sentiment_matrix': self.sentiment_matrix,
            'genre_matrix': self.genre_matrix,
            'feature_segments': self.feature_segments,
            'genre_names': self.genre_names,
        }

        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)

    # ── Core Methods ────────────────────────────────────────────────────

    def get_movie_index(self, title):
        """
        Find the DataFrame index of a movie by title (case-insensitive, partial match).
        If multiple matches, returns the most popular one.

        Args:
            title: Movie title string.

        Returns:
            Integer index into self.df.

        Raises:
            ValueError: If no movie is found.
        """
        matches = self.df[self.df['title'].str.lower() == title.lower()]
        if len(matches) == 0:
            matches = self.df[self.df['title'].str.lower().str.contains(title.lower())]
        if len(matches) == 0:
            raise ValueError(f'Movie not found: {title}')
        if len(matches) > 1:
            return matches['popularity'].idxmax()
        return matches.index[0]

    def get_feature_vector(self, idx, segment):
        """
        Extract a specific feature segment vector for a movie.

        Args:
            idx: Movie index in the DataFrame.
            segment: One of 'tfidf', 'topics', 'embedding', 'sentiment', 'genre'.

        Returns:
            1D numpy array of feature values.
        """
        if segment == 'tfidf':
            return self.tfidf_matrix[idx].toarray().flatten()
        elif segment == 'topics':
            return self.lda_matrix[idx]
        elif segment == 'embedding':
            return self.embedding_matrix[idx]
        elif segment == 'sentiment':
            return self.sentiment_matrix[idx]
        elif segment == 'genre':
            return self.genre_matrix[idx]
        else:
            raise ValueError(f'Unknown segment: {segment}')

    def generate_combinations(self, input_titles, max_combinations=50):
        """
        Generate combinatorial assignments of feature segments to input movies.

        Each combination specifies which input movie provides which feature segment.
        Only combinations using at least 3 different input movies are kept.

        Args:
            input_titles: List of movie title strings.
            max_combinations: Maximum number of combinations to return.

        Returns:
            List of dicts mapping segment names to movie titles.
        """
        segments = list(self.feature_segments.keys())
        n_movies = len(input_titles)

        all_combos = list(itertools_product(range(n_movies), repeat=len(segments)))
        filtered = [c for c in all_combos if len(set(c)) >= 3]

        if len(filtered) > max_combinations:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(filtered), size=max_combinations, replace=False)
            filtered = [filtered[i] for i in indices]

        combinations = []
        for combo in filtered:
            assignment = {segments[i]: input_titles[combo[i]] for i in range(len(segments))}
            combinations.append(assignment)

        return combinations

    def build_hybrid_profile(self, combination):
        """
        Build a hybrid feature profile from a combination assignment.

        Args:
            combination: Dict mapping segment names to movie titles.

        Returns:
            Dict mapping segment names to weighted feature vectors.
        """
        profile_parts = {}
        for segment, movie_title in combination.items():
            idx = self.get_movie_index(movie_title)
            vec = self.get_feature_vector(idx, segment)
            weight = self.feature_segments[segment]['weight']
            profile_parts[segment] = vec * weight
        return profile_parts

    def compute_similarity(self, profile_parts, top_n=10, exclude_titles=None):
        """
        [DEPRECATED] Standard similarity computation. 
        Use compute_similarity_optimized for better performance.
        """
        exclude_titles = set(t.lower() for t in (exclude_titles or []))

        total_sim = np.zeros(len(self.df))
        segment_sims = {}

        for segment, weighted_vec in profile_parts.items():
            weight = self.feature_segments[segment]['weight']
            raw_vec = weighted_vec / weight
            matrix = self._matrices[segment]
            # LARGE MATRIX OPERATION
            sim = cosine_similarity(raw_vec.reshape(1, -1), matrix)[0]
            segment_sims[segment] = sim
            total_sim += sim * weight

        if exclude_titles:
            mask = np.isin(self._lowercase_titles, list(exclude_titles))
            total_sim[mask] = -1

        top_indices = total_sim.argsort()[-top_n:][::-1]

        results = []
        for idx in top_indices:
            if total_sim[idx] < 0:
                continue
            result = {
                'index': idx,
                'title': self.df.iloc[idx]['title'],
                'composite_score': total_sim[idx],
                'vote_average': self.df.iloc[idx]['vote_average'],
                'genres': self.df.iloc[idx]['genres'],
                'year': self.df.iloc[idx]['release_year'],
                'overview': str(self.df.iloc[idx]['overview'])[:200],
                'segment_scores': {seg: segment_sims[seg][idx] for seg in self.feature_segments},
            }
            results.append(result)

        return results

    def compute_similarity_optimized(self, combination, precomputed_scores, top_n=10, exclude_titles=None):
        """
        Optimized similarity computation using pre-calculated scores.
        
        Args:
            combination: Dict mapping segment -> movie_title.
            precomputed_scores: Dict mapping segment -> movie_title -> similarity_array.
        """
        exclude_titles = set(t.lower() for t in (exclude_titles or []))
        total_sim = np.zeros(len(self.df))
        
        segment_results = {}
        for segment, movie_title in combination.items():
            weight = self.feature_segments[segment]['weight']
            sim = precomputed_scores[segment][movie_title]
            segment_results[segment] = sim
            total_sim += sim * weight

        # Vectorized Filter
        if exclude_titles:
            mask = np.isin(self._lowercase_titles, list(exclude_titles))
            total_sim[mask] = -1

        top_indices = total_sim.argsort()[-top_n:][::-1]

        results = []
        for idx in top_indices:
            if total_sim[idx] < 0:
                continue
            result = {
                'index': idx,
                'title': self.df.iloc[idx]['title'],
                'composite_score': total_sim[idx],
                'vote_average': self.df.iloc[idx]['vote_average'],
                'genres': self.df.iloc[idx]['genres'],
                'year': self.df.iloc[idx]['release_year'],
                'overview': str(self.df.iloc[idx]['overview'])[:200],
                'segment_scores': {seg: segment_results[seg][idx] for seg in self.feature_segments},
            }
            results.append(result)

        return results

    def explain_recommendation(self, result, combination):
        """
        Generate a human-readable explanation for a recommendation.

        Args:
            result: Result dict from compute_similarity.
            combination: The combination dict that produced this result.

        Returns:
            Multi-line explanation string.
        """
        lines = []
        lines.append(f'\n{"=" * 60}')
        lines.append(f'  {result["title"]} ({result["year"]})')
        lines.append(f'    Rating: {result["vote_average"]:.1f}/10 | Genres: {result["genres"]}')
        lines.append(f'    Composite Score: {result["composite_score"]:.4f}')
        lines.append(f'    Overview: {result["overview"]}...')
        lines.append(f'\n    Segment Breakdown:')

        for seg in self.feature_segments:
            source = combination[seg]
            score = result['segment_scores'][seg]
            weight = self.feature_segments[seg]['weight']
            desc = self.feature_segments[seg].get('description', seg)
            bar = '#' * int(score * 30)
            lines.append(f'      {seg:12s} ({desc}) <- {source}')
            lines.append(f'      {"":12s} Similarity: {score:.3f} x weight {weight:.2f} = {score*weight:.3f}  {bar}')

        weighted_scores = {
            seg: result['segment_scores'][seg] * self.feature_segments[seg]['weight']
            for seg in self.feature_segments
        }
        top_seg = max(weighted_scores, key=weighted_scores.get)
        desc = self.feature_segments[top_seg].get('description', top_seg)
        lines.append(f'\n    -> Primary match via "{top_seg}" ({desc}) from {combination[top_seg]}')

        return '\n'.join(lines)

    def get_web_explanation(self, combination):
        """
        Generate a concise web-friendly explanation of the hybrid components.
        
        Example: "Combines the thematic content from Inception, the emotional tone from Interstellar..."
        """
        parts = []
        # Group segments by movie title to make the sentence more readable
        movie_to_segments = {}
        for seg, title in combination.items():
            desc = self.feature_segments[seg].get('description', seg).lower()
            movie_to_segments.setdefault(title, []).append(desc)
            
        for title, descs in movie_to_segments.items():
            if len(descs) > 1:
                seg_str = " & ".join([", ".join(descs[:-1]), descs[-1]]) if len(descs) > 1 else descs[0]
            else:
                seg_str = descs[0]
            parts.append(f"the {seg_str} from {title}")
            
        return "FUSE engineered this recommendation by unifying " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."

    # ── High-Level API ──────────────────────────────────────────────────

    def recommend(self, input_titles, top_n=10, max_combinations=30,
                  quality_weight=0.1, verbose=True):
        """
        FUSE Recommendation Pipeline.

        Takes a list of input movie titles, generates hybrid combinations,
        finds recommendations for each, and returns the best results with
        explanations.

        Args:
            input_titles: List of 2–5 movie title strings.
            top_n: Number of final recommendations to return.
            max_combinations: Maximum hybrid profiles to evaluate.
            quality_weight: How much to weight vote_average in final ranking.
            verbose: Whether to print progress information.

        Returns:
            List of (result_dict, combination_dict, explanation_str) tuples,
            sorted by final_score descending.
        """
        if verbose:
            print(f'\n{"=" * 60}')
            print(f'FUSE Engine — Input Movies:')
            for t in input_titles:
                idx = self.get_movie_index(t)
                row = self.df.iloc[idx]
                print(f'  * {row["title"]} ({row["release_year"]}) — {row["genres"]}')
            print(f'{"=" * 60}')

        combinations = self.generate_combinations(input_titles, max_combinations=max_combinations)
        if verbose:
            print(f'\nGenerated {len(combinations)} hybrid combinations.')

        # PRE-CALCULATION PHASE (Optimization)
        if verbose:
            print("Pre-calculating segment similarities...")
            
        precomputed_scores = {seg: {} for seg in self.feature_segments}
        for seg in self.feature_segments:
            matrix = self._matrices[seg]
            for title in input_titles:
                try:
                    idx = self.get_movie_index(title)
                    vec = self.get_feature_vector(idx, seg)
                    # This is the heavy operation, done once per input movie per segment
                    sim = cosine_similarity(vec.reshape(1, -1), matrix)[0]
                    precomputed_scores[seg][title] = sim
                except Exception as e:
                    if verbose: print(f"Warning: could not process {title} for segment {seg}: {e}")

        all_results = []
        for combo in combinations:
            # profile = self.build_hybrid_profile(combo) # No longer needed for similarity
            results = self.compute_similarity_optimized(combo, precomputed_scores, top_n=5, exclude_titles=input_titles)
            for r in results:
                r['final_score'] = r['composite_score'] + quality_weight * (r['vote_average'] / 10.0)
                all_results.append((r, combo))

        # Deduplicate: keep the best score for each movie
        best_per_movie = {}
        for result, combo in all_results:
            title = result['title']
            if title not in best_per_movie or result['final_score'] > best_per_movie[title][0]['final_score']:
                best_per_movie[title] = (result, combo)

        ranked = sorted(best_per_movie.values(), key=lambda x: x[0]['final_score'], reverse=True)
        top_results = ranked[:top_n]

        if verbose:
            print(f'Unique movies found: {len(best_per_movie)}')
            print(f'Returning top {top_n} recommendations.\n')

        output = []
        for result, combo in top_results:
            explanation = self.explain_recommendation(result, combo)
            output.append((result, combo, explanation))

        return output

    def baseline_recommend(self, input_titles, top_n=10):
        """
        Standard content-based filtering baseline.

        Averages all feature vectors of input movies and finds the most
        similar real movies. Used as a comparison to FUSE's combinatorial
        approach.

        Args:
            input_titles: List of movie title strings.
            top_n: Number of recommendations to return.

        Returns:
            List of result dicts with title, score, genres, year, etc.
        """
        indices = [self.get_movie_index(t) for t in input_titles]
        exclude = set(t.lower() for t in input_titles)

        avg_vecs = {
            'tfidf': np.asarray(sp.vstack([self.tfidf_matrix[i] for i in indices]).mean(axis=0)).flatten(),
            'topics': np.mean([self.lda_matrix[i] for i in indices], axis=0),
            'embedding': np.mean([self.embedding_matrix[i] for i in indices], axis=0),
            'sentiment': np.mean([self.sentiment_matrix[i] for i in indices], axis=0),
            'genre': np.mean([self.genre_matrix[i] for i in indices], axis=0),
        }

        total_sim = np.zeros(len(self.df))
        for seg in self.feature_segments:
            matrix = self._matrices[seg]
            sim = cosine_similarity(avg_vecs[seg].reshape(1, -1), matrix)[0]
            total_sim += sim * self.feature_segments[seg]['weight']

        if exclude:
            mask = np.isin(self._lowercase_titles, list(exclude))
            total_sim[mask] = -1

        results = []
        for idx in total_sim.argsort()[-top_n:][::-1]:
            if total_sim[idx] >= 0:
                results.append({
                    'title': self.df.iloc[idx]['title'],
                    'score': total_sim[idx],
                    'genres': self.df.iloc[idx]['genres'],
                    'year': self.df.iloc[idx]['release_year'],
                    'vote_avg': self.df.iloc[idx]['vote_average'],
                    'index': idx,
                })

        return results

    # ── Utilities ───────────────────────────────────────────────────────

    @property
    def n_movies(self):
        """Number of movies in the dataset."""
        return len(self.df)

    @property
    def segment_names(self):
        """List of feature segment names."""
        return list(self.feature_segments.keys())

    def __repr__(self):
        return (
            f'FUSEEngine(movies={self.n_movies}, '
            f'segments={self.segment_names})'
        )


# ── CLI / Quick Test ────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    
    pkl_path = os.path.join('Artifacts', 'models', 'fuse_engine.pkl')
    artifacts_path = os.path.join('Artifacts', 'features')

    if os.path.exists(pkl_path):
        print(f'Loading FUSE engine from {pkl_path}...')
        engine = FUSEEngine.from_pkl(pkl_path)
    elif os.path.exists(artifacts_path):
        print(f'Loading FUSE engine from {artifacts_path}...')
        engine = FUSEEngine.from_artifacts(artifacts_path)
    else:
        print('Error: No data found. Run Notebook 03 and 04 first.')
        sys.exit(1)

    print(f'Engine loaded: {engine}')
    print()

    # Quick test with default movies
    test_titles = ['Inception', 'Interstellar', 'The Dark Knight']
    print(f'Testing with: {test_titles}')

    results = engine.recommend(test_titles, top_n=5)
    for result, combo, explanation in results:
        print(explanation)
    
