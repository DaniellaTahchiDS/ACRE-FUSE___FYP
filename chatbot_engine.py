import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

class ChatbotEngine:
    def __init__(self, fuse_engine, movies_df):
        """
        Initialize Chatbot with existing FUSE engine and movies dataframe.
        """
        self.fuse = fuse_engine
        self.df = movies_df
        self.chat_histories = {}
        
        # Initialize Gemini
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        self.use_gemini = False
        
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                self.use_gemini = True
                self.model = "gemma-3-27b-it"
                print("✅ Chatbot: Gemini API connected.")
            except ImportError:
                print("⚠️ Chatbot: google-genai not installed. Running in LOCAL mode.")
            except Exception as e:
                print(f"⚠️ Chatbot: Gemini initialization failed: {e}")
        else:
            print("⚠️ Chatbot: No GEMINI_API_KEY found. Running in LOCAL mode.")

        self.system_prompt = """You are a world-class movie expert assistant with encyclopedic knowledge of cinema.
YOU KNOW EVERYTHING ABOUT MOVIES INCLUDING:
• Box office, budgets, cast, crew, awards, and trivia.
• Critical reception and historical legacy.

RESPONSE RULES:
1. ALWAYS GIVE SPECIFIC FACTS AND NUMBERS.
2. ALWAYS GIVE SPECIFIC NAMES.
3. FORMATTING: Use HTML tags (<strong>, <em>, <p>, <ul>, <li>, <h3>). Do NOT use markdown.
4. If you have database info provided, use it for ratings and genres.
5. Be enthusiastic, factual, and confident."""

    def process(self, user_input, session_id="default"):
        if not user_input or not user_input.strip():
            return "<p>Please type a message! 😊</p>"

        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []

        # Retrieve Context from our Database
        movie_context = self._retrieve_context(user_input)

        if self.use_gemini:
            response = self._generate_gemini_response(user_input, movie_context, self.chat_histories[session_id])
        else:
            response = self._generate_local_response(user_input, movie_context)

        # Store History
        self.chat_histories[session_id].append({"user": user_input, "bot": response})
        if len(self.chat_histories[session_id]) > 10:
            self.chat_histories[session_id] = self.chat_histories[session_id][-10:]

        return response

    def _retrieve_context(self, user_input):
        """Perform semantic and keyword search to find relevant movies."""
        query = user_input.lower()
        
        # Simple Title Search
        title_matches = self.df[self.df['title'].str.lower().str.contains(query, na=False)].head(5)
        
        # Semantic Search using FUSE embeddings
        semantic_matches = pd.DataFrame()
        if self.fuse and hasattr(self.fuse, 'embedding_matrix'):
            try:
                # Use simplified embedding from query if we had a transformer, 
                # but for simplicity we'll just check if query contains a movie we know
                # and get recommendations for it.
                words = query.split()
                best_match = None
                for word in words:
                    if len(word) > 3:
                        match = self.df[self.df['title'].str.lower() == word].head(1)
                        if not match.empty:
                            best_match = match
                            break
                
                if best_match is not None:
                    # Get similar from FUSE
                    sim_recs = self.fuse.baseline_recommend([best_match.iloc[0]['title']], top_n=5)
                    # Convert list of dicts to DF
                    semantic_matches = pd.DataFrame(sim_recs)
            except:
                pass

        # Combine
        combined = pd.concat([title_matches, semantic_matches]).drop_duplicates(subset=['title']).head(10)
        
        # Format for LLM
        context_parts = []
        for _, movie in combined.iterrows():
            context_parts.append(f"MOVIE: {movie.get('title')}\n"
                                 f"  Year: {movie.get('release_year', 'N/A')}\n"
                                 f"  Rating: {movie.get('vote_average', 'N/A')}/10\n"
                                 f"  Genres: {movie.get('genres', 'N/A')}\n"
                                 f"  Overview: {movie.get('overview', 'N/A')}")
        
        return "\n---\n".join(context_parts) if context_parts else "No specific database info found."

    def _generate_gemini_response(self, user_message, movie_context, history):
        from google import genai
        
        prompt = f"""{self.system_prompt}
        
DATABASE CONTEXT:
{movie_context}

USER MESSAGE: {user_message}
"""
        contents = []
        for h in history:
            contents.append({"role": "user", "parts": [{"text": h["user"]}]})
            contents.append({"role": "model", "parts": [{"text": h["bot"]}]})
        
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config={"temperature": 0.5}
            )
            text = response.text.replace("```html", "").replace("```", "").strip()
            return text
        except Exception as e:
            print(f"Gemini Error: {e}")
            return self._generate_local_response(user_message, movie_context)

    def _generate_local_response(self, user_message, movie_context):
        if "MOVIE:" not in movie_context:
            return "<p>I'm sorry, I don't have information on that. Could you be more specific? I'm better at talking about movies! 🎬</p>"
        
        html = f"<p>I found some information related to your request:</p><ul>"
        for block in movie_context.split("---"):
            lines = block.strip().split("\n")
            title = lines[0].replace("MOVIE:", "").strip()
            html += f"<li><strong>{title}</strong></li>"
        html += "</ul><p>I am currently running in <em>Local Mode</em>. Add a Gemini API key to unlock full conversational expert knowledge! 🚀</p>"
        return html
