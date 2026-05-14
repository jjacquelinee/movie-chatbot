from query_expansion import expand_query
import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from preprocessing import load_and_prepare, clean_text

MOOD_GENRE_MAP = {
    'funny': 'comedy', 'hilarious': 'comedy', 'laugh': 'comedy',
    'scary': 'horror', 'terrifying': 'horror', 'creepy': 'horror',
    'romantic': 'romance', 'love': 'romance',
    'action': 'action', 'fight': 'action', 'explosive': 'action',
    'sad': 'drama', 'emotional': 'drama', 'cry': 'drama',
    'animated': 'animation', 'cartoon': 'animation',
    'space': 'sci-fi', 'future': 'sci-fi', 'robot': 'sci-fi',
    'magic': 'fantasy', 'adventure': 'adventure',
    'mystery': 'mystery', 'detective': 'mystery', 'crime': 'crime',
    'mind-bending': 'thriller', 'twist': 'thriller',
}

class MovieRecommender:
    def __init__(self):
        print("Loading movies...")
        self.df = load_and_prepare()      # ← load df DULU
        self._build_tfidf()
        print(f"Ready! {len(self.df)} movies loaded.")

    def _build_tfidf(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['content_clean'])

    def _parse_user_input(self, text):
        text_lower = text.lower()
        query_words = []
        for mood, genre in MOOD_GENRE_MAP.items():
            if mood in text_lower:
                query_words.append(genre)
        query_words.append(clean_text(text))
        query = ' '.join(query_words)

        year_after = None
        year_before = None
        m = re.search(r'after (\d{4})', text_lower)
        if m:
            year_after = int(m.group(1))
        m = re.search(r'before (\d{4})', text_lower)
        if m:
            year_before = int(m.group(1))
        m = re.search(r'(\d{4})s', text_lower)
        if m:
            decade = int(m.group(1))
            year_after = decade
            year_before = decade + 9

        exclude = []
        for genre in ['horror', 'romance', 'comedy', 'animation', 'documentary']:
            if f'no {genre}' in text_lower or f'not {genre}' in text_lower:
                exclude.append(genre)

        return query, year_after, year_before, exclude

    def _extract_reference_movie(self, text):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['WORK_OF_ART', 'ORG', 'PERSON']:
                match = self.df[
                    self.df['title_clean'].str.lower().str.contains(
                        ent.text.lower(), na=False
                    )
                ]
                if not match.empty:
                    return match.iloc[0]
        return None

    def recommend(self, user_input, top_n=5):
        query, year_after, year_before, exclude = self._parse_user_input(user_input)

        ref_movie = self._extract_reference_movie(user_input)
        if ref_movie is not None:
            query = query + ' ' + str(ref_movie['content_clean'])

        # Query Expansion
        query = expand_query(query)

        if not query.strip():
            return None, "I didn't quite understand that. Can you describe what kind of movie you're in the mood for?"

        # Filter dataframe
        filtered = self.df.copy()
        if year_after:
            filtered = filtered[filtered['year'].astype(float) >= year_after]
        if year_before:
            filtered = filtered[filtered['year'].astype(float) <= year_before]
        for ex in exclude:
            filtered = filtered[~filtered['genres'].str.lower().str.contains(ex)]
        filtered = filtered[filtered['sentiment'] >= -0.3]

        if len(filtered) == 0:
            return None, "No movies found with those filters. Try relaxing some conditions!"

        # TF-IDF similarity
        query_vec_tfidf = self.vectorizer.transform([query])
        tfidf_sims = cosine_similarity(query_vec_tfidf, self.tfidf_matrix[filtered.index]).flatten()

        # Hybrid: 60% TF-IDF + 40% Word2Vec
        from word2vec_model import get_query_vector, get_movie_vectors, load_word2vec
        try:
            w2v_model = load_word2vec()
            movie_vectors = get_movie_vectors(filtered, w2v_model)
            query_vec_w2v = get_query_vector(query, w2v_model)
            if query_vec_w2v is not None:
                w2v_sims = cosine_similarity([query_vec_w2v], movie_vectors).flatten()
                sims = 0.6 * tfidf_sims + 0.4 * w2v_sims
            else:
                sims = tfidf_sims
        except:
            sims = tfidf_sims  # fallback ke TF-IDF kalau Word2Vec gagal

        top_idx = sims.argsort()[-top_n:][::-1]
        results = filtered.iloc[top_idx][['title_clean', 'year', 'genres', 'sentiment']].copy()
        results['score'] = sims[top_idx]
        results = results[results['score'] > 0]

        if results.empty:
            return None, "I couldn't find anything matching that. Try different keywords!"

        return results, query