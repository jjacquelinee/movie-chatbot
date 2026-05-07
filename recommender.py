import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from preprocessing import load_and_prepare, clean_text

# Genre mapping — biar user bisa tulis "scary" → "horror"
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
        self.df = load_and_prepare()
        self._build_tfidf()
        print(f"Ready! {len(self.df)} movies loaded.")

    def _build_tfidf(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['content_clean'])

    def _parse_user_input(self, text):
        """Extract genre/mood, year filter, dan exclude dari input user."""
        text_lower = text.lower()

        # Map mood → genre
        query_words = []
        for mood, genre in MOOD_GENRE_MAP.items():
            if mood in text_lower:
                query_words.append(genre)

        # Ambil juga kata langsung dari input
        query_words.append(clean_text(text))
        query = ' '.join(query_words)

        # Cek filter tahun, e.g. "after 2010", "from 2000s"
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

        # Cek exclude genre
        exclude = []
        for genre in ['horror', 'romance', 'comedy', 'animation', 'documentary']:
            if f'no {genre}' in text_lower or f'not {genre}' in text_lower:
                exclude.append(genre)

        return query, year_after, year_before, exclude
    
    def _extract_reference_movie(self, text):
        """Cek apakah user menyebut judul film tertentu pakai NER."""
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['WORK_OF_ART', 'ORG', 'PERSON']:
                # Cari judul yang mirip di dataset
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

        # NER: cek apakah user sebut film tertentu
        ref_movie = self._extract_reference_movie(user_input)
        if ref_movie is not None:
            # Tambahkan genre & tags dari film referensi ke query
            query = query + ' ' + str(ref_movie['content_clean'])

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

        # Filter sentiment — hanya rekomendasikan film dengan tag positif/netral
        filtered = filtered[filtered['sentiment'] >= -0.3]

        if len(filtered) == 0:
            return None, "No movies found with those filters. Try relaxing some conditions!"

        # TF-IDF similarity
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix[filtered.index]).flatten()
        top_idx = sims.argsort()[-top_n:][::-1]

        results = filtered.iloc[top_idx][['title_clean', 'year', 'genres', 'sentiment']].copy()
        results['score'] = sims[top_idx]
        results = results[results['score'] > 0]

        if results.empty:
            return None, "I couldn't find anything matching that. Try different keywords!"

        return results, query