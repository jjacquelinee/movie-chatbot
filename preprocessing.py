import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Lowercase, hapus karakter aneh, stopwords, lalu lemmatize."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def load_and_prepare(data_path='data/ml-latest-small'):
    movies = pd.read_csv(f'{data_path}/movies.csv')
    tags   = pd.read_csv(f'{data_path}/tags.csv')

    # Gabungkan semua tag per movie jadi satu string
    tags_grouped = tags.groupby('movieId')['tag'].apply(
        lambda x: ' '.join(x.astype(str))
    ).reset_index()
    tags_grouped.columns = ['movieId', 'all_tags']

    # Merge movies + tags
    df = movies.merge(tags_grouped, on='movieId', how='left')
    df['all_tags'] = df['all_tags'].fillna('')

    # Gabung genre + tags jadi satu kolom "content"
    df['genres_clean'] = df['genres'].str.replace('|', ' ', regex=False)
    df['content'] = df['genres_clean'] + ' ' + df['all_tags']
    df['content_clean'] = df['content'].apply(clean_text)
    df['sentiment'] = df['all_tags'].apply(get_sentiment_score)

    # Ekstrak tahun dari judul, e.g. "Toy Story (1995)"
    df['year'] = df['title'].str.extract(r'\((\d{4})\)')
    df['title_clean'] = df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

    return df

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    """Return compound sentiment score: +1 (positive) to -1 (negative)."""
    if not text or text.strip() == '':
        return 0
    return analyzer.polarity_scores(text)['compound']

if __name__ == '__main__':
    df = load_and_prepare()
    print(df[['title_clean', 'year', 'genres', 'content_clean']].head(10))
    print(f'\nTotal movies: {len(df)}')