import numpy as np
from gensim.models import Word2Vec
from preprocessing import load_and_prepare, clean_text

def train_word2vec(df):
    """Train Word2Vec dari tag + genre corpus."""
    sentences = [text.split() for text in df['content_clean'] if text.strip()]
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=10
    )
    model.save("data/word2vec_movies.model")
    print("Word2Vec model trained and saved!")
    return model

def load_word2vec():
    """Load model yang sudah di-train."""
    return Word2Vec.load("data/word2vec_movies.model")

def get_query_vector(query, model):
    """Ubah query string jadi rata-rata vektor Word2Vec."""
    words = clean_text(query).split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def get_movie_vectors(df, model):
    """Buat vektor untuk setiap film."""
    vectors = []
    for text in df['content_clean']:
        words = text.split()
        word_vecs = [model.wv[w] for w in words if w in model.wv]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)

if __name__ == '__main__':
    df = load_and_prepare()
    model = train_word2vec(df)
    movie_vectors = get_movie_vectors(df, model)

    print("\nWords similar to 'horror':")
    print(model.wv.most_similar('horror', topn=5))

    print("\nWords similar to 'funny':")
    print(model.wv.most_similar('funny', topn=5))