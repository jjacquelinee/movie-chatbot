from nltk.corpus import wordnet
from preprocessing import clean_text

# Kata-kata yang TIDAK perlu di-expand karena sudah spesifik
SKIP_WORDS = {'movie', 'film', 'watch', 'want', 'like', 'good', 
              'great', 'show', 'something', 'one', 'get'}

# Batasi ke sinonim yang benar-benar relevan per domain film
FILM_SYNONYMS = {
    'scary': ['horror', 'frightening', 'terrifying'],
    'funny': ['comedy', 'humorous', 'hilarious'],
    'sad': ['emotional', 'drama', 'tearful'],
    'exciting': ['thrilling', 'action', 'adventure'],
    'romantic': ['romance', 'love'],
    'weird': ['surreal', 'bizarre', 'fantasy'],
    'old': ['classic', 'vintage'],
    'new': ['recent', 'modern'],
    'smart': ['intelligent', 'clever', 'mystery'],
    'dark': ['noir', 'thriller', 'gritty'],
    'epic': ['adventure', 'fantasy', 'war'],
    'animated': ['animation', 'cartoon'],
    'alien': ['sci-fi', 'space'],
    'future': ['sci-fi', 'dystopia'],
    'magic': ['fantasy', 'supernatural'],
}

def get_synonyms(word, max_synonyms=2):
    """Cek custom dict dulu, baru WordNet sebagai fallback."""
    if word in FILM_SYNONYMS:
        return FILM_SYNONYMS[word][:max_synonyms]

    # WordNet fallback — hanya ambil dari synset pertama
    synonyms = set()
    synsets = wordnet.synsets(word)
    if not synsets:
        return []
    # Hanya pakai synset pertama (paling relevan)
    for lemma in synsets[0].lemmas():
        candidate = lemma.name().replace('_', ' ').lower()
        if candidate != word and candidate not in SKIP_WORDS:
            synonyms.add(candidate)
        if len(synonyms) >= max_synonyms:
            break
    return list(synonyms)

def expand_query(query, max_synonyms=2):
    """Perluas query dengan sinonim yang relevan."""
    words = clean_text(query).split()
    expanded = list(words)

    for word in words:
        if word not in SKIP_WORDS:
            synonyms = get_synonyms(word, max_synonyms)
            expanded.extend(synonyms)

    return ' '.join(expanded)

if __name__ == '__main__':
    test_queries = [
        "scary movie",
        "funny romantic comedy",
        "mind-bending science fiction",
    ]
    for q in test_queries:
        expanded = expand_query(q)
        print(f"Original : {q}")
        print(f"Expanded : {expanded}")
        print()