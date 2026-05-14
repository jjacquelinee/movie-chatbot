from recommender import MovieRecommender

rec = MovieRecommender()

# Test cases: (query, genre yang diharapkan)
TEST_CASES = [
    ("scary horror movie", "Horror"),
    ("funny comedy", "Comedy"),
    ("romantic love story", "Romance"),
    ("sci-fi space adventure", "Sci-Fi"),
    ("action fight explosive", "Action"),
    ("sad emotional drama", "Drama"),
    ("animated cartoon kids", "Animation"),
    ("mystery detective crime", "Mystery"),
    ("fantasy magic", "Fantasy"),
    ("thriller mind-bending twist", "Thriller"),
]

def precision_at_5(results, expected_genre):
    """Hitung berapa dari 5 rekomendasi yang genrenya sesuai."""
    if results is None or results.empty:
        return 0.0
    relevant = 0
    for _, row in results.iterrows():
        if expected_genre.lower() in row['genres'].lower():
            relevant += 1
    return relevant / min(5, len(results))

def run_evaluation():
    print("=" * 50)
    print("EVALUATION RESULTS — Precision@5")
    print("=" * 50)

    scores = []
    for query, expected_genre in TEST_CASES:
        results, _ = rec.recommend(query, top_n=5)
        score = precision_at_5(results, expected_genre)
        scores.append(score)
        status = "✅" if score >= 0.6 else "⚠️"
        print(f"{status} [{expected_genre:12}] '{query[:35]}' → P@5: {score:.2f}")

    avg = sum(scores) / len(scores)
    print("=" * 50)
    print(f"Average Precision@5: {avg:.2f} ({avg*100:.1f}%)")
    print("=" * 50)

if __name__ == '__main__':
    run_evaluation()