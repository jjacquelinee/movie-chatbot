# Movie Recommendation Chatbot

A conversational movie recommendation chatbot built with NLP techniques.

**Course:** NLP & Text Mining, Spring 2026 — NTUT CSIE

## Features
- TF-IDF + Cosine Similarity
- Word2Vec hybrid scoring (gensim)
- Named Entity Recognition (spaCy)
- Sentiment Analysis (VADER)
- Query Expansion (WordNet)
- Year & genre filters
- Streamlit web UI

## Setup
### 1. Requirements
Python 3.11 is required (gensim not yet supported on 3.14+)

### 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

### 3. Download dataset
mkdir -p data && cd data
curl -O https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip && cd ..

### 4. Train Word2Vec
python word2vec_model.py

### 5. Run
streamlit run app.py
