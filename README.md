# 🎬 Movie Recommendation Chatbot

A conversational movie recommendation chatbot built with NLP techniques.

**Course:** Natural Language Processing and Text Mining, Spring 2026 — NTUT CSIE  
**Team:** 鐘美萍 (112590041), 李岱磬 (112590036)

## Features
- TF-IDF + Cosine Similarity for recommendations
- Named Entity Recognition (spaCy) to detect movie references
- Sentiment Analysis (VADER) to filter low-quality tags
- Year and genre filters
- Streamlit web interface

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/USERNAME/movie-chatbot.git
cd movie-chatbot
```

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn nltk streamlit vaderSentiment spacy
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

### 3. Download dataset
```bash
mkdir -p data
cd data
curl -O https://files.grouplogs.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip
cd ..
```

### 4. Run
```bash
streamlit run app.py
```