import streamlit as st
from recommender import MovieRecommender

st.set_page_config(page_title="Movie Chatbot", page_icon="🎬", layout="centered")

@st.cache_resource
def load_recommender():
    return MovieRecommender()

rec = load_recommender()

# --- UI ---
st.title("🎬 Movie Recommendation Chatbot")
st.caption("Tell me what kind of movie you're in the mood for!")

# Contoh quick buttons
st.markdown("**Try these:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("😱 Something scary"):
        st.session_state.quick = "something scary and thrilling"
with col2:
    if st.button("😂 Make me laugh"):
        st.session_state.quick = "funny comedy movie"
with col3:
    if st.button("🚀 Sci-fi adventure"):
        st.session_state.quick = "sci-fi space adventure"

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! What kind of movie are you in the mood for today?"}
    ]
if "history" not in st.session_state:
    st.session_state.history = []

# Tampilkan chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle quick button
if "quick" in st.session_state:
    user_input = st.session_state.pop("quick")
else:
    user_input = st.chat_input("E.g. 'I want a mind-bending sci-fi from the 2000s'")

# Proses input
if user_input:
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Gabungkan context
    if st.session_state.history:
        combined = st.session_state.history[-1] + ' ' + user_input
    else:
        combined = user_input

    # Get rekomendasi
    results, query = rec.recommend(combined)

    if results is None:
        response = f"😕 {query}"
    else:
        lines = ["🍿 Here are my recommendations:\n"]
        for i, row in enumerate(results.itertuples(), 1):
            genres = row.genres.replace('|', ', ')
            year = row.year if row.year else '?'
            lines.append(f"**{i}. {row.title_clean}** ({year})")
            lines.append(f"&nbsp;&nbsp;&nbsp;🎭 {genres}\n")
        response = '\n'.join(lines)
        st.session_state.history.append(combined)

    # Tampilkan respons chatbot
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)