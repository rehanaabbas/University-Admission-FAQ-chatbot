import pandas as pd
import streamlit as st
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="University Admission FAQ Chatbot",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("faq_data.csv")

# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data["processed_question"] = data["question"].apply(preprocess)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

question_vectors = vectorizer.fit_transform(
    data["processed_question"]
)

# -----------------------------
# Response Function
# -----------------------------
def get_response(user_input):

    user_input = preprocess(user_input)

    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )

    max_score = similarity.max()

    if max_score < 0.25:
        return (
            "Sorry, I could not find a relevant admission answer. "
            "Please try asking in a different way."
        )

    index = similarity.argmax()

    return data.iloc[index]["answer"]

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "recommended_questions" not in st.session_state:
    st.session_state.recommended_questions = random.sample(
        data["question"].tolist(),
        min(6, len(data))
    )

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.title("🎓 Admission Assistant")

    st.subheader("📚 Recent Questions")

    user_questions = [
        msg["content"]
        for msg in st.session_state.messages
        if msg["role"] == "user"
    ]

    if len(user_questions) == 0:
        st.info("No questions asked yet")
    else:
        for q in reversed(user_questions[-10:]):
            st.write("•", q)

    st.divider()

    st.metric("Total FAQs", len(data))

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.recommended_questions = random.sample(
            data["question"].tolist(),
            min(6, len(data))
        )
        st.rerun()

# -----------------------------
# Header
# -----------------------------
st.title("🎓 University Admission FAQ Chatbot")

st.caption(
    "Ask questions about admissions, scholarships, hostels, fees, merit lists, enrollment and university policies."
)

# -----------------------------
# Recommended Questions
# -----------------------------
st.subheader("💡 Recommended Questions")

recommended = st.session_state.recommended_questions

cols = st.columns(2)

for i, question in enumerate(recommended):

   if cols[i % 2].button(question, key=f"rec_{i}"):

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    response = get_response(question)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.session_state.recommended_questions = random.sample(
        data["question"].tolist(),
        min(6, len(data))
    )

    st.rerun()

# -----------------------------
# Chat Messages
# -----------------------------
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# Chat Input
# -----------------------------
# -----------------------------
# Chat Input
# -----------------------------
user_question = st.chat_input(
    "Ask an admission-related question..."
)

if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    response = get_response(user_question)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    st.session_state.recommended_questions = random.sample(
        data["question"].tolist(),
        min(6, len(data))
    )

    st.rerun()
    