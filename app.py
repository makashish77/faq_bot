import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set up page config
st.set_page_config(page_title="FAQ Assistant", page_icon="🤖", layout="centered")

# App Header
st.title("🤖 AI-Powered FAQ Assistant")
st.write("Ask a question, and our AI will find the most relevant answer from our database.")

# Cache the model so it only loads once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load model with a spinner spinner
with st.spinner("Loading AI Model... Please wait."):
    model = load_model()

# FAQs Database
faq_questions = [
    "What is AI?",
    "What is Machine Learning?",
    "How does Deep Learning work?",
    "What is Python used for?"
]

faq_answers = [
    "AI enables machines to mimic human intelligence.",
    "Machine Learning allows systems to learn from data.",
    "Deep Learning uses neural networks with many layers.",
    "Python is widely used in AI, ML, and web development."
]

# User Input
user_query = st.text_input("Ask your question:", placeholder="e.g., What can I do with Python?")

# Process when the user enters a query
if user_query:
    with st.spinner("Searching for the best answer..."):
        # Generate Embeddings
        faq_embeddings = model.encode(faq_questions)
        query_embedding = model.encode([user_query])

        # Compute Similarity
        scores = cosine_similarity(query_embedding, faq_embeddings)

        # Get Best Answer Index
        best_index = np.argmax(scores)
        best_score = scores[0][best_index]

    # Display Results
    st.divider()
    
    # Optional: Setting a confidence threshold
    if best_score > 0.35:
        st.subheader("📌 Most Relevant FAQ Found")
        st.info(f"**Question:** {faq_questions[best_index]}")
        
        st.subheader("💡 Answer")
        st.success(faq_answers[best_index])
        
        # Display match confidence score
        st.caption(f"Match Confidence: {best_score * 100:.1f}%")
    else:
        st.warning("I couldn't find a highly relevant match for your question in our current FAQs. Try rephrasing or asking something else!")

# Optional Sidebar to view available FAQs
with st.sidebar:
    st.header("📋 Available FAQs")
    for q in faq_questions:
        st.markdown(f"- {q}")