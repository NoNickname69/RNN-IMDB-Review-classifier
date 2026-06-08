# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('Simple_RNN_IMDB.keras')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [min(word_index.get(word, 2) + 3, 9999) for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Import streamlit
import streamlit as st

# Page configuration
st.set_page_config(page_title="Movie Sentiment", page_icon="🎬", layout="centered")

#Styling using CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .stTextArea textarea { background-color: #1c1f26; color: white; border-radius: 10px; }
    .stButton>button { background-color: #e50914; color: white; border-radius: 8px; width: 100%; font-size: 16px; }
    .stButton>button:hover { background-color: #b20710; }
    </style>
""", unsafe_allow_html=True)

st.markdown("# Movie Review Sentiment Analysis")
st.markdown("*Powered by a Simple RNN trained on IMDB data*")
st.divider()

with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses a Simple RNN model trained on 25,000 IMDB reviews.")
    st.write("**Vocabulary size:** 10,000 words")
    st.write("**Max review length:** 500 words")

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    if sentiment == 'Positive':
        st.success(f"Sentiment: **{sentiment}**")
    else:
        st.error(f"Sentiment: **{sentiment}**")

    st.metric(label="Confidence Score", value=f"{prediction[0][0]:.2%}")

    st.progress(float(prediction[0][0]))
else:
    st.write('Please enter a movie review.')
