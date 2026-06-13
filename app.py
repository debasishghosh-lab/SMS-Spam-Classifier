import streamlit as st
import pickle
import re

import nltk

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load Model
model = pickle.load(
    open("spam_model.pkl", "rb")
)

tfidf = pickle.load(
    open("tfidf_vectorizer.pkl", "rb")
)

# NLP Tools
lemmatizer = WordNetLemmatizer()

stop_words = set(
    stopwords.words("english")
)

# Page Configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Title
st.title("📩 SMS Spam Detection System")

st.write(
    "Detect whether an SMS message is Spam or Ham using Natural Language Processing."
)

# Input
message = st.text_area(
    "Enter SMS Message"
)

# Prediction Function
def preprocess_text(text):

    review = re.sub(
        r'[^a-zA-Z]',
        ' ',
        text
    )

    review = review.lower()

    review = review.split()

    review = [
        lemmatizer.lemmatize(word)
        for word in review
        if word not in stop_words
    ]

    review = ' '.join(review)

    return review


# Button
if st.button("Predict"):

    processed_text = preprocess_text(
        message
    )

    vector = tfidf.transform(
        [processed_text]
    )

    prediction = model.predict(
        vector
    )[0]

    if prediction == 1:

        st.error(
            "🚨 Spam Message Detected"
        )

    else:

        st.success(
            "✅ Legitimate Ham Message"
        )