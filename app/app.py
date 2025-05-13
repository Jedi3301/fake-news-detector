import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

# Load model and vectorizer
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# UI
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️‍♂️")
st.title("🕵️‍♂️ Fake News Detector")
st.write("Enter a news headline and tweet count to detect if the news is real or fake.")

# Inputs
title_input = st.text_area("News Title", height=150)
tweet_num = st.number_input("Number of tweets mentioning this article", min_value=0, value=0)

# Predict
if st.button("Detect"):
    if title_input.strip() == "":
        st.warning("Please enter a news title.")
    else:
        # Step 1: Vectorize the title
        title_vector = vectorizer.transform([title_input])

        # Step 2: Add tweet_num as a second feature (reshape it to 2D)
        tweet_feature = np.array([[tweet_num]])

        # Step 3: Combine TF-IDF vector with tweet_num
        final_input = hstack([title_vector, tweet_feature])

        # Step 4: Predict
        prediction = model.predict(final_input)[0]
        confidence = model.predict_proba(final_input)[0][prediction]

        # Step 5: Display result
        if prediction == 1:
            st.success(f"✅ This news is **REAL** with {confidence * 100:.2f}% confidence.")
        else:
            st.error(f"🚨 This news is **FAKE** with {confidence * 100:.2f}% confidence.")
