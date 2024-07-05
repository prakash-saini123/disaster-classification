import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Load models and vectorizers
best_xgb_model = joblib.load('best_xgb_model.pkl')
best_nb_model = joblib.load('best_nb_model.pkl')
count_vectorizer = joblib.load('count_vectorizer.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit interface
st.title("Text Classification")

keyword = st.text_input("Keyword")
location = st.text_input("Location")
text = st.text_area("Text")

if st.button("Classify"):
    # Process input
    keyword_vector = count_vectorizer.transform([keyword])
    location_vector = count_vectorizer.transform([location])
    text_vector = tfidf_vectorizer.transform([text])
    
    # Combine vectors using hstack
    input_vector = hstack([keyword_vector, location_vector, text_vector])
    
    # Predict
    xgb_prediction = best_xgb_model.predict(input_vector)
    nb_prediction = best_nb_model.predict(input_vector)
    
    st.write(f"XGBoost Prediction: {'Disaster' if xgb_prediction[0] == 1 else 'Not Disaster'}")
    st.write(f"Naive Bayes Prediction: {'Disaster' if nb_prediction[0] == 1 else 'Not Disaster'}")
