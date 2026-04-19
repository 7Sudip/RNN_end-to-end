import streamlit as st

st.set_page_config(page_title="Sentiment App", layout="centered")

st.title("Movie Sentiment Analysis App")
st.write("Welcome! Use the sidebar to navigate.")

st.info("Go to Predict page to analyze sentiment.") 
st.info("Go to Batch_Prediction to CSV File sentiment analyze") 
st.info('Go to Model_info to look the Model information')