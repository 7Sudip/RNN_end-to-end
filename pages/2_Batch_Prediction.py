import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

from src.preprocess import preprocess_text, text_to_sequence
from src.config import MODEL_PATH

# Load model (cached)
@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

model = load_model_cached()

st.title("Batch Sentiment Prediction")

st.write("Upload a CSV file containing text data for sentiment analysis.")

# Upload file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview of your")
    st.dataframe(df.head())

    # Columns name in small case
    df.columns = df.columns.str.lower()

    # Define our allowed/required columns
    required_col = 'text'
    index_col = 's.n'

    if required_col in df.columns:
        # Check if the file contains ONLY 'text' or 'text' + 's.n'
        allowed_cols = {required_col, index_col}
        current_cols = set(df.columns)
        
        # Check if there are any "extra" columns (like sentiment or label)
        extra_cols = current_cols - allowed_cols
        
        if not extra_cols:
            # If it's clean, we keep only the 'text' column for processing
            df = df[[required_col]]
            
        else:
            # Error if columns like 'sentiment' or 'label' are found
            st.error(f"CSV contains unauthorized columns: {', '.join(extra_cols)}")
    else:
        st.error("Missing required 'text' column!")
        
    if st.button("Run Batch Prediction"):

        results = []
        confidence_score = []
        progress_bar = st.progress(0)

        total = len(df)

        for i, text in enumerate(df["text"]):

            if isinstance(text, str):
                sequence = text_to_sequence(preprocess_text(text))
                sequence = np.vstack(sequence) # For Fast process 
                prediction = model.predict(sequence)[0][0]
                confidence_score.append(int(prediction*100))
                label = "Positive" if prediction > 0.5 else "Negative"
            else:
                label = "Invalid"

            results.append(label)

            progress_bar.progress((i + 1) / total)

        # Add results
        df.rename(columns={'text': 'Text'}, inplace=True)
        df["Sentiment"] = results
        df['Confidence_Score(%)'] = confidence_score

        st.success("Batch prediction completed!")


        st.subheader("Results")
        st.dataframe(df)

        # Download button
        df.drop('Confidence_Score(%)',axis=1,inplace=True) # delete this column
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="batch_predictions.csv",
            mime="text/csv"
        )