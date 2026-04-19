import streamlit as st
import time
from tensorflow.keras.models import load_model

from src.preprocess import preprocess_text, text_to_sequence
from src.config import MODEL_PATH

# Load model (cached)
@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

model = load_model_cached()

# UI
st.title("Sentiment Prediction")

st.markdown(
    """
    <style>
    div[data-baseweb="textarea"] textarea {
        resize: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

user_input = st.text_area("Type your review here:", height=150)

if st.button("Analyze Sentiment"):
    if user_input:

        # Preprocess
        sequence = text_to_sequence(preprocess_text(user_input))

        # Predict
        prediction = model.predict(sequence)[0][0]
        final_percent = int(prediction * 100)
        label = "Positive" if prediction > 0.5 else "Negative"

        # Animation
        status_text = st.empty()
        progress_bar = st.progress(0)

        for i in range(final_percent + 1):
            time.sleep(0.01)
            progress_bar.progress(i)
            status_text.metric("Analyzing Model Weights...", f"{i}%")

        # Result
        if label == "Positive":
            st.markdown(
                """
                <div style="
                    background-color: #d4edda;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                ">
                    <h2 style="color: #155724; margin: 0;">Positive 🙂</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.balloons()

        else:
            st.markdown(
                """
                <div style="
                    background-color: #f8d7da;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                ">
                    <h2 style="color: #721c24; margin: 0;">Negative 😠</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.snow()

    else:
        st.warning("Please enter text to analyze.")