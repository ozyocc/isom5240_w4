import streamlit as st
from transformers import pipeline

st.set_page_config(layout="centered", page_title="Sentiment Analysis App")
st.title("Sentiment Analysis with Hugging Face Transformers")

# Initialize the sentiment analysis pipeline
# This will download the model if it's not already cached
@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis")

sentiment_pipeline = get_sentiment_pipeline()

user_input = st.text_area("Enter text for sentiment analysis:", "Deep Learning (DL) represents a highly promising approach to developing applications in Artificial Intelligence (AI).")

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing sentiment..."):
            result = sentiment_pipeline(user_input)
            label = result[0]["label"]
            score = result[0]["score"]

            st.success("Analysis Complete!")
            st.write(f"**Sentiment:** {label}")
            st.write(f"**Confidence Score:** {score:.4f}")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---<br>_This app uses the Hugging Face `transformers` library for sentiment analysis._", unsafe_allow_html=True)
