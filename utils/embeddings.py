import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def generate_gemini_embedding(text):
    os.environ["GEMINI_API_KEY"] = st.secrets["GOOGLE_API_KEY"]["key"]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return embeddings.embed_query(text)
