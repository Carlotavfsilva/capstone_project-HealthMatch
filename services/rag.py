import streamlit as st
from utils.embeddings import generate_gemini_embedding
from utils.mongo import get_health_collection

def search_pathology_documents(user_query, k=5):
    mongo_uri = st.secrets["MONGODB_URI"]["uri"]

    collection = get_health_collection(mongo_uri)
    query_embedding = generate_gemini_embedding(user_query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": k
            }
        },
        {
            "$project": {
                "_id": 0,
                "name of pathology": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    return list(collection.aggregate(pipeline))