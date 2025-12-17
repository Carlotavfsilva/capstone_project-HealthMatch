import os
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np

# --- Configuration ---
# NOTE: Replace with your actual key and URI.
GOOGLE_API_KEY = "AIzaSyDBZymorU4KO-vh_E3nDlk6-sXnmflPf8U"
MONGODB_URI = "mongodb+srv://moraisgoncalo365_db_user:YzvStBWHLMxeQ8qF@capstone-project.ykhbgmr.mongodb.net/?appName=Capstone-Project"

# --- Initialize LangChain with Google Generative AI Embeddings ---
def initialize_embeddings():
    """Initializes the GoogleGenerativeAIEmbeddings using environment variable."""
    # Set the environment variable which the LangChain class will automatically read
    os.environ['GEMINI_API_KEY'] = GOOGLE_API_KEY 
    
    # Use the specified model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") 
    return embeddings

# --- Function to generate embeddings using LangChain and Google Gemini ---
def generate_gemini_embedding(text):
    """Generates an embedding vector for the given text."""
    embeddings = initialize_embeddings()
    embedding = embeddings.embed_query(text)
    return embedding

# --- Connect to MongoDB ---
def connect_to_mongo():
    """Establishes a connection to MongoDB and returns the collection."""
    try:
        client = MongoClient(MONGODB_URI)
        db = client["HealthMatch"]
        collection = db["Health Dictionary"]
        client.admin.command('ismaster') 
        print("Successfully connected to MongoDB.")
        return collection
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None

# =========================================================================
# === NEW FUNCTION TO START OVER (Delete Existing Embeddings) ===
# =========================================================================
def clear_existing_embeddings():
    """
    Connects to MongoDB and removes the 'embedding' field from ALL documents.
    This effectively allows the process to start over from scratch.
    """
    collection = connect_to_mongo()
    if collection is None:
        return
        
    print("\n--- WARNING: STARTING OVER ---")
    print("Clearing existing 'embedding' fields from ALL documents...")
    
    try:
        # Use update_many with an empty query {} to target all documents
        # Use $unset to remove the specified field
        result = collection.update_many(
            {},
            {"$unset": {"embedding": ""}}
        )
        print(f"Successfully cleared 'embedding' field from {result.modified_count} documents.")
        print("----------------------------\n")
        
    except Exception as e:
        print(f"Error while clearing embeddings: {e}")


# --- Function to resume and store embeddings in MongoDB ---
def store_embeddings_in_mongo():
    """
    Retrieves and processes documents missing the 'embedding' field.
    Since clear_existing_embeddings() was called first, this will process ALL documents.
    """
    collection = connect_to_mongo()
    
    if collection is None:
        return

    print("Starting embedding generation and MongoDB update process...")
    
    # Query for documents where the 'embedding' field DOES NOT exist (which is all of them now)
    resume_query = {"embedding": {"$exists": False}}
    documents_to_process = collection.find(resume_query)

    count = 0
    
    # Generate embeddings and update each document
    for doc in documents_to_process:
        text = doc.get('text', '') 
        
        if text:
            try:
                embedding = generate_gemini_embedding(text)
                
                # Update MongoDB document with the generated embedding
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"embedding": embedding}}
                )
                count += 1
                print(f"Updated document ID: {doc['_id']}. Total new embeddings: {count}")
                
            except Exception as e:
                # Log the error and continue to the next document
                print(f"Error generating embedding for document ID {doc.get('_id', 'N/A')}: {e}")
        else:
             print(f"Skipping document ID: {doc['_id']} - 'text' field is missing or empty.")

    print("---")
    print(f"Embedding generation and MongoDB update process finished. {count} documents processed in this run.")

# --- Trigger the embedding generation process ---
if __name__ == "__main__":
    # 1. DELETE ALL EXISTING EMBEDDINGS
    clear_existing_embeddings() 
    
    # 2. START THE EMBEDDING PROCESS FROM SCRATCH
    store_embeddings_in_mongo()