import os
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()  # reads .env if present

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in environment variables.")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not found in environment variables.")

# ============================================================
# CONSTANTS
# ============================================================

DB_NAME = "HealthMatch"
COLLECTION_NAME = "Health Dictionary"
EMBEDDING_MODEL = "models/text-embedding-004"

# ============================================================
# EMBEDDINGS
# ============================================================

def initialize_embeddings():
    """
    Initializes Google Generative AI embeddings.
    LangChain reads the API key from the environment.
    """
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def generate_embedding(text: str):
    """
    Generates a vector embedding for a given text.
    """
    embeddings = initialize_embeddings()
    return embeddings.embed_query(text)


# ============================================================
# MONGODB
# ============================================================

def connect_to_mongo():
    """
    Connects to MongoDB and returns the collection.
    """
    try:
        client = MongoClient(MONGODB_URI)
        client.admin.command("ping")
        db = client[DB_NAME]
        print("Connected to MongoDB.")
        return db[COLLECTION_NAME]
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return None


# ============================================================
# MAIN LOGIC
# ============================================================

def embed_missing_documents():
    """
    Generates embeddings ONLY for documents that:
    - have a non-empty 'text' field
    - do NOT already have an 'embedding' field
    """
    collection = connect_to_mongo()
    if collection is None:
        return

    query = {
        "text": {"$exists": True, "$ne": ""},
        "embedding": {"$exists": False}
    }

    documents = list(collection.find(query))

    if not documents:
        print("All documents already have embeddings. Nothing to do.")
        return

    print(f"Found {len(documents)} documents missing embeddings.")

    processed = 0

    for doc in documents:
        try:
            embedding = generate_embedding(doc["text"])

            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"embedding": embedding}}
            )

            processed += 1
            print(f"Embedded document {doc['_id']} ({processed})")

        except Exception as e:
            print(f"Failed embedding document {doc['_id']}: {e}")

    print(f"Done. {processed} embeddings added.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    embed_missing_documents()
