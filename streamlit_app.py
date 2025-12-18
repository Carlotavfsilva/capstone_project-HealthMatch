import streamlit as st
from PIL import Image
from google import genai
import os
import base64
from langfuse import Langfuse, observe
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymongo import MongoClient



# Fetching secrets from the Streamlit secrets file
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]["key"]
MONGODB_URI = st.secrets["MONGODB_URI"]["uri"]
LANGFUSE_PUBLIC_KEY = st.secrets["LANGFUSE"]["public_key"]
LANGFUSE_SECRET_KEY = st.secrets["LANGFUSE"]["secret_key"]
LANGFUSE_HOST = st.secrets["LANGFUSE"]["host"]



# --- Streamlit App Page Setup ---
st.set_page_config(
    page_title="HealthMatch - Your Personalized Health Assistant",
    page_icon="🤖",
    layout="centered"
)


# Initialize session state for chat history
if "session_id" not in st.session_state:
    st.session_state.session_id = os.urandom(16).hex()

# Initialize Langfuse client once
# --- Initialize Langfuse client only once ---
def initialize_langfuse():
    """Initializes the Langfuse client."""
    try:
        langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
        st.session_state.langfuse_client = langfuse_client
        return True
    except KeyError as e:
        st.error(f"Langfuse key not found in secrets: {e}. Please check `.streamlit/secrets.toml`.")
        return False
    except Exception as e:
        st.error(f"Error initializing Langfuse client: {e}")
        return False
    

# --- Check if Langfuse client is already initialized ---
if "langfuse_client" not in st.session_state:
    # Try to initialize Langfuse client if it doesn't exist
    if not initialize_langfuse():
        st.stop()  # If initialization fails, stop execution
# If Langfuse client is initialized successfully, it will be available in session state.


# --- Initialize LangChain with Google Generative AI Embeddings ---
def initialize_embeddings():
    os.environ['GEMINI_API_KEY'] = GOOGLE_API_KEY
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return embeddings

# --- Function to generate embeddings using LangChain and Google Gemini ---
def generate_gemini_embedding(text):
    embeddings = initialize_embeddings()
    embedding = embeddings.embed_query(text)
    return embedding

# --- Connect to MongoDB ---
def connect_to_mongo():
    client = MongoClient(MONGODB_URI)
    db = client["HealthMatch"]
    collection = db["Health Dictionary"]
    return collection

# --- Function to retrieve relevant pathologies based on user input ---
def search_pathology_documents(user_query, k=5):
    collection = connect_to_mongo()

    # 1. Embed user query
    query_embedding = generate_gemini_embedding(user_query)

    # 2. MongoDB vector search
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",   # EXACT name from Atlas
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": k
            }
        },
        {
            "$project": {
                "_id": 0,
                "title": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    results = list(collection.aggregate(pipeline))
    return results


# --- Function to reset the chat session ---
def reset_chat():
    """Forces the chat to be re-initialized when settings change."""    
    if "langfuse_client" in st.session_state and st.session_state.langfuse_client:
        st.session_state.langfuse_client.flush() 

    st.session_state["chat"] = None
    st.session_state["messages"] = []
    st.session_state["session_id"] = os.urandom(16).hex() 


# --- Function to generate a response using Langfuse and Google Gemini ---
@observe()
def generate_response_with_langfuse(langfuse_client, user_input, model_name, system_instr, user_id, api_key):
    """Generates a response using Langfuse and Google Gemini API."""

    try:
        client = genai.Client(api_key=api_key)  # Initialize with Google Gemini API key
        response = client.models.generate_content(
            model=model_name,
            contents=user_input,
            config={
                "system_instruction": system_instr,
                "temperature": st.session_state.temperature
            }
        )

        response_text = response.text
        return response_text
    except Exception as e:
        raise e
    

def submit_message():
    user_query = st.session_state.get("chat_input", "").strip()

    if not user_query:
        return

    # 1. Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    try:
        # 2. Retrieve relevant documents (RAG)
        docs = search_pathology_documents(user_query)

        if docs:
            context = " ".join(doc["text"] for doc in docs)
        else:
            context = "No relevant medical information found."

        # 3. Generate response with Langfuse + Gemini
        response = generate_response_with_langfuse(
            langfuse_client=st.session_state.langfuse_client,
            user_input=f"Context:\n{context}\n\nQuestion:\n{user_query}",
            model_name="gemini-2.5-flash",
            system_instr=st.session_state.system_instruction,
            user_id=st.session_state.session_id,
            api_key=GOOGLE_API_KEY
        )

        # 4. Store assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "An error occurred while generating the response."
        })
        st.error(e)

    # 5. Update search history
    st.session_state.search_history.append(user_query)

    # 6. Clear input and rerun
    st.session_state.chat_input = ""


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize search history
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# --- Search History ---
st.sidebar.header("🕒 Search History")
if st.session_state.search_history:
    for i, search in enumerate(st.session_state.search_history):
        if st.sidebar.button(f"🔍 {search}", key=f"search_{i}"):
            st.session_state["chat_input"] = search
            st.rerun()

# --- Streamlit App Page Setup ---
st.title("Your Personalized Health Assistant")

# Load and display images 
BG_IMAGE_PATH = 'C:/Users/morai/OneDrive/Documentos/3ano/Capstone_Project/Fundo para o Streamlit.png'
LOGO_IMAGE_PATH = 'C:/Users/morai/OneDrive/Documentos/3ano/Capstone_Project/Logo para Streamlit.png'

try:
    with open(BG_IMAGE_PATH, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    st.markdown(f"""<style>.stApp {{ background-image: url("data:image/png;base64,{encoded_string}"); background-size: cover; background-position: center; background-attachment: fixed; }}</style>""", unsafe_allow_html=True)
except FileNotFoundError:
    st.error(f"Background image not found at: {BG_IMAGE_PATH}. Please check the path.")
    st.stop()

try:
    logo = Image.open(LOGO_IMAGE_PATH)
    st.image(logo, use_container_width=True)
except FileNotFoundError:
    st.error(f"Logo image not found at: {LOGO_IMAGE_PATH}. Please check the path.")
    st.stop()

# --- FRONTEND: Display chat history ---
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Input field for the user
st.text_input(
    "Describe your symptoms or ask for advice:",
    key="chat_input",
    on_change=submit_message
)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "system_instruction" not in st.session_state:
        st.session_state.system_instruction = "You are a helpful assistant. Be concise and friendly."

    st.slider(
        "Temperature",
        min_value=0.0, max_value=2.0, value=st.session_state.temperature,
        key="temperature", on_change=reset_chat,
        help="Lower = focused, Higher = creative"
    )
    st.text_area(
        "System Instruction",
        value=st.session_state.system_instruction, height=100,
        key="system_instruction", on_change=reset_chat,
        help="Define how the AI should behave"
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    if st.button("🗑️ Clear Search History", use_container_width=True):
        st.session_state.search_history = []
        st.success("Search history cleared!")

# --- Display Introductory Message ---
st.markdown("""
    👋 **Welcome to Your Personalized Health Assistant!**
    This chatbot can help you find medical services, doctor recommendations, and more!
""")
st.divider()
