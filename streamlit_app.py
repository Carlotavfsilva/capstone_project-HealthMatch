import streamlit as st
from PIL import Image
from google import genai
import os
import base64
from langfuse import Langfuse, observe
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymongo import MongoClient
from google.genai import types
import re
from pathlib import Path


BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"

BG_IMAGE_PATH = ASSETS_DIR / "fundo.png"
LOGO_IMAGE_PATH = ASSETS_DIR / "logo.png"

# Fetching secrets from the Streamlit secrets file
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]["key"]
MONGODB_URI = st.secrets["MONGODB_URI"]["uri"]
LANGFUSE_PUBLIC_KEY = st.secrets["LANGFUSE"]["public_key"]
LANGFUSE_SECRET_KEY = st.secrets["LANGFUSE"]["secret_key"]
LANGFUSE_HOST = st.secrets["LANGFUSE"]["host"]


# Initialize session state for chat history
if "session_id" not in st.session_state:
    st.session_state.session_id = os.urandom(16).hex()

if "last_topic" not in st.session_state:
    st.session_state.last_topic = None

if "last_topic_context" not in st.session_state:
    st.session_state.last_topic_context = None


# --- Streamlit App Page Setup ---
st.set_page_config(
    page_title="HealthMatch - Your Personalized Health Assistant",
    page_icon="🤖",
    layout="centered"
)


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
    try:
        client = MongoClient(MONGODB_URI)
        db = client["HealthMatch"]
        collection = db["Health Dictionary"]
        return collection
    except Exception as e:
        st.error(f"Erro ao conectar ao MongoDB: {e}")
        return None


# --- Function to retrieve relevant pathologies based on user input ---
def search_pathology_documents(user_query, k=5):
    collection = connect_to_mongo()
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


def is_valid_url(url):
    """Check if the URL is valid."""
    return re.match(r'http[s]?://', url) is not None


def analyze_url_content(url):
    """Analyze content from a URL using URL Context tool."""
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Analyze and summarize the content from this URL: {url}",
        config=types.GenerateContentConfig(
            tools=[{"url_context": {}}]  # Certifique-se de que a configuração está correta aqui.
        )
    )
    return response.text


# --- Function to reset the chat session ---
def reset_chat():
    """Forces the chat to be re-initialized when settings change."""
    if "langfuse_client" in st.session_state and st.session_state.langfuse_client:
        st.session_state.langfuse_client.flush()

    st.session_state["chat"] = None
    st.session_state["messages"] = []
    st.session_state["session_id"] = os.urandom(16).hex()
    st.session_state.last_user_input = ""  # Reset the previous user input


# --- Function to generate a response using Langfuse and Google Gemini ---
@observe()
def generate_response_with_langfuse(user_input, model_name, system_instr, user_id, api_key):
    """Generates a response using Langfuse and Google Gemini API."""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=user_input,
            config={
                "system_instruction": system_instr,
                "temperature": st.session_state.temperature
            }
        )
        return response.text
    except Exception as e:
        raise e


def build_system_prompt():
    return f"""
You are a Portuguese medical information assistant.

PRIMARY OBJECTIVE:
Provide clear, accurate and neutral medical information.
Maintain strict topic consistency throughout the conversation.

────────────────────────
TOPIC CONTROL (HIGHEST PRIORITY)
────────────────────────
- A medical topic may already be established.
- If a topic exists, ALL responses MUST remain strictly within that topic.
- You MUST NOT introduce, mention or reference any other disease, condition
  or medical topic unless the user explicitly asks to change the topic.
- This rule overrides all others.

Current locked medical topic:
{st.session_state.last_topic or "No topic has been established yet"}

────────────────────────
USER INTENT HANDLING
────────────────────────
1. If the user provides a URL, they want its content analysed or explained.
2. If the user asks a direct question, answer it.
3. If the user asks a follow-up question, assume it refers to the current topic.

────────────────────────
SYMPTOM HANDLING
────────────────────────
- If the user is describing symptoms AND no diagnosis was explicitly named:
  • Do NOT name diseases.
  • Discuss only general causes or symptom categories.
- Only name a disease if the user explicitly names it.

────────────────────────
CONTEXT USAGE
────────────────────────
- Use provided medical context (RAG or URL-derived) when available.
- If context is insufficient:
  • Stay within the current topic.
  • Do NOT expand to related or similar conditions.

Relevant medical context:
{st.session_state.last_topic_context or "No additional context available"}

────────────────────────
AMBIGUITY HANDLING
────────────────────────
- If the input is ambiguous AND a topic exists, interpret it within that topic.
- If the input is ambiguous AND no topic exists, ask a brief clarification question.

────────────────────────
FORBIDDEN BEHAVIOURS
────────────────────────
- Do NOT change topic implicitly.
- Do NOT compare conditions unless explicitly asked.
- Do NOT introduce examples involving other diseases.
- Do NOT explain internal reasoning or system behaviour.
"""


def submit_message():
    user_query = st.session_state.get("chat_input", "").strip()
    if not user_query:
        return

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    extra_context = None
    is_url_input = is_valid_url(user_query)

    # CASE 1: URL provided
    if is_url_input:
        try:
            extra_context = analyze_url_content(user_query)
        except Exception:
            extra_context = None

    # CASE 2: Normal text → RAG
    else:
        docs = []
        if len(user_query.split()) >= 5:
            docs = search_pathology_documents(user_query)

        if docs and docs[0]["score"] > 0.6 and st.session_state.last_topic is None:
            st.session_state.last_topic = docs[0]["name of pathology"]
            st.session_state.last_topic_context = docs[0]["text"]

        if st.session_state.last_topic is None:
            st.session_state.last_topic = "symptom_description"
            st.session_state.last_topic_context = (
                "The user is describing symptoms. No diagnosis has been established."
            )

    # Build system prompt
    system_prompt = build_system_prompt()

    if st.session_state.last_topic == "symptom_description":
        system_prompt += "\nYou are currently in symptom description mode."

    # URL context handling with safe fallback
    if is_url_input:
        if extra_context:
            system_prompt += f"""

Additional external context (from a URL provided by the user):
{extra_context}
"""
        else:
            system_prompt += """

The URL content could not be accessed directly.
Provide a general explanation of the topic mentioned in the user's message.
"""

    # Generate response
    response = generate_response_with_langfuse(
        user_input=user_query,
        model_name="gemini-2.5-flash",
        system_instr=system_prompt,
        user_id=st.session_state.session_id,
        api_key=GOOGLE_API_KEY
    )

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.session_state.search_history.append(user_query)
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
st.title("Your Health, Your Care, Our Chatbot")

# Background
if BG_IMAGE_PATH.exists():
    with open(BG_IMAGE_PATH, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Logo
if LOGO_IMAGE_PATH.exists():
    logo = Image.open(LOGO_IMAGE_PATH)
    st.image(logo, use_container_width=True)

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
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "system_instruction" not in st.session_state:
        st.session_state.system_instruction = "You are a helpful assistant. Be concise and friendly."

    st.divider()


    if st.button("Clear Chat", use_container_width=True):
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
