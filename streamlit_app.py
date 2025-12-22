# --- Base ---
import streamlit as st
from PIL import Image
import os
import base64
from pathlib import Path
# --- Services ---
from services.rag import search_pathology_documents
from services.llm import generate_response
# --- Utils ---
from utils.URL_validation import is_valid_url
# --- Prompts ---
from prompts.system_prompt import build_system_prompt
# --- Langfuse ---
from observability.langfuse_client import init_langfuse    



BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"

BG_IMAGE_PATH = ASSETS_DIR / "fundo.png"
LOGO_IMAGE_PATH = ASSETS_DIR / "logo.png"

# Fetching secrets from the Streamlit secrets file
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]["key"]
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


# --- Check if Langfuse client is already initialized ---
if "langfuse_client" not in st.session_state:
    st.session_state.langfuse_client = init_langfuse(
        LANGFUSE_PUBLIC_KEY,
        LANGFUSE_SECRET_KEY,
        LANGFUSE_HOST
    )


# --- Function to reset the chat session ---
def reset_chat():
    """Forces the chat to be re-initialized when settings change."""
    if "langfuse_client" in st.session_state and st.session_state.langfuse_client:
        st.session_state.langfuse_client.flush()

    st.session_state["chat"] = None
    st.session_state["messages"] = []
    st.session_state["session_id"] = os.urandom(16).hex()
    st.session_state.last_user_input = ""  # Reset the previous user input


# --- 
def submit_message():
    user_query = st.session_state.get("chat_input", "").strip()
    if not user_query:
        return

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })
    
    is_url_input = (
        is_valid_url(user_query)
        and user_query.strip().startswith("http")
        and len(user_query.strip().split()) == 1
    )
    
    if is_url_input:
        st.session_state.last_topic = None
        st.session_state.last_topic_context = None

    # CASE 1: URL → não faz RAG
    if not is_url_input:
        docs = []
        if len(user_query.split()) >= 5:
            docs = search_pathology_documents(user_query)

        if (docs and docs[0]["score"] > 0.6 and st.session_state.last_topic in (None, "symptom_description")):
            st.session_state.last_topic = docs[0]["name of pathology"]
            st.session_state.last_topic_context = docs[0]["text"]

        if st.session_state.last_topic is None:
            st.session_state.last_topic = "symptom_description"
            st.session_state.last_topic_context = (
                "The user is describing symptoms. No diagnosis has been established."
            )

    # Build system prompt
    system_prompt = build_system_prompt(
        st.session_state.last_topic,
        st.session_state.last_topic_context
    )

    # Generate response
    response = generate_response(
        user_input=user_query,
        system_prompt=system_prompt,
        temperature=st.session_state.temperature,
        api_key=GOOGLE_API_KEY,
        use_url_tool=is_url_input
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

    if st.button("🔄 Change Topic", use_container_width=True):
        st.session_state.last_topic = None
        st.session_state.last_topic_context = None

        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "Ok 😊 Vamos mudar de tema.\n\n"
                "Sobre que assunto de saúde gostarias de falar agora?"
            )
        })
        st.rerun()

# --- Display Introductory Message ---
st.markdown("""
    👋 **Welcome to Your Personalized Health Assistant!**
    This chatbot can help you find medical services, doctor recommendations, and more!
""")
st.divider()