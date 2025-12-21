# HealthMatch

Your Personalized Health Assistant

## Overview

HealthMatch is an AI-powered conversational health assistant designed to help users understand medical conditions and access relevant health-related information through natural language interaction.

The application allows users to describe symptoms, ask medical questions, or provide external URLs, and receive clear, contextualized responses grounded in medical knowledge. It is aimed at users who want quick and accessible medical information
to support understanding, without replacing professional medical advice.

## Features

- Conversational chatbot for medical information and symptom-related questions.
- Context-aware multi-turn conversations.
- Retrieval-Augmented Generation (RAG) using a medical health dictionary.
- Semantic search with vector embeddings stored in MongoDB.
- Strict topic isolation to prevent unintended introduction of unrelated medical conditions.
- URL content analysis using Gemini URL Context tool.
- AI observability and tracing with Langfuse.

## Tech Stack

**Backend:**
- Python
- Google Gemini API

**Frontend:**
- Streamlit

**Database:**
- MongoDB

**AI/ML:**
- Google Gemini LLM
- LangChain Google Generative AI Embeddings
- MongoDB Vector Search
- Langfuse for observability

## Architecture

The application follows a layered architecture with clear separation of concerns:

- UI Layer: Streamlit interface for user interaction and chat display.
- Service Layer: Application logic handling user input, context management, and orchestration.
- AI Layer: Gemini LLM for response generation combined with RAG for grounded answers.
- Data Layer: MongoDB storing medical documents and vector embeddings.

An architecture diagram and further explanation can be found in docs/ARCHITECTURE.md.

## Installation & Setup

### Prerequisites
- Python 3.x
- MongoDB with Vector Search enabled
- API keys for Google Gemini and Langfuse

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/Carlotavfsilva/capstone_project-HealthMatch
cd capstone_project-HealthMatch
```

2. Install dependencies:
```bash
uv sync
```

**Required environment variables:**
This application uses Streamlit Secrets for configuration.

Create a `.streamlit/secrets.toml` file with the following structure:

```
[GOOGLE_API_KEY]
key = "your_gemini_api_key"

[MONGODB_URI]
uri = "your_mongodb_uri"

[LANGFUSE]
public_key = "your_langfuse_public_key"
secret_key = "your_langfuse_secret_key"
host = "https://cloud.langfuse.com"
```

4. Run the application:
```bash
uv run streamlit run app.py
```

## Usage

Instructions and examples for using your application. Include:
- How to navigate the interface
- Key workflows
- Screenshots or GIFs demonstrating functionality (recommended for visual clarity)

**Example:**

1. Navigate to the main page
2. Upload a document or enter your query
3. Interact with the AI assistant through the chat interface
4. View results and explore additional features

*Add screenshots or GIFs here to visually demonstrate your application's key features*

## Deployment

**Live Application:** [Your deployed URL]

**Deployment Platform:** Streamlit Cloud

Instructions for deploying your own instance (if applicable).

## Project Structure

```
project-root/
├── app.py                 # Main application entry point
├── services/              # Business logic layer
├── tools/                 # Function calling tools
├── utils/                 # Utility functions
├── docs/
│   └── ARCHITECTURE.md    # Architecture decisions and explanations
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Team

- Carlota Fradinho e Silva – Project Management, Documentation, UI
- Gonçalo Morais – Backend Development, UI Support
- Gonçalo Palhoto – Technical Architecture, Research, Medical Content

---