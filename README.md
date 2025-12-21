# HealthMatch

Your Personalized Health Assistant

## Overview

HealthMatch is an AI-powered conversational health assistant designed to help users understand medical conditions and access relevant health-related information through natural language interaction.

The application allows users to describe symptoms, ask medical questions, or provide external URLs, and receive clear, contextualized responses grounded in medical knowledge. It is aimed at users who want quick and accessible medical information without navigating complex healthcare websites.

## Features

- Conversational chatbot for medical information and symptom-related questions
- Context-aware multi-turn conversations
- Retrieval-Augmented Generation (RAG) using a medical health dictionary
- Semantic search with vector embeddings stored in MongoDB
- URL content analysis using Gemini URL Context tool
- AI observability and tracing with Langfuse

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

- UI Layer: Streamlit interface for user interaction and chat display
- Service Layer: Application logic handling user input, context management, and orchestration
- AI Layer: Gemini LLM for response generation combined with RAG for grounded answers
- Data Layer: MongoDB storing medical documents and vector embeddings

An architecture diagram can be found in docs/ARCHITECTURE.md.

## Installation & Setup

### Prerequisites
- Python 3.x
- MongoDB with Vector Search enabled
- API keys for Google Gemini and Langfuse

### Installation Steps

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [project-name]
```

2. Install dependencies:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required environment variables:**
```
GOOGLE_API_KEY=your_gemini_api_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
# Add other required API keys
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

**Deployment Platform:** [Streamlit Cloud / Render / Vercel / etc.]

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
├── .env.example          # Environment variable template
└── README.md             # This file
```

**Note:** Component-level READMEs (e.g., `services/README.md`, `tools/README.md`) are recommended if those components need detailed explanation.

## Team

- Carlota Fradinho e Silva – Project Management, Documentation, UI
- Gonçalo Morais – Research, Medical Content, UI Support
- Gonçalo Palhoto – Technical Architecture, Backend Development

---