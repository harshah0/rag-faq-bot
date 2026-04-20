# AI Document Assistant (RAG Microservice)

An enterprise-grade, decoupled Retrieval-Augmented Generation (RAG) architecture that allows users to upload PDF documents and interactively query them using a dual-brain search engine and conversational memory.

## Architecture Highlights
This project avoids the common "monolithic script" approach by splitting the application into a decoupled client-server model:

* **Frontend Client (Streamlit):** A lightweight, interactive chat interface that handles file uploads and manages session state (chat history). It acts solely as a display layer, communicating with the backend via RESTful HTTP requests.
* **Backend API (Flask):** The core engine. It exposes two endpoints (`/upload` and `/chat`) to ingest documents and process user queries.
* **Hybrid Search Retrieval:** To combat LLM hallucination and ensure high precision, the retrieval engine uses an **Ensemble Retriever**. It combines:
  * **Dense Vector Search (Chroma):** Finds context based on semantic meaning.
  * **Sparse Keyword Search (BM25):** Ensures exact-match precision for specific acronyms, names, or terminology.
* **Conversational Memory:** The backend utilizes a history-aware query reformulator, allowing the AI to understand pronouns and follow-up context across multiple turns of conversation.

## Tech Stack
* **Language:** Python
* **Backend Framework:** Flask, Flask-CORS
* **Frontend UI:** Streamlit
* **AI & Orchestration:** LangChain, Google Gemini API
* **Vector Database:** Chroma
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)

## Usage
*Ensure you have a `.env` file with your `GOOGLE_API_KEY` before running.*

1. **Start the Backend:**
   ```bash
   python api.py