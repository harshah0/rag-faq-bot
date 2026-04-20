import streamlit as st
import requests

# Architecture Configuration
BACKEND_SERVICE_URL = "http://127.0.0.1:5000"
INGESTION_ENDPOINT = f"{BACKEND_SERVICE_URL}/api/v1/document/ingest"
QUERY_ENDPOINT = f"{BACKEND_SERVICE_URL}/api/v1/chat/query"

# UI Configuration
st.set_page_config(page_title="Intelligent Document Engine", page_icon="🧠", layout="centered")
st.title("🧠 Intelligent Document Engine")
st.caption("Custom Microservice Architecture: Flask + LangChain + Streamlit")

# Session State Initialization
if "conversation_log" not in st.session_state:
    st.session_state.conversation_log = []
if "active_document" not in st.session_state:
    st.session_state.active_document = None

# --- SIDEBAR OR TOP WIDGET: File Upload ---
uploaded_document = st.file_uploader("Upload a knowledge base (PDF format)", type="pdf")

if uploaded_document is not None:
    # Trigger ingestion only if it's a new file
    if st.session_state.active_document != uploaded_document.name:
        with st.spinner("Transmitting document to backend parsing engine..."):
            
            payload_files = {"file": (uploaded_document.name, uploaded_document.getvalue(), "application/pdf")}
            
            try:
                ingest_response = requests.post(INGESTION_ENDPOINT, files=payload_files)
                if ingest_response.status_code == 200:
                    st.success("Backend processing complete. Ready for queries.")
                    st.session_state.active_document = uploaded_document.name
                    st.session_state.conversation_log = [] # Reset memory for new file
                else:
                    st.error(f"Ingestion Failure: {ingest_response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Fatal Error: Unable to establish connection with the Backend Microservice.")

    # Render Chat History
    for dialogue in st.session_state.conversation_log:
        with st.chat_message(dialogue["role"]):
            st.markdown(dialogue["content"])

    # --- CHAT INPUT ---
    user_input = st.chat_input("Enter your query regarding this document...")
    
    if user_input:
        # Optimistically render user input
        st.session_state.conversation_log.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Transmit query to backend
        with st.chat_message("assistant"):
            with st.spinner("Processing context and generating response..."):
                try:
                    query_payload = {
                        "question": user_input,
                        "chat_history": st.session_state.conversation_log
                    }
                    api_response = requests.post(QUERY_ENDPOINT, json=query_payload)
                    
                    if api_response.status_code == 200:
                        engine_output = api_response.json().get("answer", "System returned an empty response.")
                        st.markdown(engine_output)
                        st.session_state.conversation_log.append({"role": "assistant", "content": engine_output})
                    else:
                        try:
                            error_details = api_response.json().get("error", "Unknown processing error")
                            st.error(f"Engine Error: {error_details}")
                        except requests.exceptions.JSONDecodeError:
                            st.error("Critical Failure: The backend service crashed unexpectedly.")
                            
                except requests.exceptions.ConnectionError:
                    st.error("Fatal Error: Connection to the RAG backend was lost.")
else:
    st.info("Awaiting document upload to initialize the context engine.")