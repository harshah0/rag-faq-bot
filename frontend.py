import streamlit as st
import requests

# The URL of your new Flask Backend
API_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="AI Resume Reader", page_icon="📄")
st.title("📄 AI Document Reader")
st.caption("Powered by a custom Flask + LangChain REST API")

# 1. The File Uploader
uploaded_file = st.file_uploader("Drop your PDF here", type="pdf")

if uploaded_file is not None:
    # We only want to upload the file to the backend once
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Sending PDF to the Backend Server for processing..."):
            # Package the file and send it via HTTP POST
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                response = requests.post(f"{API_URL}/upload", files=files)
                if response.status_code == 200:
                    st.success("Backend successfully digested the PDF!")
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.messages = [] # Clear chat history for the new file
                else:
                    st.error(f"Backend error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect! Is your Flask backend server running?")

    # Setup Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 2. The Chat Interface
    if prompt := st.chat_input("Ask the backend a question about this document..."):
        # Display user message instantly
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send question to Flask API via JSON
        with st.chat_message("assistant"):
            with st.spinner("Backend is thinking..."):
                try:
                    payload = {"question": prompt}
                    response = requests.post(f"{API_URL}/chat", json=payload)
                    
                    if response.status_code == 200:
                        # Extract the answer from the backend's JSON response
                        answer = response.json().get("answer", "No answer found.")
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(f"API Error: {response.json().get('error', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("🚨 Lost connection to the backend server.")
else:
    st.info("Please upload a PDF file to activate the chat.")