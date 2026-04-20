import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader # Upgraded Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load API Key securely
load_dotenv()

# 2. Cache the RAG Setup, but link it to the uploaded file!
@st.cache_resource
def setup_rag(file_path):
    # Read the PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    # Embedding & Storage
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    # AI Brain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    system_prompt = (
        "You are a helpful and friendly assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

# --- STREAMLIT UI ---

st.title("📄 AI PDF Reader")
st.write("Upload a syllabus, resume, or guide, and ask me anything about it!")

# 3. The File Uploader Widget
uploaded_file = st.file_uploader("Drop your PDF here", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily so LangChain can read it
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
        
    # Show a loading spinner while the AI digests the document
    with st.spinner("Digesting the PDF..."):
        rag_chain = setup_rag("temp.pdf")
        
    st.success("PDF loaded successfully! Ask away.")

    # Setup Chat History in Streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("What is this document about?"):
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response and display it
        with st.chat_message("assistant"):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
else:
    st.info("Please upload a PDF file to activate the chat.")