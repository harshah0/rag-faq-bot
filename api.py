import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# 1. Setup the Server
load_dotenv()
app = Flask(__name__)
CORS(app) # Allows other apps (like your frontend) to talk to this API

# Global variable to store our AI "Brain" in server memory
global_rag_chain = None


# 2. Endpoint 1: Uploading the PDF
@app.route('/upload', methods=['POST'])
def upload_pdf():
    global global_rag_chain
    
    # --- MISSING LINES RESTORED: Catch and save the uploaded file ---
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    file_path = "server_temp.pdf"
    file.save(file_path)
    # ----------------------------------------------------------------

    # --- YOUR UPGRADED LANGCHAIN CODE ---
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    # 1. Brain A: Semantic Search (Chroma Vectors)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
    
    # 2. Brain B: Keyword Search (BM25)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3 
    
    # 3. The Hybrid Engine: Combine them 50/50
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], 
        weights=[0.5, 0.5]
    )
    
   # 4. Setup the AI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # NEW: Prompt 1 - The Question Reformulator
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # This creates the "Smart Searcher" that rewrites questions using history
    history_aware_retriever = create_history_aware_retriever(
        llm, ensemble_retriever, contextualize_q_prompt
    )

    # NEW: Prompt 2 - The Actual Answer Generator
    system_prompt = (
        "You are a helpful and friendly assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Context: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Combine the Smart Searcher and the Answer Generator
    global_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return jsonify({"message": "PDF successfully ingested and vectorized!"}), 200

# 3. Endpoint 2: Asking Questions (Upgraded with Memory)
@app.route('/chat', methods=['POST'])
def chat():
    global global_rag_chain
    
    if global_rag_chain is None:
        return jsonify({"error": "Please upload a PDF to the /upload endpoint first."}), 400
    
    data = request.get_json()
    user_question = data.get("question")
    raw_history = data.get("chat_history", []) # Get history from frontend
    
    if not user_question:
        return jsonify({"error": "No question provided in the JSON payload."}), 400

    # Convert Streamlit's history format into LangChain's strict Message Objects
    formatted_history = []
    for msg in raw_history:
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["content"]))
        else:
            formatted_history.append(AIMessage(content=msg["content"]))

    # Pass BOTH the input and the history to the chain
    response = global_rag_chain.invoke({
        "input": user_question,
        "chat_history": formatted_history
    })
    
    return jsonify({"answer": response["answer"]}), 200

# Run the server
if __name__ == '__main__':
    print("🚀 Starting Flask API Server on port 5000...")
    app.run(debug=True, port=5000)