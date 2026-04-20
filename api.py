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
    
    # Check if a file was sent in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    file_path = "server_temp.pdf"
    file.save(file_path)

    # --- YOUR EXACT LANGCHAIN CODE ---
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
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
    global_rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)
    # ---------------------------------
    
    return jsonify({"message": "PDF successfully ingested and vectorized!"}), 200


# 3. Endpoint 2: Asking Questions
@app.route('/chat', methods=['POST'])
def chat():
    global global_rag_chain
    
    # Make sure a PDF was uploaded first
    if global_rag_chain is None:
        return jsonify({"error": "Please upload a PDF to the /upload endpoint first."}), 400
    
    # Grab the JSON data sent by the user
    data = request.get_json()
    user_question = data.get("question")
    
    if not user_question:
        return jsonify({"error": "No question provided in the JSON payload."}), 400

    # Ask the AI
    response = global_rag_chain.invoke({"input": user_question})
    
    # Return the answer as a JSON object
    return jsonify({"answer": response["answer"]}), 200

# Run the server
if __name__ == '__main__':
    print("🚀 Starting Flask API Server on port 5000...")
    app.run(debug=True, port=5000)