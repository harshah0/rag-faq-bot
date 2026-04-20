import os
from typing import Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Search & Vector Libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# AI & Orchestration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Initialize Environment and App
load_dotenv()
app = Flask(__name__)
CORS(app)

# Global singleton for the active RAG pipeline
ACTIVE_ASSISTANT_PIPELINE = None

@app.route('/api/v1/document/ingest', methods=['POST'])
def process_document() -> tuple[Dict[str, str], int]:
    """
    Endpoint to ingest a PDF, chunk the text, and initialize a 
    dual-retrieval (Dense + Sparse) hybrid search engine.
    """
    global ACTIVE_ASSISTANT_PIPELINE
    
    if 'file' not in request.files:
        return jsonify({"error": "Missing payload: No PDF detected."}), 400
        
    uploaded_pdf = request.files['file']
    temp_storage_path = "runtime_storage.pdf"
    uploaded_pdf.save(temp_storage_path)

    # 1. Document Extraction & Segmentation
    doc_parser = PyPDFLoader(temp_storage_path)
    extracted_pages = doc_parser.load()
    
    segmenter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    document_segments = segmenter.split_documents(extracted_pages)
    
    # 2. Semantic Search Initialization (ChromaDB)
    hf_embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    semantic_db = Chroma.from_documents(documents=document_segments, embedding=hf_embedding_model)
    semantic_fetcher = semantic_db.as_retriever(search_kwargs={"k": 3}) 
    
    # 3. Keyword Search Initialization (BM25)
    keyword_fetcher = BM25Retriever.from_documents(document_segments)
    keyword_fetcher.k = 3 
    
    # 4. Hybrid Search Aggregator
    hybrid_engine = EnsembleRetriever(
        retrievers=[keyword_fetcher, semantic_fetcher], 
        weights=[0.5, 0.5]
    )
    
    # 5. Language Model Initialization
    core_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Prompt: Final Answer Generation
    answer_prompt_config = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert analytical assistant. Utilize the provided context "
            "to formulate a precise answer. If the context does not contain the answer, "
            "state clearly that the information is unavailable. "
            "Context: {context}"
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(core_llm, answer_prompt_config)
    
   # Bind the final pipeline (Direct Hybrid Search)
    ACTIVE_ASSISTANT_PIPELINE = create_retrieval_chain(hybrid_engine, document_chain)
    return jsonify({"message": "Document successfully parsed and indexed."}), 200


@app.route('/api/v1/chat/query', methods=['POST'])
def execute_query() -> tuple[Dict[str, Any], int]:
    """
    Endpoint to receive a user query and session history, routing it 
    through the RAG pipeline to generate an AI response.
    """
    global ACTIVE_ASSISTANT_PIPELINE
    
    if ACTIVE_ASSISTANT_PIPELINE is None:
        return jsonify({"error": "Pipeline uninitialized. Upload a document first."}), 400
    
    request_payload = request.get_json()
    user_query = request_payload.get("question")
    session_history = request_payload.get("chat_history", [])
    
    if not user_query:
        return jsonify({"error": "Malformed request: Missing question payload."}), 400

    # Parse standard JSON history into LangChain Message schemas
    parsed_history = []
    for interaction in session_history:
        if interaction.get("role") == "user":
            parsed_history.append(HumanMessage(content=interaction.get("content")))
        else:
            parsed_history.append(AIMessage(content=interaction.get("content")))

    # Execute RAG Pipeline
    pipeline_result = ACTIVE_ASSISTANT_PIPELINE.invoke({
        "input": user_query,
        "chat_history": parsed_history
    })
    
    return jsonify({"answer": pipeline_result.get("answer", "No response generated.")}), 200


if __name__ == '__main__':
    print("Initializing Custom RAG Microservice on Port 5000...")
    app.run(debug=True, port=5000)