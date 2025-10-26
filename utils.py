# utils.py

import os
import base64
import uuid
import datetime
from PIL import Image
from mistralai import Mistral
from pyairtable import Api
from sentence_transformers import SentenceTransformer
# from langchain.schema import Document
# from langchain_community.document_loaders import Docx2txtLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# --- Load secrets from env or st.secrets ---
def get_secret(key):
    return os.getenv(key) or st.secrets.get(key)

MISTRAL_API_KEY = get_secret("MISTRAL_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
AIRTABLE_API_KEY = get_secret("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = get_secret("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = get_secret("AIRTABLE_TABLE_NAME")

# # --- API Keys from .env ---
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
# AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
# AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")
INDEX_NAME = "vectorization-chunk"

# --- Initialize Clients ---
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API_KEY)
api = Api(AIRTABLE_API_KEY)
table = api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)

# --- Setup Pinecone Index ---
if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# --- OCR & Document Loading ---
def ocr_with_mistral(file_path: str, file_type: str) -> Document:
    """Use Mistral OCR on PDF or image files."""
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    if file_type == "pdf":
        document = {"type": "document_url", "document_url": f"data:application/pdf;base64,{encoded}"}
    elif file_type in ["jpg", "jpeg", "png"]:
        document = {"type": "image_url", "image_url": f"data:image/{file_type};base64,{encoded}"}
    else:
        raise ValueError("Unsupported file type for OCR")
    response = mistral_client.ocr.process(model="mistral-ocr-latest", document=document)
    return Document(page_content="\n".join([page.markdown for page in response.pages]))

def load_document(file_path: str) -> list[Document]:
    """Load document content from file based on extension."""
    ext = file_path.split('.')[-1].lower()
    if ext in ["pdf", "jpg", "jpeg", "png"]:
        return [ocr_with_mistral(file_path, ext)]
    elif ext == "docx":
        return Docx2txtLoader(file_path).load()
    elif ext == "txt":
        return TextLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """Split documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# --- Vector Embedding & Pinecone ---
def embed_chunks(chunks: list) -> tuple:
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.encode(texts)
    return texts, embeddings

def store_in_pinecone(index, texts: list, embeddings: list) -> None:
    vectors = [
        {"id": f"doc-{i}", "values": emb, "metadata": {"text": text}}
        for i, (text, emb) in enumerate(zip(texts, embeddings))
    ]
    index.upsert(vectors=vectors, batch_size=100)

def process_and_store(file_path: str) -> int:
    """Load, chunk, embed, and store document in Pinecone."""
    docs = load_document(file_path)
    chunks = chunk_documents(docs)
    texts, embeddings = embed_chunks(chunks)
    store_in_pinecone(index, texts, embeddings)
    return len(chunks)

def get_context_from_vector_store(query_text: str, top_k: int = 2) -> str:
    """Query Pinecone for relevant document chunks based on query embedding."""
    query_embedding = embedder.encode([query_text])[0].tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return "\n\n".join([match['metadata'].get("text", "") for match in results['matches']])

# --- Airtable Chat Memory ---
def store_message_in_airtable(
    session_id: str,
    user_input: str,
    llm_response: str,
    user_id: str = "default_user",
    thread_id: str = "default_thread",
    source_type: str = "chat",
    chunk_id: str = "",
    embedding_vector: list = None,
    metadata: dict = None
):
    """Store chat interaction with embeddings in Airtable and Pinecone."""
    try:
        if not embedding_vector and user_input:
            embedding_vector = embedder.encode([user_input])[0].tolist()
            chunk_id = f"chat-{uuid.uuid4()}"
            index.upsert(vectors=[{
                "id": chunk_id,
                "values": embedding_vector,
                "metadata": {"text": user_input, "type": "chat"}
            }])

        current_time = datetime.datetime.utcnow().isoformat() + "Z"
        new_data = {
            "Session_ID": session_id,
            "User_Input": user_input,
            "LLM_Response": llm_response,
            "Timestamp": current_time,
            "Source_Type": source_type,
            "Chunk_ID": chunk_id,
            "Embedding_Vector": str(embedding_vector) if embedding_vector else "",
            "Metadata": str(metadata) if metadata else "",
            "User_ID": user_id,
            "Thread_ID": thread_id
        }
        table.create(new_data)
    except Exception as e:
        print(f"Error storing in Airtable: {str(e)}")

def retrieve_conversation_history(
    session_id: str = None,
    user_id: str = "default_user",
    thread_id: str = "default_thread",
    limit: int = 10
) -> list:
    """Retrieve last few messages from Airtable memory."""
    try:
        formula_parts = [f'User_ID = "{user_id}"']
        if session_id:
            formula_parts.append(f'Session_ID = "{session_id}"')
        if thread_id:
            formula_parts.append(f'Thread_ID = "{thread_id}"')
        formula = f"AND({', '.join(formula_parts)})"

        records = table.all(formula=formula, sort=["Timestamp"], max_records=limit)
        return [
            {
                "role": "user" if record["fields"].get("User_Input") else "assistant",
                "content": record["fields"].get("User_Input", "") or record["fields"].get("LLM_Response", ""),
                "timestamp": record["fields"].get("Timestamp", "")
            }
            for record in records
        ]
    except Exception as e:
        print(f"Error retrieving from Airtable: {str(e)}")
        return []

