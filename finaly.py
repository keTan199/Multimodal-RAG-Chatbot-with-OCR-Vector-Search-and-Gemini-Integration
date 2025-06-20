import os
import base64
from PIL import Image
from mistralai import Mistral
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.document_loaders import Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import google.generativeai as genai
from google.generativeai import GenerativeModel

# Initialize API keys
MISTRAL_API_KEY = "Vf2NV8rYMD97nJh0bpjBk31piTEsjZpk"
PINECONE_API_KEY = "pcsk_6DzCw2_71mSh3YY3BGw28tjYd9K6gLmZwCGtP32iaiXe2nvatQHAZ7dbFbvoq3bJ29v2hZ"
GOOGLE_API_KEY = "AIzaSyCmp9luJBO34n1tSdE6LPJ7lzpxO3E6TFE"

# Initialize clients
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
genai.configure(api_key=GOOGLE_API_KEY)

# Setup Pinecone index
index_name = "vectorization-chunk"
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# OCR with Mistral
def ocr_with_mistral(file_path: str, file_type: str) -> Document:
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    if file_type == "pdf":
        document = {"type": "document_url", "document_url": f"data:application/pdf;base64,{encoded}"}
    elif file_type in ["jpg", "jpeg", "png"]:
        document = {"type": "image_url", "image_url": f"data:image/{file_type};base64,{encoded}"}
    else:
        raise ValueError("Unsupported file type for OCR")
    response = mistral_client.ocr.process(model="mistral-ocr-latest", document=document)
    all_text = "\n".join([page.markdown for page in response.pages])
    return Document(page_content=all_text)

# Load and parse file
def load_document(file_path: str) -> list[Document]:
    ext = file_path.split('.')[-1].lower()
    if ext in ["pdf", "jpg", "jpeg", "png"]:
        return [ocr_with_mistral(file_path, ext)]
    elif ext == "docx":
        return Docx2txtLoader(file_path).load()
    elif ext == "txt":
        return TextLoader(file_path).load()
    raise ValueError(f"Unsupported file type: {ext}")

# Chunk documents
def chunk_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# Embed and store
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
    docs = load_document(file_path)
    chunks = chunk_documents(docs)
    texts, embeddings = embed_chunks(chunks)
    store_in_pinecone(index, texts, embeddings)
    return len(chunks)

def get_context_from_vector_store(query_text: str, top_k: int = 2) -> str:
    query_embedding = embedder.encode([query_text])[0].tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    retrieved_texts = [match['metadata'].get("text", "") for match in results['matches']]
    return "\n\n".join(retrieved_texts)

# Streamlit UI
st.title("RAG-Based Chatbot with Mistral OCR")

# Sidebar - Upload and Process
st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, TXT, or Image", type=["pdf", "docx", "txt", "jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_file:
    temp_file_paths = []
    for uploaded_file in uploaded_file:
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        temp_file_paths.append(temp_file_path)

    if st.sidebar.button("Update Vector Store"):
        with st.spinner("Processing and storing all in vector store..."):
            total_chunks = 0
            for path in temp_file_paths:
                total_chunks += process_and_store(path)
            st.sidebar.success(f"✅ Stored total {total_chunks} chunks.")

    if st.sidebar.button("Process as Raw Data"):
        with st.spinner("Extracting text from documents..."):
            full_raw = ""
            for path in temp_file_paths:
                documents = load_document(path)
                full_raw += "\n\n".join([doc.page_content for doc in documents]) + "\n\n"
            st.session_state.raw_content = full_raw
            st.sidebar.success("✅ Combined raw text extracted.")

    for path in temp_file_paths:
        if os.path.exists(path):
            os.remove(path)

# Model selection
st.sidebar.header("Model Selection")
available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
selected_model = st.sidebar.selectbox("Choose Gemini Model", available_models, index=available_models.index("gemini-2.0-flash") if "gemini-2.0-flash" in available_models else 0)
model = GenerativeModel(selected_model)

# Chat
st.header("Chat with the Bot")
use_rag = st.checkbox("Use RAG (contextual retrieval)", value=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "raw_content" not in st.session_state:
    st.session_state.raw_content = ""

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            context = get_context_from_vector_store(prompt) if use_rag else ""
            raw = st.session_state.raw_content
            full_prompt = f"""Answer the question below:
Question: {prompt}
Context from vector store: {context}
Raw uploaded document: {raw}"""
            response = model.generate_content(full_prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
