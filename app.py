import streamlit as st
import google.generativeai as genai
from google.generativeai import GenerativeModel
import os
import uuid
from utils import (
    process_and_store,
    get_context_from_vector_store,
    load_document,
    store_message_in_airtable,
    retrieve_conversation_history
)

# --- Unified Secret Loader ---
def get_secret(key):
    return os.getenv(key) or st.secrets.get(key)

# Configure Gemini
genai.configure(api_key=get_secret("GOOGLE_API_KEY"))

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>RAG-Based Chatbot with Mistral OCR</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------- Session State ----------
st.session_state.setdefault("session_id", str(uuid.uuid4()))
st.session_state.setdefault("user_id", "default_user")
st.session_state.setdefault("thread_id", "default_thread")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("raw_content", "")

# ---------- Sidebar: Chat History + Upload + Model ----------
# --- Chat History ---
st.sidebar.header("Chat History")
history = retrieve_conversation_history(
    session_id=st.session_state.session_id,
    user_id=st.session_state.user_id,
    thread_id=st.session_state.thread_id,
    limit=20
)

user_messages = [m for m in history if m["role"] == "user"]
recent_msgs = user_messages[-5:][::-1]

for i, msg in enumerate(recent_msgs, 1):
    preview = " ".join(msg["content"].split()[:5]) + "..."
    st.sidebar.markdown(
        f"""<div class='airtable-msg' title="{msg["content"]}"><strong>{i}.</strong> {preview}</div>""",
        unsafe_allow_html=True
    )

# --- Document Upload ---
st.sidebar.header("Document Upload")
st.sidebar.subheader("Update RAG Database")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, TXT, or Image",
    type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    temp_file_paths = []
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        temp_file_paths.append(temp_path)

    if st.sidebar.button("Update Vector Store"):
        with st.spinner("Processing..."):
            total_chunks = sum(process_and_store(path) for path in temp_file_paths)
            st.success(f"Stored {total_chunks} chunks.")

    if st.sidebar.button("Extract Raw Text"):
        with st.spinner("Extracting text..."):
            st.session_state.raw_content = ""
            for path in temp_file_paths:
                docs = load_document(path)
                st.session_state.raw_content += "\n\n".join(doc.page_content for doc in docs) + "\n\n"
            st.success("Text extracted.")

    for path in temp_file_paths:
        os.remove(path)

# --- Model Selection ---
st.sidebar.header("Model Selection")
available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
default_index = available_models.index("models/gemini-2.0-flash") if "models/gemini-2.0-flash" in available_models else 0
selected_model = st.sidebar.selectbox("Choose Gemini Model", available_models, index=default_index)

# ---------- Load Selected Model ----------
model = GenerativeModel(selected_model)

# ---------- Main Chat Area ----------
st.header("Chat with the Bot")
use_rag = st.toggle("Use RAG (contextual retrieval)", value=True)

# Initial message load
if not st.session_state.messages:
    st.session_state.messages = retrieve_conversation_history(
        session_id=st.session_state.session_id,
        user_id=st.session_state.user_id,
        thread_id=st.session_state.thread_id,
        limit=10
    )

with st.container():
    st.markdown('<div class="chat-scroll-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    st.markdown("</div>", unsafe_allow_html=True)

# Input
chat_input_placeholder = st.empty()
prompt = chat_input_placeholder.chat_input("Ask your question")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Generating response..."):
        context = get_context_from_vector_store(prompt) if use_rag else ""
        history_context = "\n\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-4:-1]
        )
        raw = st.session_state.raw_content
        full_prompt = f"""Conversation History:\n{history_context}\n\nContext from vector store:\n{context}\n\nRaw uploaded document:\n{raw}\n\nQuestion: {prompt}"""

        try:
            response = model.generate_content(full_prompt)
            response_text = response.text
        except Exception as e:
            response_text = f"Error: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})

    store_message_in_airtable(
        session_id=st.session_state.session_id,
        user_input=prompt,
        llm_response=response_text,
        user_id=st.session_state.user_id,
        thread_id=st.session_state.thread_id,
        source_type="chat",
        metadata={"model": selected_model}
    )
