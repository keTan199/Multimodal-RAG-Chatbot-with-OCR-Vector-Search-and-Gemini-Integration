# 🤖 Multimodal RAG Chatbot with OCR, Vector Search, and Gemini Integration

A powerful **Streamlit-based AI chatbot** that performs **multimodal Retrieval-Augmented Generation (RAG)** by combining the following:
- 🧠 **Google Gemini LLM**
- 🧾 **Mistral OCR API**
- 📄 **PDF, DOCX, TXT, Image** ingestion
- 🔍 **Vector Search using Pinecone**
- 💬 **Chat Memory with Airtable**
- 📸 **Multimodal inputs with deep embedding + OCR pipeline**

> 🌐 Live Demo: [Streamlit App](https://rag-chatbot-n3m296qczrmhwobmgy2yew.streamlit.app/)

---

## 📸 Interface Screenshot

![Chatbot Screenshot](https://drive.google.com/uc?id=1VxOMtXx67shcMxFjKisRazK8QKJYn5sV)

---

## 🏗️ System Architecture (Step-by-Step)

1. **User Interface (Streamlit):**  
   Users interact with a clean, chat-style interface to upload files and ask questions.

2. **Document Upload Handling:**  
   - Accepts `.pdf`, `.docx`, `.txt`, `.jpg`, `.png` files.  
   - Image or scanned files are processed using **Mistral OCR**.  
   - Text-based files are loaded using **LangChain loaders**.

3. **Text Chunking:**  
   Documents are split into smaller overlapping chunks using `RecursiveCharacterTextSplitter`.

4. **Embedding Generation:**  
   Each chunk is embedded using `sentence-transformers` (`all-MiniLM-L6-v2` model).

5. **Vector Storage in Pinecone:**  
   Embeddings and corresponding chunks are stored in a **Pinecone** vector index.

6. **Chat Input from User:**  
   Users enter a prompt/question in the chat input box.

7. **Contextual Retrieval (RAG):**  
   If RAG is enabled:
   - The prompt is embedded.
   - Pinecone retrieves top-k semantically similar chunks as context.

8. **Prompt Construction:**  
   Combines:
   - Retrieved context
   - Chat history from Airtable
   - Raw uploaded text

9. **Gemini LLM Response:**  
   The full prompt is sent to **Google Gemini** (via `google-generativeai`).  
   The LLM generates a context-aware answer.

10. **Output & Storage:**  
    - The assistant's reply is displayed in the chat.  
    - The interaction (prompt, reply, metadata, embedding) is saved to **Airtable**.

---

## 📌 Key Features

- 🔍 **RAG Pipeline**: Combine vector search with context-aware Gemini LLM.
- 📑 **Document Support**: Upload and extract text from `.pdf`, `.docx`, `.txt`, `.jpg`, `.png`.
- 🧠 **Mistral OCR**: OCR processing for scanned images or PDFs.
- 📚 **Vector Store with Pinecone**: Efficient semantic chunk search.
- 🧾 **Airtable Chat Memory**: Persistent, structured storage of messages with embeddings.
- 🔁 **Conversation Threading**: Session-based multi-turn conversation history.
- 🧪 **Model Selector**: Switch Gemini model versions on the fly.
- 🎯 **Lightweight & Modular**: Easy to extend with new models or tools.

---

## ⚙️ Tech Stack

- `streamlit` — Web app UI
- `google-generativeai` — Gemini LLM APIs
- `mistralai` — OCR APIs from Mistral
- `pyairtable` — Airtable for memory and logging
- `pinecone-client` — Vector DB for semantic search
- `sentence-transformers` — Embedding generation
- `langchain` — Chunking, Doc parsing, schema

---







---

## 🤖 Daily Automated Update

Last Updated: 18/7/2026, 9:00:01 am


---

## 🤖 Daily Automated Update

Last Updated: 19/7/2026, 9:00:01 am


---

## 🤖 Daily Automated Update

Last Updated: 20/7/2026, 9:00:01 am


---

## 🤖 Daily Automated Update

Last Updated: 21/7/2026, 9:00:01 am


---

## 🤖 Daily Automated Update

Last Updated: 22/7/2026, 9:00:01 am


---

## 🤖 Daily Automated Update

Last Updated: 23/7/2026, 9:00:01 am


---

## 🤖 Daily Automated Update

Last Updated: 24/7/2026, 9:00:01 am
