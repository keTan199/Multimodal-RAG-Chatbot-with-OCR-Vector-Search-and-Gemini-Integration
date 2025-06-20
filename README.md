# RAG Chatbot with OCR using Mistral, Gemini, Pinecone

A Streamlit chatbot that accepts documents/images, extracts text using Mistral OCR, chunks and embeds it into Pinecone, and uses Google Gemini for context-aware answers.

## Features

- Upload PDF, DOCX, TXT, or Image
- Extract text using Mistral OCR
- Chunk and embed using `sentence-transformers`
- Store and retrieve from Pinecone
- Answer using Gemini with or without RAG context

## Setup

1. Create `.streamlit/secrets.toml` with API keys.
2. Install dependencies:
