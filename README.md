**Chat with PDF - A Streamlit Q&A Application**

This project is an interactive chatbot powered by a Large Language Model (LLM) that allows you to have a conversation with your PDF documents. Upload a PDF, and the application will enable you to ask questions about its content, providing intelligent answers based on the text.

The application is built using Streamlit for the user interface and LangChain for orchestrating the question-answering pipeline with the Groq API for fast LLM inference.

✨ Features
Upload PDF Files: Easily upload any PDF document directly through the web interface.

Text Extraction and Chunking: Automatically extracts text and splits it into manageable chunks for processing.

Vector Embeddings: Creates vector embeddings from the text using HuggingFace sentence transformers.

Persistent Vector Stores: Caches the PDF's vector store in a .pkl file to avoid reprocessing the same document.

Question Answering: Utilizes the Groq API with LLaMA 3 for real-time, intelligent answers to your questions.

User-Friendly Interface: A clean and simple UI built with Streamlit.

🛠️ Tech Stack
Application Framework: Streamlit

Orchestration: LangChain

LLM Provider: Groq (for fast inference with LLaMA 3)

PDF Processing: PyPDF2

Embeddings: HuggingFace Sentence Transformers

Vector Store: FAISS (from faiss-cpu)

Environment Variables: python-dotenv
