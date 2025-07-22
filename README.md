# RAG Chatbot with FAISS and LLaMA

This is a Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that lets users upload a PDF and ask questions based on its content using FAISS for vector search and Ollama (LLaMA model) for answering questions.

---

## 🔧 Features
- Upload and parse PDFs
- Embed content using HuggingFace sentence transformers
- Store and retrieve embeddings with FAISS
- Ask natural language questions and get accurate answers from the uploaded document

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

Make sure you also have Ollama running locally:
```bash
ollama run llama3:latest
```

### 3. Run the Streamlit App
```bash
streamlit run rag_chatbot.py
```

---

## 🧠 How It Works
1. PDF content is extracted using PyPDF2
2. Text is chunked and embedded using HuggingFace embeddings
3. Vectors are stored and searched with FAISS
4. A RetrieverQA chain fetches relevant chunks
5. Ollama's LLaMA model answers based on context

---

## 🐳 Docker Support
To run this app in a Docker container, see the `Dockerfile` included.
```bash
docker build -t rag-chatbot .
docker run -p 8501:8501 rag-chatbot
```

---

## 📂 Directory Structure
```
rag-chatbot/
├── rag_chatbot.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📝 License
MIT License

---

## 🙋‍♂️ Questions?
Feel free to open an issue or contact the author.
