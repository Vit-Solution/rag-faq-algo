# 🤖 RAG Chat API (FastAPI Backend)

This project provides a lightweight, production-ready backend for a **Retrieval-Augmented Generation (RAG)** QnA knowledgbase for starting and running businesses in Nigeria using **FastAPI**. It exposes a `/chat` API endpoint that accepts user messages, sends them to a LangGraph agent, and returns AI-generated assistant replies.

---

## 📦 Features

- Fast, asynchronous REST API using FastAPI
- Multi-turn chat support
- LangGraph-based RAG integration
- CORS enabled (open to all origins)
- Health check endpoint (`/`)

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Vit-Solution/rag-faq-algo.git
cd rag-faq-algo.git

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn main:app --reload

```