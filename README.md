# 🇳🇬 RAG Chat API for Nigerian Business Q&A (FastAPI Backend)

This project provides a lightweight, production-ready backend for a **Retrieval-Augmented Generation (RAG)** knowledge base focused on helping users **start and run businesses in Nigeria**. Built with **FastAPI**, it integrates a retrieval tool that searches a business-focused document store and returns relevant context to a LangGraph agent.

It exposes a `/chat` API endpoint that accepts user messages, queries the retriever (`retrieve_context`) for relevant information, and returns AI-generated assistant replies grounded in Nigerian business knowledge.

---

## Features

- Fast, asynchronous REST API using FastAPI
- Multi-turn chat support
- LangGraph-based RAG integration
- CORS enabled (open to all origins)
- Health check endpoint (`/`)

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Vit-Solution/rag-faq-algo.git
cd rag-faq-algo.git
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
```bash
### 4. Run the server
uvicorn main:app --reload

```

## API Endpoints
GET /

Health check endpoint.

Response:
```bash
{
  "success": true
}

```

POST /chat

Submit a chat message (or conversation history). The backend responds with an assistant reply.

Request Body

``` bash
{
  "messages": [
    {
      "role": "user",
      "content": "How can I register a business in Nigeria as a foreigner?"
    }
  ]
}

```
Response Body

```bash
{
  "message": {
    "role": "assistant",
    "content": "
        As a foreigner, you can register a business in Nigeria, but you cannot register a "Business Name" (also known as an Enterprise), as this category is reserved for Nigerian citizens. You will need to register a limited liability company or another suitable business structure.

        Here are some key points and requirements for foreigners:

        Business Permit: A foreigner needs a business permit to work and carry out business in Nigeria.
        Expatriate Quota: If you plan to employ expatriates, your company will need an Expatriate Quota, which is granted by the Ministry of Interior.
        Minimum Share Capital: There is a required minimum share capital of ₦10,000,000 for foreign-owned businesses in Nigeria.
        Directors: A private company requires a minimum of one director.
        Registration Body: The Corporate Affairs Commission (CAC) is the government body responsible for registering businesses in Nigeria.
        Timeline: Business registration typically takes between 1-3 weeks.

        To register a business, you will generally need to provide details such as proposed business names, a registered address, contact information, and identification documents for the proprietor(s).

        It is highly recommended to seek professional advice to determine the most suitable type of business to register based on your goals, the nature of your business, and your long-term plans.

         For more information, you can visit the Corporate Affairs Commission (CAC) website." }
}
```
Data Models
```bash
{
  "role": "user" | "assistant",
  "content": "Message text here"
}
```

## ChatRequest

messages: a single Message object or a list of them

ChatResponse

message: the assistant's reply (Message)

## Tech Stack

FastAPI - Web framework

Pydantic - Data validation

Uvicorn - ASGI server

LangGraph - RAG agent framework (via utils.agent)

CORS Middleware - Allows frontend integration

## Project Structure
```bash
.
├── main.py           # FastAPI app with /chat endpoint
├── utils.py          # RAG agent logic (LangChain)
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```