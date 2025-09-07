from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware

from utils import agent  # Make sure this points to your LangGraph agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class Message(BaseModel):
    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    messages: Union[Message, List[Message]]


class ChatResponse(BaseModel):
    message: Message  # Single assistant message


@app.get("/")
async def health_check():
    """Health Check, ensures the server is running fine!"""
    return JSONResponse(content={"success": True})


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Handles a chat request by normalizing message input, invoking the RAG pipeline,
    and returning the assistant's response.

    This endpoint:
    - Accepts a ChatRequest object containing one or more chat messages.
    - Normalizes the message(s) into a list of dictionaries.
    - Passes the conversation to a retrieval-augmented generation (RAG) agent.
    - Extracts and returns the assistant's final reply.

    Parameters:
        req (ChatRequest): The incoming request object containing chat messages.
                           `req.messages` may be a single message or a list of messages.

    Returns:
        ChatResponse: A response object containing the assistant's reply as a `Message`.
                      If the assistant did not return a reply, a default message is used.
    """
    # Normalize to list of dicts
    if isinstance(req.messages, list):
        conversation = [msg.model_dump() for msg in req.messages]
    else:
        conversation = [req.messages.model_dump()]

    # Invoke RAG pipeline
    result = agent.invoke({"messages": conversation})
    assistant_reply = result["messages"][-1].content

    if not assistant_reply:
        return ChatResponse(
            message=Message(role="assistant", content="[No assistant reply]")
        )

    return ChatResponse(message=Message(role="assistant", content=assistant_reply))
