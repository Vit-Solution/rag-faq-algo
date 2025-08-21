from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Literal

from utils import agent  # ← Replace with actual file name

app = FastAPI()


# Request schema
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


# Response schema


class ChatResponse(BaseModel):
    messages: List[Message]  # Return full message history (after update)
    response: str  # Last assistant reply


@app.get("/")
async def health_check():
    return JSONResponse(content={"success": True})


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # Convert messages to dicts
    conversation = [msg.model_dump() for msg in req.messages]

    # Call LangGraph agent
    result = agent.invoke({"messages": conversation})

    # Extract new assistant message
    new_assistant_message = None

    last_res = result["messages"][-1].content
    if last_res:
        mssg = {}
        mssg["role"] = "assistant"
        mssg["content"] = last_res
        new_assistant_message = mssg

    if not new_assistant_message:
        return ChatResponse(response="[No assistant reply]", messages=conversation)

    # Append assistant reply to conversation
    conversation.append(new_assistant_message)

    # Keep only the last 3 user–assistant pairs (6 messages max)
    def trim_conversation(messages: List[dict]) -> List[dict]:
        trimmed = []
        pairs = []
        current_pair = []
        for msg in messages:
            current_pair.append(msg)
            if msg["role"] == "assistant":
                pairs.append(current_pair)
                current_pair = []
        # Keep only the last 3 pairs
        last_three_pairs = pairs[-3:]
        for pair in last_three_pairs:
            trimmed.extend(pair)
        return trimmed

    trimmed_convo = trim_conversation(conversation)

    return ChatResponse(
        response=new_assistant_message["content"], messages=trimmed_convo
    )
