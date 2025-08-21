from config import doc_store, llm

from langchain.tools.retriever import create_retriever_tool
from typing_extensions import Literal
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState


retriever = doc_store.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_context",
    "Search and return information about businesses in Nigeria.",
)


# Bind tools
tools = [retriever_tool]
tools_by_name = {tool.name: tool for tool in tools}

# Bind tools to LLM for agent functionality
llm_with_tools = llm.bind_tools(tools)


# Define the RAG agent system prompt
rag_prompt = """You are a helpful assistant tasked with retrieving information from a context on starting a business in Nigeria. 
Clarify the scope of research with the user before using your retrieval tool to gather context. Reflect on any context you fetch, and
proceed until you have sufficient context to answer the user's research request. 
In all cases, make your aswers as concise as possible and avoid unecessary explanation unless asked."""


def llm_call(state: MessagesState) -> dict:
    """LLM decides whether to call a tool or not.

    Args:
        state: Current conversation state

    Returns:
        Dictionary with new messages
    """
    return {
        "messages": [
            llm_with_tools.invoke(
                [SystemMessage(content=rag_prompt)] + state["messages"]
            )
        ]
    }


def tool_node(state: MessagesState) -> dict:
    """Performs the tool call.

    Args:
        state: Current conversation state with tool calls

    Returns:
        Dictionary with tool results
    """
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call.

    Args:
        state: Current conversation state

    Returns:
        Next node to execute
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        END: END,
    },
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()
