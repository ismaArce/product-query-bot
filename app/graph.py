from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import RunningSummary
from langchain_core.messages import AnyMessage
from langmem.short_term import SummarizationNode
from langchain_ollama import ChatOllama
from app.core.config import settings
from langchain_core.messages.utils import count_tokens_approximately

# Import the agent functions that will act as nodes
from .agents import retriever_agent, summarizer_node, responder_agent


class State(MessagesState):
    context: dict[str, RunningSummary] | None
    summarized_messages: List[AnyMessage] | None
    documents: List[Document] | None
    enhanced_query: str | None
    generation: str | None


checkpointer = InMemorySaver()

builder = StateGraph(State)


# Add the agent nodes to the graph
builder.add_node("summarizer", summarizer_node)
builder.add_node("retriever", retriever_agent)
builder.add_node("responder", responder_agent)

# Define the edges that control the flow of work
builder.add_edge(START, "summarizer")
builder.add_edge("summarizer", "retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", END)

# Compile the graph with the memory checkpointer
agent_graph = builder.compile(checkpointer=checkpointer)

print("✅ Agentic graph with in-memory checkpointer compiled successfully.")
print("🔧 Checkpointer configuration:")
print(f"   - Type: {type(checkpointer).__name__}")
print(f"   - State keys: {list(State.__annotations__.keys())}")
