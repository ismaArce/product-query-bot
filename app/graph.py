from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver
from app.core.logger import get_logger
import app.agents as agents

logger = get_logger(__name__)


def _build_agent_graph() -> StateGraph:
    """Construct the multi-agent workflow graph.

    Creates a sequential workflow: summarizer -> retriever -> responder
    with in-memory checkpointing for conversation state persistence.

    Returns:
        StateGraph: Compiled agent graph ready for execution.
    """
    checkpointer = InMemorySaver()
    builder = StateGraph(agents.State)

    builder.add_node("summarizer", agents.summarizer_node)
    builder.add_node("retriever", agents.retriever_agent)
    builder.add_node("responder", agents.responder_agent)

    builder.add_edge(START, "summarizer")
    builder.add_edge("summarizer", "retriever")
    builder.add_edge("retriever", "responder")
    builder.add_edge("responder", END)

    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Agent graph compiled successfully")
    logger.debug("Checkpointer type: %s", type(checkpointer).__name__)
    logger.debug("State keys: %s", list(agents.State.__annotations__.keys()))

    return graph


agent_graph = _build_agent_graph()
