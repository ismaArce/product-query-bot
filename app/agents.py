import chromadb
from typing import List
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import AnyMessage
from app.core.config import settings
from app.core.logger import get_logger
from langgraph.graph import MessagesState
from langmem.short_term import RunningSummary

logger = get_logger(__name__)

CHAT = ChatOllama(
    model=settings.OLLAMA_MODEL,
    temperature=0,
    base_url=settings.OLLAMA_BASE_URL,
)

EMB = OllamaEmbeddings(
    model=settings.EMBEDDING_MODEL_NAME,
    base_url=settings.OLLAMA_BASE_URL,
)


class State(MessagesState):
    """Multi-agent conversation state tracking messages and context."""

    context: dict[str, RunningSummary] | None  # Conversation summaries by context key
    summarized_messages: List[AnyMessage] | None  # Condensed message history
    documents: List[Document] | None  # Retrieved documents from vector store
    enhanced_query: str | None  # Query enhanced with conversation context
    generation: str | None  # Final generated response


summarizer_node = SummarizationNode(
    model=CHAT.bind(num_predict=256),
    max_tokens=4096,
    max_summary_tokens=1024,
    token_counter=count_tokens_approximately,
)


def _get_vectorstore() -> Chroma:
    """Initialize ChromaDB vector store client.

    Returns:
        Chroma: Configured ChromaDB vector store instance.
    """
    client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
    )
    return Chroma(
        client=client,
        collection_name="products",
        embedding_function=EMB,
    )


def retriever_agent(state: State) -> State:
    """Retrieve relevant documents using enhanced query from conversation context.

    Combines the latest user message with summarized conversation history to create
    an enhanced search query, then retrieves the most relevant documents from the
    vector store.

    Args:
        state (State): Current conversation state containing messages and summaries.

    Returns:
        State: Updated state with enhanced_query and retrieved documents.
    """
    logger.info("Starting document retrieval")

    summarized: List[AnyMessage] = state.get("summarized_messages") or state.get("messages") or []
    raw_messages: List[AnyMessage] = state.get("messages") or []

    latest_user = ""
    for msg in reversed(raw_messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            latest_user = msg.content.strip()
            break

    condensed = " ".join(getattr(m, "content", "") for m in summarized if getattr(m, "content", ""))

    # Handle empty or whitespace-only queries
    if not latest_user or latest_user.isspace():
        enhanced_query = condensed if condensed else "general product inquiry"
    else:
        enhanced_query = (f"{latest_user}. Previous context: {condensed}").strip()

    logger.debug("Enhanced query: %s", enhanced_query)

    vectorstore = _get_vectorstore()
    docs: List[Document] = vectorstore.similarity_search(enhanced_query, k=settings.RETRIEVAL_TOP_K)

    logger.info("Retrieved %d documents from vector store", len(docs))

    state["enhanced_query"] = enhanced_query
    state["documents"] = docs

    return state


def _has_clear_product_context(summarized: List[AnyMessage], question: str) -> bool:
    """Check if there's a clear product context in the conversation.

    Args:
        summarized: List of summarized conversation messages
        question: Current user question

    Returns:
        bool: True if there's a clear product context, False otherwise
    """
    elliptical_patterns = [
        "and what",
        "what about",
        "how about",
        "and how",
        "and when",
        "and where",
        "what's the",
        "what is the",
        "how's the",
        "how is the",
        "when's the",
        "when is the",
        "where's the",
        "where is the",
        "and the",
        "the price",
        "the warranty",
        "the shipping",
        "the stock",
        "the rating",
        "the reviews",
        "the brand",
        "the category",
    ]

    is_elliptical = any(
        question.lower().startswith(pattern.lower()) for pattern in elliptical_patterns
    )

    if not is_elliptical:
        return True

    conversation_text = " ".join(
        getattr(m, "content", "") for m in summarized if getattr(m, "content", "")
    )

    product_indicators = [
        "product",
        "item",
        "chair",
        "sofa",
        "bed",
        "table",
        "lamp",
        "watch",
        "shirt",
        "shoes",
        "laptop",
        "phone",
        "furniture",
        "electronics",
        "clothing",
        "accessories",
        "beauty",
        "fragrances",
        "groceries",
        "kitchen",
        "home",
        "decoration",
        "appliances",
    ]

    has_product_context = any(
        indicator.lower() in conversation_text.lower() for indicator in product_indicators
    )

    return has_product_context


def responder_agent(state: State) -> State:
    """Generate final response using retrieved documents and conversation context.

    Creates a contextual response by combining retrieved documents with the
    conversation history, then generates an answer using the configured LLM.

    Args:
        state (State): Current conversation state with documents and messages.

    Returns:
        State: Updated state with generated response and updated message history.
    """
    logger.info("Generating response")
    docs: List[Document] = state.get("documents") or []
    summarized: List[AnyMessage] = state.get("summarized_messages") or []
    raw_messages: List[AnyMessage] = state.get("messages") or []

    question = ""
    for msg in reversed(raw_messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            question = msg.content.strip()
            break

    if not question or question.isspace():
        question = "Please provide information about available products"

    has_context = _has_clear_product_context(summarized, question)

    context_str = "\n\n".join(
        f"[{i+1}] {d.page_content}\nMETA: {d.metadata}" for i, d in enumerate(docs)
    )
    logger.debug("Context length: %d characters", len(context_str))
    logger.debug("Has clear product context: %s", has_context)

    # Determine if we should provide context based on whether we have clear product context
    if has_context:
        system_prompt = (
            "You are a helpful product support bot assistant for Zubale. "
            "Answer the user's question based *only* on the conversation history and/or the following context. "
            "If the information is not in the context, clearly state that you cannot find the answer. "
            "Be concise and do not make up information.\n\n"
            "CONTEXT:\n{context}"
        )
    else:
        system_prompt = (
            "You are a helpful product support bot assistant for Zubale. "
            "The user has asked an elliptical question without clear product context. "
            "You MUST ask for clarification about which specific product they want information about. "
            "Do NOT provide information about any random product from the context. "
            "Instead, ask them to specify which product they're referring to.\n\n"
            "Available context (but do not use unless user clarifies):\n{context}"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="summarized_messages"),
            ("human", "{question}"),
        ]
    )

    chain = prompt | CHAT

    response = chain.invoke(
        {
            "summarized_messages": summarized,
            "context": context_str,
            "question": question,
        }
    )

    answer_text = getattr(response, "content", str(response))
    logger.info("Generated response (%d chars): %s...", len(answer_text), answer_text[:120])

    state["messages"] = list(raw_messages) + [AIMessage(content=answer_text)]
    state["generation"] = answer_text
    return state
