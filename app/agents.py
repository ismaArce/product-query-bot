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


# Import the application settings
from app.core.config import settings

CHAT = ChatOllama(
    model=settings.OLLAMA_MODEL,
    temperature=0,
    base_url=settings.OLLAMA_BASE_URL,
)

EMB = OllamaEmbeddings(
    model=settings.EMBEDDING_MODEL_NAME,
    base_url=settings.OLLAMA_BASE_URL,
)


summarizer_node = SummarizationNode(
    model=CHAT.bind(num_predict=256),
    max_tokens=4096,
    max_summary_tokens=1024,
    token_counter=count_tokens_approximately,
)


def _get_vectorstore() -> Chroma:
    client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
    )
    return Chroma(
        client=client,
        collection_name="products",
        embedding_function=EMB,
    )


def retriever_agent(state: dict) -> dict:
    """
    Uses the summarized history to form an enhanced query and retrieve documents from Chroma.
    Expects:
      - state["summarized_messages"] or state["messages"] (list[AnyMessage])
    Produces:
      - state["enhanced_query"], state["documents"]
    """
    print("---AGENT: RETRIEVER---")

    summarized: List[AnyMessage] = state.get("summarized_messages") or state.get("messages") or []
    raw_messages: List[AnyMessage] = state.get("messages") or []

    latest_user = ""
    for msg in reversed(raw_messages):
        if isinstance(msg, HumanMessage):
            latest_user = msg.content
            break

    condensed = " ".join(getattr(m, "content", "") for m in summarized if getattr(m, "content", ""))

    enhanced_query = (
        (f"{latest_user}. Previous context: {condensed}").strip() if latest_user else condensed
    )

    print(f"🔎 Enhanced query: {enhanced_query}")

    vectorstore = _get_vectorstore()
    docs: List[Document] = vectorstore.similarity_search(enhanced_query, k=settings.RETRIEVAL_TOP_K)

    print(f"📚 Retrieved {len(docs)} documents:")
    # for i, doc in enumerate(docs):
    #     print(f"   📄 Doc {i+1}: {doc.page_content[:100]}...")
    #     print(f"   🏷️  Metadata: {doc.metadata}")

    state["enhanced_query"] = enhanced_query
    state["documents"] = docs

    return state


def responder_agent(state: dict) -> dict:
    """
    Generates the final answer using retrieved documents + summarized chat history.
    Expects:
      - state["documents"], state["summarized_messages"], state["messages"]
    Produces:
      - state["generation"] and appends AI reply to state["messages"]
    """
    print("---AGENT: RESPONDER---")
    docs: List[Document] = state.get("documents") or []
    summarized: List[AnyMessage] = state.get("summarized_messages") or []
    raw_messages: List[AnyMessage] = state.get("messages") or []

    question = ""
    for msg in reversed(raw_messages):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    # Compose doc context
    context_str = "\n\n".join(
        f"[{i+1}] {d.page_content}\nMETA: {d.metadata}" for i, d in enumerate(docs)
    )
    print(f"🧩 Context chars: {len(context_str)}")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful product support bot assistant for Zubale. "
                "Answer the user's question based *only* on the conversation history and/or the following context. "
                "If the information is not in the context, clearly state that you cannot find the answer. "
                "Be concise and do not make up information.\n\nCONTEXT:\n{context}",
            ),
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
    print(f"✅ Generated answer: {answer_text[:120]}...")

    # Append AI reply to the raw history so the next turn can be summarized with it
    state["messages"] = list(raw_messages) + [AIMessage(content=answer_text)]
    state["generation"] = answer_text
    return state
