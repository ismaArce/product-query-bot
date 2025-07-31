from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
import app.agents as agents


class FakeRetriever:
    def __init__(self, docs: List[Document]):
        self.docs = docs

    def invoke(self, _query: str) -> List[Document]:
        return self.docs


class FakeVectorStore:
    def __init__(self, docs: List[Document]):
        self._docs = docs

    def as_retriever(self, search_type=None, search_kwargs=None):
        return FakeRetriever(self._docs)

    def similarity_search(self, _query: str, k: int = 4) -> List[Document]:
        return self._docs[:k]


def test_retriever_sets_documents_and_enhanced_query(monkeypatch):
    """Test that retriever_agent builds enhanced_query and sets documents in state."""
    docs = [
        Document(
            page_content="Annibale Colombo Sofa: premium leather, price section …",
            metadata={"product_name": "Annibale Colombo Sofa", "chunk": 0},
        )
    ]

    def fake_get_vecstore():
        return FakeVectorStore(docs)

    monkeypatch.setattr(agents, "_get_vectorstore", fake_get_vecstore, raising=False)

    state = {
        "messages": [
            HumanMessage(content="Annibale Colombo Sofa"),
            HumanMessage(content="and what is the price?"),
        ]
    }

    out = agents.retriever_agent(state)
    assert out.get("documents"), "documents should be set"
    assert isinstance(out["documents"][0], Document)
    eq = out.get("enhanced_query", "")
    assert "price" in eq.lower(), f"enhanced_query should include latest question; got: {eq!r}"


def test_retriever_uses_latest_human_message(monkeypatch):
    """Test that enhanced_query uses the latest human message and doesn't ignore it."""
    docs = [Document(page_content="dummy doc", metadata={"product_name": "X"})]

    def fake_get_vecstore():
        return FakeVectorStore(docs)

    monkeypatch.setattr(agents, "_get_vectorstore", fake_get_vecstore, raising=False)

    state = {
        "messages": [
            HumanMessage(content="First message about Another Product"),
            AIMessage(content="Ok."),
            HumanMessage(content="Now a different thing: Annibale Colombo Sofa"),
            HumanMessage(content="and what is the price?"),
        ]
    }
    out = agents.retriever_agent(state)
    eq = out.get("enhanced_query", "")
    assert (
        "and what is the price?" in eq.lower()
    ), "enhanced_query must include the latest human question"
