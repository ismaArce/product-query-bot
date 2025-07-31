from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
import app.agents as agents


def test_responder_sets_generation_and_appends_message(monkeypatch):
    """Test that responder_agent sets state['generation'] and appends an AIMessage to history."""
    captured = {"invoke_args": None}

    def fake_chain_invoke(args):
        captured["invoke_args"] = args
        return AIMessage(content="STUB_ANSWER")

    def fake_chain_creation(prompt, chat):
        class FakeChain:
            def invoke(self, args):
                return fake_chain_invoke(args)

        return FakeChain()

    monkeypatch.setattr("app.agents.ChatPromptTemplate.__or__", fake_chain_creation, raising=False)

    docs = [
        Document(
            page_content="Price: $1999",
            metadata={"product_name": "Annibale Colombo Sofa", "chunk": 0},
        )
    ]

    state = {
        "messages": [
            HumanMessage(content="Annibale Colombo Sofa"),
            HumanMessage(content="and what is the price?"),
        ],
        "summarized_messages": [
            HumanMessage(content="Annibale Colombo Sofa"),
            HumanMessage(content="and what is the price?"),
        ],
        "documents": docs,
    }

    out = agents.responder_agent(state)
    assert out.get("generation") == "STUB_ANSWER"
    assert isinstance(out["messages"][-1], AIMessage)


def test_responder_sends_context_to_llm(monkeypatch):
    """Test that document context is sent to the LLM in the prompt."""
    captured = {"invoke_args": None}

    def fake_chain_invoke(args):
        captured["invoke_args"] = args
        return AIMessage(content="OK")

    def fake_chain_creation(prompt, chat):
        class FakeChain:
            def invoke(self, args):
                return fake_chain_invoke(args)

        return FakeChain()

    monkeypatch.setattr("app.agents.ChatPromptTemplate.__or__", fake_chain_creation, raising=False)

    docs = [
        Document(
            page_content="Spec sheet with price $999",
            metadata={"product_name": "AC Chair", "chunk": 1},
        ),
        Document(page_content="Warranty info", metadata={"product_name": "AC Chair", "chunk": 2}),
    ]

    state = {
        "messages": [
            HumanMessage(content="AC Chair"),
            HumanMessage(content="and what is the price?"),
        ],
        "summarized_messages": [
            HumanMessage(content="AC Chair"),
            HumanMessage(content="and what is the price?"),
        ],
        "documents": docs,
    }

    _ = agents.responder_agent(state)
    invoke_args = captured["invoke_args"]
    context_str = invoke_args.get("context", "")
    assert "Spec sheet with price $999" in context_str, "Prompt should include the document context"
