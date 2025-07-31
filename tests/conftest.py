import pytest
import subprocess
import time
import httpx

# Register custom marks to avoid warnings
pytest_plugins = []


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line("markers", "integration: mark test as integration test")


@pytest.fixture(scope="session", autouse=True)
def override_settings_for_tests():
    from app.core.config import settings
    import app.agents as agents
    import app.graph as graph_mod
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langmem.short_term import SummarizationNode

    original_values = {
        "OLLAMA_BASE_URL": settings.OLLAMA_BASE_URL,
        "CHROMA_HOST": settings.CHROMA_HOST,
    }
    original_chat = agents.CHAT
    original_emb = agents.EMB
    original_sum = agents.summarizer_node
    original_graph = getattr(graph_mod, "agent_graph", None)

    settings.OLLAMA_BASE_URL = "http://localhost:11434"
    settings.CHROMA_HOST = "localhost"

    agents.CHAT = ChatOllama(
        model=settings.OLLAMA_MODEL, temperature=0, base_url=settings.OLLAMA_BASE_URL
    )
    agents.EMB = OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME, base_url=settings.OLLAMA_BASE_URL
    )

    agents.summarizer_node = SummarizationNode(
        model=agents.CHAT.bind(num_predict=256),
        max_tokens=512,
        max_tokens_before_summary=512,
        max_summary_tokens=256,
    )

    graph_mod.agent_graph = graph_mod._build_agent_graph()

    yield

    for k, v in original_values.items():
        setattr(settings, k, v)
    agents.CHAT = original_chat
    agents.EMB = original_emb
    agents.summarizer_node = original_sum
    if original_graph is not None:
        graph_mod.agent_graph = original_graph


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment with ChromaDB container."""
    try:
        subprocess.run(
            ["docker-compose", "up", "-d", "chromadb"], check=True, capture_output=True, text=True
        )

        for _ in range(30):
            try:
                with httpx.Client() as client:
                    response = client.get("http://localhost:8000/api/v2/heartbeat", timeout=1.0)
                    if response.status_code == 200:
                        break
            except (httpx.RequestError, httpx.TimeoutException):
                pass
            time.sleep(1)
        else:
            raise Exception("ChromaDB may not be fully ready")

    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to start ChromaDB container: {e}")

    yield

    try:
        subprocess.run(
            ["docker-compose", "stop", "chromadb"], check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to stop ChromaDB container: {e}")
