import sys
import os
import pandas as pd
import chromadb
from langchain_core.documents import Document

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.logger import get_logger
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

logger = get_logger(__name__)


def ingest_documents() -> None:
    """Ingest product data from CSV into ChromaDB vector store.

    Loads product data from CSV file, processes it into LangChain documents
    with structured metadata, and stores embeddings in ChromaDB for semantic search.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        Exception: If ChromaDB connection or ingestion fails.
    """
    logger.info("Starting document ingestion process")

    try:
        df = pd.read_csv("data/product_description.csv")
        logger.info("Successfully loaded %d rows from CSV", len(df))
    except FileNotFoundError:
        logger.error("CSV file 'data/product_description.csv' not found")
        return
    except Exception as e:
        logger.error("Failed to load CSV file: %s", e)
        return

    documents = []
    for _, row in df.iterrows():
        page_content = (
            f"Product Name: {row.get('title', 'N/A')}\n"
            f"Description: {row.get('description', 'N/A')}\n"
            f"Category: {row.get('category', 'N/A')}\n"
            f"Brand: {row.get('brand', 'N/A')}"
        )

        metadata = row.to_dict()

        cleaned_metadata = {}
        for key, value in metadata.items():
            if pd.isna(value) or value == "":
                cleaned_metadata[key] = "N/A"
            elif isinstance(value, (int, float)):
                cleaned_metadata[key] = value
            else:
                str_value = str(value)
                if len(str_value) > 1000:
                    str_value = str_value[:1000] + "..."
                cleaned_metadata[key] = str_value

        doc = Document(page_content=page_content, metadata=cleaned_metadata)
        documents.append(doc)

    logger.info("Created %d documents for ingestion", len(documents))

    try:
        ollama_ef = OllamaEmbeddingFunction(
            url=settings.OLLAMA_BASE_URL,
            model_name=settings.EMBEDDING_MODEL_NAME,
        )

        logger.info("Connecting to ChromaDB at %s:%s", settings.CHROMA_HOST, settings.CHROMA_PORT)
        db_client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )
        collection = db_client.get_or_create_collection(
            name="products",
            embedding_function=ollama_ef,
        )

        ids = [f"product_{row['id']}" for _, row in df.iterrows()]

        collection.add(
            ids=ids,
            documents=[doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )

        logger.info("Successfully ingested %d documents into 'products' collection", len(documents))

    except Exception as e:
        logger.error("Failed to ingest documents into ChromaDB: %s", e)
        raise


if __name__ == "__main__":
    ingest_documents()
