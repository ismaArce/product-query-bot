import sys
import os
import pandas as pd
import chromadb
from langchain_core.documents import Document

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)


def ingest_documents():
    """
    Reads product data from a CSV, creates documents with metadata,
    and stores their embeddings in a persistent ChromaDB vector store.
    """
    print("--- Starting Document Ingestion ---")

    # 1. Load Data from CSV using pandas
    try:
        df = pd.read_csv("data/product_description.csv")
        print(f"✅ Successfully loaded {len(df)} rows from data/product_description.csv.")
    except FileNotFoundError:
        print("❌ Error: 'data/product_description.csv' not found. Please ensure the file exists.")
        return

    documents = []
    # 2. Create LangChain Documents from DataFrame Rows
    for _, row in df.iterrows():
        # Combine the most descriptive text fields into a single
        # string for semantic embedding and retrieval.
        page_content = (
            f"Product Name: {row.get('title', 'N/A')}\n"
            f"Description: {row.get('description', 'N/A')}\n"
            f"Category: {row.get('category', 'N/A')}\n"
            f"Brand: {row.get('brand', 'N/A')}"
        )

        # Keep all other columns as structured metadata. This is crucial
        # for the LLM to answer specific questions about price, stock, etc.
        metadata = row.to_dict()

        # Clean metadata values to ensure they're compatible with ChromaDB
        cleaned_metadata = {}
        for key, value in metadata.items():
            if pd.isna(value) or value == "":
                cleaned_metadata[key] = "N/A"
            elif isinstance(value, (int, float)):
                cleaned_metadata[key] = value
            else:
                # Convert to string and ensure it's not too long
                str_value = str(value)
                if len(str_value) > 1000:  # Limit string length
                    str_value = str_value[:1000] + "..."
                cleaned_metadata[key] = str_value

        # Create a Document object
        doc = Document(page_content=page_content, metadata=cleaned_metadata)
        documents.append(doc)

    print(f"📄 Created {len(documents)} documents to be ingested.")

    # 3. Initialize the Embedding Model
    ollama_ef = OllamaEmbeddingFunction(
        url="http://host.docker.internal:11434",
        model_name=settings.EMBEDDING_MODEL_NAME,
    )
    # 4. Initialize ChromaDB and Add Documents
    print(f"📦 Connecting to ChromaDB service at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")
    db_client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
    )
    collection = db_client.get_or_create_collection(
        name="products",
        embedding_function=ollama_ef,
    )

    # Create unique IDs for each document before adding
    ids = [f"product_{row['id']}" for _, row in df.iterrows()]

    collection.add(
        ids=ids,
        documents=[doc.page_content for doc in documents],
        metadatas=[doc.metadata for doc in documents],
    )

    print(f"✅ Successfully ingested {len(documents)} documents into the 'products' collection.")
    print("--- Ingestion Complete ---")


if __name__ == "__main__":
    ingest_documents()
