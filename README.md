# Product Query Bot

A RAG-based multi-agent system for answering product-related questions using FastAPI, LangGraph, and ChromaDB.

## Project Overview

This service implements a product query system that leverages:

- **Multi-Agent Architecture**: Uses LangGraph to orchestrate a pipeline of specialized agents
- **Retrieval-Augmented Generation (RAG)**: Combines vector similarity search with LLM generation
- **Conversation Memory**: Maintains context across user interactions with conversation summarization
- **Local LLM Integration**: Uses Ollama for both embeddings and text generation

### Architecture

The system follows a three-stage pipeline:

1. **Summarizer Agent**: Condenses conversation history for context preservation
2. **Retriever Agent**: Enhances queries with context and retrieves relevant documents
3. **Responder Agent**: Generates contextual responses using retrieved information

## Features

- **FastAPI REST API** with automatic documentation
- **Vector-based document retrieval** using ChromaDB
- **Conversation state persistence** with thread-based memory
- **Comprehensive logging** for debugging and monitoring
- **Docker containerization** for easy deployment
- **Unit test coverage** with pytest

## Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **Ollama** (running locally)
- **uv** (Python package manager)

### Ollama Setup

1. Install Ollama: https://ollama.ai/
2. Pull required models:
   ```bash
   ollama pull gemma3n:e2b
   ollama pull mxbai-embed-large
   ```
3. Ensure Ollama is running on `localhost:11434`

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/ismaArce/product-query-bot.git
cd product-query-bot
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
# ChromaDB Configuration
CHROMA_HOST=chromadb
CHROMA_PORT=8000
VECTOR_STORE_PATH=./chroma_db

# Retrieval Settings
RETRIEVAL_TOP_K=3

# Ollama Configuration
EMBEDDING_MODEL_NAME=nomic-embed-text
OLLAMA_MODEL=gemma3n:e2b
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### 3. Data Preparation

Place your product data in `data/product_description.csv` with the following structure:

- Product information in CSV format
- The ingestion script will process this into the vector database

### 4. Docker Deployment

```bash
# Start all services
docker-compose up -d

# Ingest product data (run once)
docker-compose run --rm product-bot sh -c "python scripts/ingest.py"

```

### 5. API Access

- **Application**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health
- **ChromaDB**: http://localhost:8000

## Development Setup

### Local Development

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Run development server
uvicorn app.main:app --reload --port 8080

# In separate terminal, start ChromaDB
docker run -p 8000:8000 chromadb/chroma

# Ingest data
python scripts/ingest.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_agents.py -v

# Run tests with detailed output
pytest -v -s
```

## API Usage

### Query Endpoint

**POST** `/query`

```json
{
  "user_id": "unique_user_identifier",
  "query": "What is the price of the wireless headphones?"
}
```

**Response:**

```json
{
  "answer": "The Apple AirPods are priced at $129.99."
}
```

### Health Check

**GET** `/health`

**Response:**

```json
{
  "status": "ok"
}
```

## Project Structure

```
product-query-bot/
├── app/                      # Application code
│   ├── core/                 # Core utilities
│   │   ├── config.py         # Configuration management
│   │   ├── logger.py         # Logging setup
│   │   └── models.py         # Pydantic models
│   ├── agents.py             # Multi-agent logic
│   ├── graph.py              # LangGraph workflow
│   └── main.py               # FastAPI application
├── data/                     # Product data files
├── scripts/                  # Utility scripts
│   └── ingest.py             # Data ingestion
├── tests/                    # Test suite
│   ├── conftest.py           # Test fixtures
│   ├── test_retrieval_unit.py # Retrieval tests
│   ├── test_prompt_unit.py   # Prompt tests
│   └── test_api_basic.py     # API tests
├── docker-compose.yaml       # Service orchestration
├── Dockerfile                # Container definition
└── pyproject.toml            # Project configuration
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**

   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags

   # Restart Ollama service
   ollama serve
   ```

2. **ChromaDB Connection Error**

   ```bash
   # Restart ChromaDB container
   docker-compose restart chromadb

   # Check ChromaDB logs
   docker-compose logs chromadb
   ```

3. **Port Conflicts**

   ```bash
   # Check what's using the ports
   lsof -i :8080  # Application port
   lsof -i :8000  # ChromaDB port
   ```

4. **Data Ingestion Issues**

   ```bash
   # Check if CSV file exists
   ls -la data/product_description.csv

   # Run ingestion
   docker-compose run --rm product-bot sh -c "python scripts/ingest.py"
   ```
