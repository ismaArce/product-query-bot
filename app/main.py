from fastapi import FastAPI, HTTPException
from app.core.logger import get_logger
from app.core.models import QueryRequest, QueryResponse
from app.graph import agent_graph
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

logger = get_logger(__name__)

app = FastAPI(
    title="Product Query Bot",
    description="A microservice to answer product questions using a multi-agent system",
    version="1.0.0",
)


@app.get("/health", summary="Health Check")
def health_check() -> dict[str, str]:
    """Check service health status.

    Returns:
        dict[str, str]: Status message indicating service is operational.
    """
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, summary="Process a user query")
async def handle_query(request: QueryRequest) -> QueryResponse:
    """Process user query through multi-agent system.

    Args:
        request (QueryRequest): User query with user_id and query text.

    Returns:
        QueryResponse: Generated answer from the multi-agent system.

    Raises:
        HTTPException: 500 error if query processing fails.
    """
    logger.info("Received query from user '%s': '%s'", request.user_id, request.query)

    try:
        config: RunnableConfig = {"configurable": {"thread_id": request.user_id}}
        logger.debug("Using thread_id: %s", request.user_id)

        inputs = {"messages": [HumanMessage(content=request.query)]}
        logger.debug("Processing inputs: %s", inputs)

        final_state = agent_graph.invoke(inputs, config=config)  # type: ignore
        final_response = final_state.get("generation") or "Sorry, I couldn't generate a response."

        return QueryResponse(answer=final_response)

    except Exception as e:
        logger.error("Error processing query for user %s: %s", request.user_id, e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal error occurred while processing the query."
        )
