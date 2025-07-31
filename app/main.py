from fastapi import FastAPI, HTTPException
from app.core.models import QueryRequest, QueryResponse
from app.graph import agent_graph
from langchain_core.messages import HumanMessage

app = FastAPI(
    title="Product Query Bot",
    description="A microservice to answer product questions using a multi-agent system",
    version="1.0.0",
)


@app.get("/health", summary="Health Check")
def health_check():
    """Simple health check endpoint to confirm the service is running."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, summary="Process a user query")
async def handle_query(request: QueryRequest):
    """
    Accepts a user query, validates it, and redirects it to the
    multi-agent structure for processing.
    """
    print(f"Received query from user '{request.user_id}': '{request.query}'")

    try:
        # Use user_id as the conversation/thread id for the checkpointer
        config = {"configurable": {"thread_id": request.user_id}}
        print(f"🔧 Using thread_id: {request.user_id}")

        inputs = {"messages": [HumanMessage(content=request.query)]}
        print(f"🔧 Initial inputs: {inputs}")

        final_state = agent_graph.invoke(inputs, config=config)
        final_response = final_state.get("generation") or "Sorry, I couldn't generate a response."

        return QueryResponse(answer=final_response)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(
            status_code=500, detail="An internal error occurred while processing the query."
        )
