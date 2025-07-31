
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Defines the structure for an incoming query."""
    user_id: str = Field(..., description="Unique identifier for the user")
    query: str = Field(..., min_length=1, description="The user's question about a product.")

class QueryResponse(BaseModel):
    """Defines the structure for the response to the user's query."""
    answer: str