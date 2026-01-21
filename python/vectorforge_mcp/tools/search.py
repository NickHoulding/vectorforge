import httpx

from pydantic import ValidationError

from vectorforge import api
from vectorforge.config import Config
from vectorforge.models import SearchQuery

from ..instance import mcp
from ..utils import build_error_response, build_success_response


@mcp.tool(
    description="Semantic search across indexed documents using embeddings. Returns top-k most similar results with scores and metadata."
)
def search_documents(query: str, top_k: int = Config.DEFAULT_TOP_K) -> dict:
    """Perform semantic search on indexed documents.
    
    Args:
        query: Search query string (natural language).
        top_k: Number of top results to return (default: 10, max: 100).
        
    Returns:
        List of search results with document IDs, content, similarity scores, and metadata.
    """
    try:
        search_params = SearchQuery(
            query=query,
            top_k=top_k
        )
        response = api.search(search_params=search_params)
        return build_success_response(response)

    except ConnectionRefusedError:
        return build_error_response(
            Exception("VectorForge API is not available"),
            details="Connection refused - check if API is running"
        )
    except (ConnectionError, httpx.ConnectError):
        return build_error_response(
            Exception("Network error"),
            details="Unable to connect to VectorForge API"
        )
    except (TimeoutError, httpx.TimeoutException):
        return build_error_response(
            Exception("Search timeout"),
            details="Search operation timed out - try reducing top_k"
        )
    except ValidationError as e:
        return build_error_response(
            Exception("Invalid input"), 
            details=str(e)
        )
    except ValueError as e:
        return build_error_response(
            Exception("Invalid query"), 
            details=str(e)
        )
    except Exception as e:
        return build_error_response(
            Exception("Operation failed"), 
            details=str(e)
        )
