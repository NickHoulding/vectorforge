from pydantic import ValidationError

from vectorforge import api
from vectorforge.config import Config
from vectorforge.models import SearchQuery

from ..server import mcp
from ..utils import build_error_response, build_success_response


@mcp.tool
def search_documents(query: str, top_k: int = Config.DEFAULT_TOP_K) -> dict:
    """Perform semantic search on indexed documents.
    
    Args:
        query: Search query string.
        top_k: Number of top results to return.
        
    Returns:
        List of search results with document IDs, content, scores, and metadata.
    """
    try:
        search_params = SearchQuery(
            query=query,
            top_k=top_k
        )
        response = api.search(search_params=search_params)
        return build_success_response(response)

    except ValidationError as e:
        return build_error_response(Exception("Invalid input"), details=str(e))
    except ValueError as e:
        return build_error_response(Exception("Invalid query"), details=str(e))
    except Exception as e:
        return build_error_response(Exception("Operation failed"), details=str(e))
