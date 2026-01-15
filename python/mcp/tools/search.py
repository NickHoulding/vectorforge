from pydantic import ValidationError

from vectorforge import api
from vectorforge.config import Config
from vectorforge.models import SearchQuery

from ..server import mcp


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

        return {
            "success": True,
            "query": query,
            "results": [
                {
                    "id": result.id,
                    "content": result.content,
                    "metadata": result.metadata,
                    "score": result.score
                }
                for result in response.results
            ],
            "count": response.count
        }

    except ValidationError as e:
        return {
            "success": False,
            "error": "Invalid input",
            "details": str(e)
        }
    except ValueError as e:
        return {
            "success": False,
            "error": "Invalid query",
            "details": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Operation failed",
            "details": str(e)
        }
