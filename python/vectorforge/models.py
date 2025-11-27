from pydantic import BaseModel
from typing import Optional

class Document(BaseModel):
    id: Optional[str] = None
    content: str
    metadata: Optional[dict] = None

class SearchQuery(BaseModel):
    query: str
    top_k: int=10
    filters: Optional[dict] = None

class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    metadata: Optional[dict] = None
