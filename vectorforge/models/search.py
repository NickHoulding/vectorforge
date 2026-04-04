"""Pydantic models for search queries, results, and responses."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vectorforge.config import VFGConfig


class SearchQuery(BaseModel):
    """Input model for semantic search queries.

    Defines the parameters for performing a semantic similarity search across
    the vector database. Supports filtering by metadata and controlling the
    number of results returned.

    Attributes:
        query: The search text to find semantically similar documents.
        top_k: Maximum number of results to return.
        rerank: Boolean flag for search result re-ranking.
        filters: Optional metadata filters as key-value pairs for narrowing results.
        document_filter: Optional document-text filter using ``$contains`` or
            ``$not_contains``, e.g. ``{"$contains": "machine learning"}``.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is machine learning?",
                "top_k": 5,
                "rerank": True,
                "filters": {"source_file": "textbook.pdf"},
            }
        }
    )

    query: str = Field(
        ...,
        min_length=VFGConfig.MIN_QUERY_LENGTH,
        max_length=VFGConfig.MAX_QUERY_LENGTH,
        description="Search query text",
    )
    top_k: int = Field(
        default=VFGConfig.DEFAULT_TOP_K,
        ge=VFGConfig.MIN_TOP_K,
        le=VFGConfig.MAX_TOP_K,
        description="Number of results to return",
    )
    rerank: bool = Field(
        default=VFGConfig.SHOULD_RERANK,
        description="Optional flag to enable search result re-ranking",
    )
    top_n: int | None = Field(
        default=None,
        ge=VFGConfig.MIN_TOP_N,
        le=VFGConfig.MAX_TOP_N,
        description="Number of re-ranked results to return, if re-ranking is enabled",
    )
    filters: dict[str, Any] | None = Field(
        default=None, description="Optional metadata filters"
    )
    document_filter: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional document-text filter. "
            "Accepts $contains or $not_contains with a string value, "
            'e.g. {"$contains": "machine learning"}.'
        ),
    )

    @model_validator(mode="after")
    def resolve_top_n(self) -> "SearchQuery":
        """Resolve top_n to a concrete value and validate it against top_k.

        When reranking is disabled, top_n is unused - resolve it to the default
        so downstream code always receives an int.

        When reranking is enabled but top_k is 1, reranking is automatically
        disabled. There is nothing to reorder with a single candidate.

        When reranking is enabled, clamp top_n to min(DEFAULT_TOP_N, top_k - 1)
        if it was not explicitly provided, so callers that only set top_k never
        receive a validation error due to the default top_n exceeding top_k.

        Returns:
            SearchQuery: The validated model instance with top_n set to an int.

        Raises:
            ValueError: If an explicit top_n is >= top_k when reranking is enabled.
        """
        if not self.rerank or self.top_k == 1:
            self.rerank = False
            self.top_n = (
                self.top_n if self.top_n is not None else VFGConfig.DEFAULT_TOP_N
            )
            return self

        if self.top_n is None:
            self.top_n = min(VFGConfig.DEFAULT_TOP_N, self.top_k - 1)
        elif self.top_n >= self.top_k:
            raise ValueError(
                f"top_n ({self.top_n}) must be strictly less than top_k ({self.top_k})"
            )

        return self

    @model_validator(mode="after")
    def validate_filter_operators(self) -> "SearchQuery":
        """Validate that operator expressions in filters are well-formed.

        Each filter value may be either a scalar (exact equality match) or a
        dict of operator expressions (e.g. ``{"$gte": 10}``). This validator
        checks that only recognised operators are used and that each operator
        receives a value of the correct type.

        Returns:
            SearchQuery: The validated model instance.

        Raises:
            ValueError: If an unknown operator is used, or an operator receives
                a value of the wrong type.
        """
        if not self.filters:
            return self

        valid_scalar_types = VFGConfig.VALID_SCALAR_TYPES
        valid_operators = VFGConfig.VALID_FILTER_OPERATORS

        for field, filter_value in self.filters.items():
            if not isinstance(filter_value, dict):
                continue

            operators_used = set(filter_value.keys())

            for op in operators_used:
                if op not in valid_operators:
                    raise ValueError(
                        f"filters[{field!r}] contains unknown operator {op!r}. "
                        f"Allowed operators: {sorted(valid_operators)}"
                    )

            if "$in" in operators_used:
                in_operand = filter_value["$in"]
                if not isinstance(in_operand, list):
                    raise ValueError(
                        f"filters[{field!r}]['$in'] must be a list, "
                        f"got {type(in_operand).__name__!r}"
                    )
                invalid_items = [
                    item for item in in_operand if type(item) not in valid_scalar_types
                ]
                if invalid_items:
                    bad_types = {type(item).__name__ for item in invalid_items}
                    raise ValueError(
                        f"filters[{field!r}]['$in'] contains unsupported value types: "
                        f"{sorted(bad_types)}. "
                        f"Allowed types: str, int, float, bool"
                    )

            for scalar_op in ("$gte", "$lte", "$ne"):
                scalar_operand = filter_value.get(scalar_op)
                if (
                    scalar_operand is not None
                    and type(scalar_operand) not in valid_scalar_types
                ):
                    raise ValueError(
                        f"filters[{field!r}][{scalar_op!r}] must be a str, int, float, or bool, "
                        f"got {type(scalar_operand).__name__!r}"
                    )

        return self

    @model_validator(mode="after")
    def validate_document_filter(self) -> "SearchQuery":
        """Validate that document_filter contains only recognised operators with string values.

        Returns:
            SearchQuery: The validated model instance.

        Raises:
            ValueError: If document_filter contains an unknown operator or a non-string value.
        """
        if not self.document_filter:
            return self

        valid_operators = VFGConfig.VALID_DOCUMENT_FILTER_OPERATORS

        for op, operand in self.document_filter.items():
            if op not in valid_operators:
                raise ValueError(
                    f"document_filter contains unknown operator {op!r}. "
                    f"Allowed operators: {sorted(valid_operators)}"
                )
            if not isinstance(operand, str):
                raise ValueError(
                    f"document_filter[{op!r}] must be a str, "
                    f"got {type(operand).__name__!r}"
                )

        return self


class SearchResult(BaseModel):
    """Individual search result with similarity scoring.

    Represents a single document match from a semantic search query, including
    the document's content, metadata, and cosine similarity score indicating
    how closely it matches the search query.

    Attributes:
        id: Unique identifier of the matched document.
        content: Full text content of the document chunk.
        metadata: Document metadata including source file and chunk information.
        score: Cosine similarity score (0-1), where 1 is perfect match.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "abc-123-def",
                "content": "Machine learning is a subset of AI that focuses on...",
                "metadata": {"source_file": "textbook.pdf", "chunk_index": 0},
                "score": 0.89,
            }
        }
    )

    id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Document text content")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Document metadata"
    )
    score: float = Field(..., description="Similarity score")


class SearchResponse(BaseModel):
    """Complete search results response.

    Contains all matching documents from a semantic search query, ranked by
    similarity score in descending order. Includes the original query for
    reference and the total count of results returned.

    Attributes:
        query: The original search query text that was submitted.
        results: List of matching documents ranked by similarity score.
        count: Total number of results returned (may be less than top_k if fewer matches exist).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is machine learning?",
                "results": [
                    {
                        "id": "abc-123",
                        "content": "Machine learning is a subset of AI...",
                        "metadata": {"source_file": "textbook.pdf", "chunk_index": 0},
                        "score": 0.89,
                    },
                    {
                        "id": "def-456",
                        "content": "ML algorithms learn from data patterns...",
                        "metadata": {"source_file": "textbook.pdf", "chunk_index": 1},
                        "score": 0.82,
                    },
                ],
                "count": 2,
            }
        }
    )

    query: str = Field(..., description="Original search query")
    results: list[SearchResult] = Field(..., description="List of search results")
    count: int = Field(..., description="Number of results returned")
