from typing import Any

from pydantic import BaseModel, Field, model_validator


class StandardMetadata(BaseModel):
    """Standard metadata fields for document chunks.

    IMPORTANT: This model serves as DOCUMENTATION for the standard metadata
    structure used in VectorForge. It is NOT used for runtime validation of
    search filters - filters can use source_file, chunk_index, or both
    independently without restriction.

    PURPOSE:
    - Documents the conventional metadata structure for chunks
    - Provides type hints and IDE autocomplete support
    - Can be used with create_metadata() helper for document creation

    PAIRING RULE (for document creation only):
    When creating documents via add_doc(), if you include source_file or
    chunk_index in metadata, BOTH must be provided together. This ensures
    proper document tracking and reconstruction.

    FILTERING (no pairing requirement):
    When filtering search results, you can filter by:
    - source_file alone → returns all chunks from that file
    - chunk_index alone → returns all chunks at that index (any file)
    - both together → returns specific chunk from specific file
    - neither → no filtering applied

    Attributes:
        source_file: Filename or path of the source document.
        chunk_index: Zero-based index of the chunk within the source file.

    Raises:
        ValueError: If only one of source_file or chunk_index is provided
                   when using this model directly (creation only, not filtering).
    """

    source_file: str | None = Field(
        default=None, description="Source document filename or path"
    )
    chunk_index: int | None = Field(
        default=None, description="Zero-based chunk index within source file", ge=0
    )

    @model_validator(mode="after")
    def validate_paired_fields(self) -> "StandardMetadata":
        """Ensure source_file and chunk_index are both set or both None."""
        has_source = self.source_file is not None
        has_index = self.chunk_index is not None

        if has_source != has_index:
            raise ValueError(
                "source_file and chunk_index must both be provided or both be omitted"
            )

        return self

    class ConfigDict:
        json_schema_extra = {
            "example": {"source_file": "document.pdf", "chunk_index": 0}
        }


def create_metadata(**kwargs: Any) -> dict[str, Any]:
    """Create a metadata dictionary with optional standard and custom fields.

    This is a convenience function for building metadata dictionaries when
    CREATING documents. It validates that standard fields (source_file,
    chunk_index) follow the pairing rule if provided.

    NOTE: This function is for document creation only. Search filters do NOT
    require pairing - you can filter by source_file alone, chunk_index alone,
    or both together.

    Args:
        **kwargs: Metadata key-value pairs. Can include standard fields
                 (source_file, chunk_index) and any custom fields.

    Returns:
        Dictionary containing all provided metadata fields.

    Raises:
        ValueError: If only one of source_file or chunk_index is provided
                   (for document creation, both or neither required).

    Examples:
        >>> create_metadata(source_file="doc.pdf", chunk_index=0)
        {'source_file': 'doc.pdf', 'chunk_index': 0}

        >>> create_metadata(source_file="doc.pdf", chunk_index=0, author="Alice")
        {'source_file': 'doc.pdf', 'chunk_index': 0, 'author': 'Alice'}

        >>> create_metadata(custom_field="value")
        {'custom_field': 'value'}
    """
    source_file = kwargs.get("source_file")
    chunk_index = kwargs.get("chunk_index")

    if source_file is not None or chunk_index is not None:
        StandardMetadata(source_file=source_file, chunk_index=chunk_index)

    return kwargs
