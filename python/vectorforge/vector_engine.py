from sentence_transformers import SentenceTransformer
from typing import Any
from torch import Tensor
import uuid

from models import SearchResult


class VectorEngine:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.embeddings: list[Tensor] = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_to_doc_id = []
        self.doc_id_to_index = {}
        self.deleted_docs = set()

        if self.should_compact():
            self.compact()

    def should_compact(self) -> bool:
        """Decide whether compaction is needed based on ratio of deleted docs"""
        if not self.embeddings:
            return False

        deleted_ratio = len(self.deleted_docs) / len(self.embeddings)
        return deleted_ratio > 0.25

    def compact(self) -> None:
        """Rebuild index and free deleted doc memory"""
        new_embeddings = []
        new_index_to_doc_id = []
        new_doc_id_to_index = {}

        for old_pos, doc_id in enumerate(self.index_to_doc_id):
            if doc_id not in self.deleted_docs:
                new_pos = len(new_embeddings)
                new_embeddings.append(self.embeddings[old_pos])
                new_index_to_doc_id.append(doc_id)
                new_doc_id_to_index[doc_id] = new_pos

        self.embeddings = new_embeddings
        self.index_to_doc_id = new_index_to_doc_id
        self.doc_id_to_index = new_doc_id_to_index
        self.deleted_docs.clear()

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search the vector index based on the query"""
        query_embedding = self.model.encode(query)
        results = []

        for pos, embedding in enumerate(self.embeddings):
            doc_id = self.index_to_doc_id[pos]

            if doc_id in self.deleted_docs:
                continue

            # TODO: Implement the remaining search logic

        return results
    
    def get_doc(self, doc_id: str) -> dict | None:
        """Retreive a doc with the specified doc id"""
        return self.documents.get(doc_id, None)

    def add_doc(
            self, 
            content: str, 
            metadata: dict[str, Any] | None = None
        ) -> str:
        """Adds a new doc with the specified content and metadata"""
        doc_id: str = str(uuid.uuid4())

        self.documents[doc_id] = {
            "content": content, 
            "metadata": metadata or {}
        }

        embedding: Tensor = self.model.encode(content)
        vector_index: int = len(self.embeddings)

        self.embeddings.append(embedding)
        self.index_to_doc_id.append(doc_id)
        self.doc_id_to_index[doc_id] = vector_index

        return doc_id

    def remove_doc(self, doc_id: str) -> bool:
        """Removes the doc with the specified doc_id (lazy deletion)"""
        if doc_id not in self.documents:
            return False

        del self.documents[doc_id]
        self.deleted_docs.add(doc_id)

        if self.should_compact():
            self.compact()

        return True
