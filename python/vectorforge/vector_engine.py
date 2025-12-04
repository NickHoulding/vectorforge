from sentence_transformers import SentenceTransformer
from typing import Any
import numpy as np
import uuid

from models import SearchResult


class VectorEngine:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.embeddings: list[np.ndarray] = []
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

        for doc_id in self.deleted_docs:
            if doc_id in self.documents:
                del self.documents[doc_id]
        
        self.deleted_docs.clear()

    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Calculates the cosine similarity between pre-normalized embeddings"""
        return np.dot(emb_a, emb_b)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search the vector index based on the query"""
        if not self.embeddings:
            return []

        query_embedding: np.ndarray = self.model.encode(
            sentences=query, 
            convert_to_numpy=True
        )
        normalized_query_embedding: np.ndarray = query_embedding / np.linalg.norm(query_embedding)
        results = []

        for pos, embedding in enumerate(self.embeddings):
            doc_id = self.index_to_doc_id[pos]

            if doc_id in self.deleted_docs:
                continue

            score: float = self.cosine_similarity(
                emb_a=normalized_query_embedding, 
                emb_b=embedding
            )
            results.append((pos, score))

        results.sort(
            key=lambda result: result[1], 
            reverse=True
        )

        search_results = []
        for pos, score in results[:top_k]:
            doc_id = self.index_to_doc_id[pos]
            doc = self.documents[doc_id]

            search_results.append(SearchResult(
                id=doc_id,
                content=doc["content"],
                metadata=doc["metadata"],
                score=score
            ))

        return search_results
    
    def list_files(self) -> list[str]:
        """List all files"""
        filenames = set()

        for doc_id, doc in self.documents.items():
            filename = doc.get("metadata", {}).get("source_file", "")
            
            if doc_id in self.deleted_docs or filename == "":
                continue
            else:
                filenames.add(filename)

        filenames = list(filenames)
        filenames.sort()

        return filenames
    
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

        embedding: np.ndarray = self.model.encode(content, convert_to_numpy=True)
        normalized_embedding: np.ndarray = embedding / np.linalg.norm(embedding)
        vector_index: int = len(self.embeddings)

        self.embeddings.append(normalized_embedding)
        self.index_to_doc_id.append(doc_id)
        self.doc_id_to_index[doc_id] = vector_index

        return doc_id

    def remove_doc(self, doc_id: str) -> bool:
        """Removes the doc with the specified doc_id (lazy deletion)"""
        if doc_id not in self.documents:
            return False

        self.deleted_docs.add(doc_id)

        if self.should_compact():
            self.compact()

        return True
