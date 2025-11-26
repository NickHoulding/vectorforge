from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from torch import Tensor
import uuid

class VectorEngine:
    def __init__(self) -> None:
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embeddings: List[Tensor] = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_to_doc_id = []
        self.doc_id_to_index = {}

    def add_doc(
            self, 
            content: str, 
            metadata: Dict[str, Any] | None = None
        ) -> str:
        """Adds the doc to the vector index"""
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
