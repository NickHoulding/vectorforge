from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Any
from torch import Tensor
import uuid

class VectorEngine:
    def __init__(self) -> None:
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embeddings: List[Tensor] = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_to_doc_id = []
        self.doc_id_to_index = {}
        self.deleted_docs = set()
    
    def get_doc(self, doc_id: str) -> Union[Dict, None]:
        """Retreive a doc with the specified doc id"""
        return self.documents.get(doc_id, None)

    def add_doc(
            self, 
            content: str, 
            metadata: Union[Dict[str, Any], None] = None
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

        # TODO: Implement compaction to free memory when too many deleted docs accumulate
        # Then, add forced compaction on startup when there is any amount of deleted docs

        return True
