from typing import List, Dict, Any
from torch import Tensor

class VectorEngine:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = []

    def add_docs(self, chunks: List[Dict[str, Any]], embeddings: List[Tensor]) -> bool:
        raise NotImplementedError