# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from typing import List, Dict, Any, Optional
import numpy as np
from .vector_store import VectorStore

class WeaviateStore(VectorStore):
    """
    Weaviate vector store implementation.
    """
    def __init__(self, url: str, api_key: Optional[str] = None):
        import weaviate
        auth_config = weaviate.auth.AuthApiKey(api_key=api_key) if api_key else None
        self.client = weaviate.Client(url=url, auth_client_secret=auth_config)
        
        # Create schema if it doesn't exist
        self.client.schema.create_class({
            "class": "TextChunk",
            "vectorizer": "none",
            "properties": [
                {"name": "text", "dataType": ["text"]},
                {"name": "metadata", "dataType": ["object"]}
            ]
        })

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None):
        with self.client.batch as batch:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                properties = {
                    "text": chunk,
                    "metadata": metadata[i] if metadata else {}
                }
                batch.add_data_object(
                    data_object=properties,
                    class_name="TextChunk",
                    vector=embedding
                )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        result = (
            self.client.query
            .get("TextChunk", ["text", "metadata"])
            .with_near_vector({
                "vector": query_embedding,
                "certainty": 0.7
            })
            .with_limit(top_k)
            .do()
        )
        
        return [
            {
                "id": item["_additional"]["id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "distance": 1 - item["_additional"]["certainty"]
            }
            for item in result["data"]["Get"]["TextChunk"]
        ]

    def delete(self, ids: List[str]) -> None:
        for id in ids:
            self.client.data_object.delete(id, "TextChunk")

class QdrantStore(VectorStore):
    """
    Qdrant vector store implementation.
    """
    def __init__(self, url: str, collection_name: str, api_key: Optional[str] = None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        
        # Create collection if it doesn't exist
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None):
        from qdrant_client.models import PointStruct
        
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            meta = metadata[i] if metadata else {}
            meta["text"] = chunk
            points.append(PointStruct(
                id=i,
                vector=embedding,
                payload=meta
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return [
            {
                "id": str(hit.id),
                "text": hit.payload["text"],
                "metadata": {k:v for k,v in hit.payload.items() if k != "text"},
                "distance": hit.score
            }
            for hit in results
        ]

    def delete(self, ids: List[str]) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )
