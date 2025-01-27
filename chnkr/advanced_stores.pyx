# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from typing import List, Dict, Any, Optional
import numpy as np
from .vector_store import VectorStore

class MilvusStore(VectorStore):
    """
    Milvus vector store implementation.
    """
    def __init__(self, host: str, port: int, collection_name: str):
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        
        self.collection_name = collection_name
        connections.connect(host=host, port=port)
        
        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
            ]
            schema = CollectionSchema(fields=fields)
            self.collection = Collection(name=collection_name, schema=schema)
            
            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
        else:
            self.collection = Collection(name=collection_name)
            self.collection.load()

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None):
        entities = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            entities.append([
                i,  # id
                chunk,  # text
                metadata[i] if metadata else {},  # metadata
                embedding  # embedding
            ])
            
        self.collection.insert(entities)
        self.collection.flush()

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "metadata"]
        )
        
        return [
            {
                "id": str(hit.id),
                "text": hit.entity.get('text'),
                "metadata": hit.entity.get('metadata'),
                "distance": hit.distance
            }
            for hit in results[0]
        ]

    def delete(self, ids: List[str]) -> None:
        self.collection.delete(f"id in {[int(id) for id in ids]}")

class FAISSStore(VectorStore):
    """
    FAISS vector store implementation with disk persistence.
    """
    def __init__(self, index_path: Optional[str] = None, dimension: int = 768):
        import faiss
        import os
        
        self.index_path = index_path
        self.dimension = dimension
        
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self._load_metadata()
        else:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product similarity
            self.texts = []
            self.metadata_list = []
    
    def _save_metadata(self):
        if self.index_path:
            import pickle
            metadata = {
                'texts': self.texts,
                'metadata': self.metadata_list
            }
            with open(f"{self.index_path}.meta", 'wb') as f:
                pickle.dump(metadata, f)
    
    def _load_metadata(self):
        import pickle
        with open(f"{self.index_path}.meta", 'rb') as f:
            metadata = pickle.load(f)
            self.texts = metadata['texts']
            self.metadata_list = metadata['metadata']

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None):
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        
        self.texts.extend(chunks)
        self.metadata_list.extend(metadata if metadata else [{} for _ in chunks])
        
        if self.index_path:
            import faiss
            faiss.write_index(self.index, self.index_path)
            self._save_metadata()

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_array, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:  # Valid index
                results.append({
                    "id": str(idx),
                    "text": self.texts[idx],
                    "metadata": self.metadata_list[idx],
                    "distance": float(distance)
                })
        return results

    def delete(self, ids: List[str]) -> None:
        # FAISS doesn't support direct deletion
        # We'll create a new index without the deleted items
        import faiss
        
        id_set = set(int(id) for id in ids)
        new_texts = []
        new_metadata = []
        new_embeddings = []
        
        for i in range(self.index.ntotal):
            if i not in id_set:
                embedding = self.index.reconstruct(i)
                new_embeddings.append(embedding)
                new_texts.append(self.texts[i])
                new_metadata.append(self.metadata_list[i])
        
        self.index = faiss.IndexFlatIP(self.dimension)
        if new_embeddings:
            self.index.add(np.array(new_embeddings))
        self.texts = new_texts
        self.metadata_list = new_metadata
        
        if self.index_path:
            faiss.write_index(self.index, self.index_path)
            self._save_metadata()

    def __del__(self):
        if hasattr(self, 'index_path') and self.index_path:
            self._save_metadata()
