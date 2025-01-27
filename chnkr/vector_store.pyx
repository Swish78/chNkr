# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from abc import ABC, abstractmethod
import chromadb
import pinecone
from neo4j import GraphDatabase
import numpy as np
from typing import List, Dict, Any, Optional

class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    """
    @abstractmethod
    def add_chunks(self, chunks: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        pass

class ChromaStore(VectorStore):
    """
    ChromaDB vector store implementation.
    """
    def __init__(self, collection_name: str, persist_directory: Optional[str] = None):
        self.client = chromadb.Client()
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.create_collection(name=collection_name)

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None):
        ids = [str(i) for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata if metadata else [{}] * len(chunks),
            ids=ids
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return [
            {"id": id, "text": doc, "metadata": meta, "distance": dist}
            for id, doc, meta, dist in zip(
                results["ids"][0], 
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def delete(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)

class PineconeStore(VectorStore):
    """
    Pinecone vector store implementation.
    """
    def __init__(self, api_key: str, environment: str, index_name: str):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None):
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            meta = metadata[i] if metadata else {}
            meta["text"] = chunk
            vectors.append((str(i), embedding, meta))
        
        self.index.upsert(vectors=vectors)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [
            {
                "id": match.id,
                "text": match.metadata["text"],
                "metadata": {k:v for k,v in match.metadata.items() if k != "text"},
                "distance": match.score
            }
            for match in results.matches
        ]

    def delete(self, ids: List[str]) -> None:
        self.index.delete(ids=ids)

class Neo4jStore(VectorStore):
    """
    Neo4j vector store implementation using vector index.
    """
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        
        # Create vector index if it doesn't exist
        with self.driver.session(database=database) as session:
            session.run("""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:Chunk)
                ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 768,
                    `vector.similarity_function`: 'cosine'
                }}
            """)

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None):
        with self.driver.session(database=self.database) as session:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                meta = metadata[i] if metadata else {}
                session.run("""
                    CREATE (c:Chunk {
                        id: $id,
                        text: $text,
                        embedding: $embedding,
                        metadata: $metadata
                    })
                """, {
                    "id": str(i),
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": meta
                })

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            results = session.run("""
                CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $embedding)
                YIELD node, score
                RETURN node.id as id, node.text as text, node.metadata as metadata, score as distance
                ORDER BY score DESC
            """, {"top_k": top_k, "embedding": query_embedding})
            
            return [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "metadata": record["metadata"],
                    "distance": record["distance"]
                }
                for record in results
            ]

    def delete(self, ids: List[str]) -> None:
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (c:Chunk)
                WHERE c.id IN $ids
                DELETE c
            """, {"ids": ids})

    def __del__(self):
        self.driver.close()