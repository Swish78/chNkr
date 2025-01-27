from chnkr import Chunker
from chnkr.vector_store import ChromaStore, PineconeStore, Neo4jStore
from sentence_transformers import SentenceTransformer

def vector_store_examples():
    # Initialize components
    chunker = Chunker(style="semantic", max_tokens=100)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Sample text
    text = """
    Vector databases are specialized databases designed for storing and searching high-dimensional vectors.
    These vectors often represent embeddings of text, images, or other data types.
    Similarity search in vector databases uses algorithms like HNSW or IVF for efficient retrieval.
    Vector databases are crucial for modern AI applications and recommendation systems.
    """
    
    # Generate chunks and embeddings
    chunks = chunker.chunk(text)
    embeddings = model.encode(chunks)
    
    # Example with ChromaDB
    print("\nChromaDB Example:")
    chroma_store = ChromaStore(collection_name="demo_collection")
    chroma_store.add_chunks(chunks, embeddings)
    
    # Search example
    query = "What are vector databases used for?"
    query_embedding = model.encode([query])[0]
    results = chroma_store.search(query_embedding, top_k=2)
    
    print("\nQuery:", query)
    print("Top 2 results:")
    for r in results:
        print(f"- {r['text']} (distance: {r['distance']:.3f})")
    
    # Example with Pinecone (requires API key)
    """
    pinecone_store = PineconeStore(
        api_key="your-api-key",
        environment="your-environment",
        index_name="demo-index"
    )
    pinecone_store.add_chunks(chunks, embeddings)
    results = pinecone_store.search(query_embedding, top_k=2)
    """
    
    # Example with Neo4j (requires Neo4j installation)
    """
    neo4j_store = Neo4jStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    neo4j_store.add_chunks(chunks, embeddings)
    results = neo4j_store.search(query_embedding, top_k=2)
    """

if __name__ == "__main__":
    vector_store_examples()
