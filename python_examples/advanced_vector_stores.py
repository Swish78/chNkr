from chnkr import Chunker
from chnkr.advanced_stores import MilvusStore, FAISSStore
from sentence_transformers import SentenceTransformer
import tempfile
import os

def advanced_vector_store_examples():
    # Initialize components
    chunker = Chunker(style="semantic", max_tokens=100)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Sample text
    text = """
    Artificial Intelligence has transformed various industries.
    Machine learning algorithms can analyze vast amounts of data.
    Deep learning models excel at pattern recognition tasks.
    Natural Language Processing enables human-like text understanding.
    Computer Vision systems can detect and classify objects in images.
    Reinforcement Learning allows agents to learn through interaction.
    """
    
    # Generate chunks and embeddings
    chunks = chunker.chunk(text)
    embeddings = model.encode(chunks)
    metadata = [{"source": "AI_overview", "index": i} for i in range(len(chunks))]
    
    # Example with FAISS (local, memory-efficient)
    print("\nFAISS Example:")
    
    # Create a temporary file for FAISS index
    with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as tmp:
        index_path = tmp.name
    
    faiss_store = FAISSStore(index_path=index_path)
    faiss_store.add_chunks(chunks, embeddings, metadata)
    
    # Search example
    query = "How does AI understand human language?"
    query_embedding = model.encode([query])[0]
    results = faiss_store.search(query_embedding, top_k=2)
    
    print("\nQuery:", query)
    print("Top 2 results from FAISS:")
    for r in results:
        print(f"- {r['text']} (distance: {r['distance']:.3f})")
    
    # Example with Milvus (requires Milvus server)
    """
    print("\nMilvus Example:")
    milvus_store = MilvusStore(
        host="localhost",
        port=19530,
        collection_name="ai_chunks"
    )
    milvus_store.add_chunks(chunks, embeddings, metadata)
    
    results = milvus_store.search(query_embedding, top_k=2)
    print("\nTop 2 results from Milvus:")
    for r in results:
        print(f"- {r['text']} (distance: {r['distance']:.3f})")
    """
    
    # Clean up
    os.unlink(index_path)
    if os.path.exists(f"{index_path}.meta"):
        os.unlink(f"{index_path}.meta")

if __name__ == "__main__":
    advanced_vector_store_examples()
