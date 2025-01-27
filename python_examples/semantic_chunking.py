from chnkr import Chunker
from sentence_transformers import SentenceTransformer

def semantic_chunking_example():
    # Initialize chunker with semantic style
    chunker = Chunker(
        style="semantic",
        max_tokens=100,
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.7
    )
    
    # Sample text
    text = """
    Machine learning is a subset of artificial intelligence that focuses on developing systems that can learn from data.
    Deep learning is a type of machine learning that uses neural networks with multiple layers.
    Neural networks are inspired by the human brain's structure and function.
    Convolutional neural networks are particularly effective for image processing tasks.
    Natural language processing uses various techniques to understand human language.
    Transformer models have revolutionized NLP with their attention mechanisms.
    BERT and GPT are two popular transformer-based language models.
    """
    
    # Generate chunks
    chunks = chunker.chunk(text)
    
    print("Semantic Chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)

if __name__ == "__main__":
    semantic_chunking_example()
