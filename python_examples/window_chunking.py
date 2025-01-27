from chnkr import Chunker

def window_chunking_example():
    # Initialize chunker with window style and overlap
    chunker = Chunker(
        style="window",
        max_tokens=50,
        overlap=10  # 10 token overlap between chunks
    )
    
    # Sample text with continuous context
    text = """
    The development of artificial neural networks was inspired by biological neurons.
    These artificial neurons are organized in layers, with each layer performing specific computations.
    The input layer receives the initial data, while hidden layers process this information through various transformations.
    The output layer produces the final results based on the processed information from previous layers.
    This layered architecture allows neural networks to learn complex patterns and representations from data.
    The connections between neurons are weighted, and these weights are adjusted during the training process.
    Through backpropagation, the network learns to minimize errors in its predictions by updating these weights.
    """
    
    # Generate overlapping chunks
    chunks = chunker.chunk(text)
    
    print("Window-based Chunks with Overlap:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)
        print("-" * 50)
    
    # Show overlap between consecutive chunks
    if len(chunks) >= 2:
        print("\nOverlap Example:")
        chunk1_words = set(chunks[0].split())
        chunk2_words = set(chunks[1].split())
        overlap_words = chunk1_words.intersection(chunk2_words)
        
        print("\nOverlapping words between first two chunks:")
        print(", ".join(overlap_words))

if __name__ == "__main__":
    window_chunking_example()
