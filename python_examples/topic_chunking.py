from chnkr import Chunker

def topic_chunking_example():
    # Initialize chunker with topic style
    chunker = Chunker(
        style="topic",
        max_tokens=100,
        topic_method='lda'  # or 'kmeans'
    )
    
    # Sample text with multiple topics
    text = """
    Climate change is causing global temperatures to rise at an unprecedented rate.
    Greenhouse gas emissions from human activities are the main driver of climate change.
    Renewable energy sources like solar and wind power can help reduce carbon emissions.
    
    Machine learning models are becoming increasingly sophisticated.
    Deep neural networks can now perform complex tasks like image recognition and language translation.
    Transfer learning allows models to leverage knowledge from pre-trained networks.
    
    Space exploration has entered a new era with private companies leading the way.
    SpaceX has revolutionized rocket technology with reusable boosters.
    NASA's Artemis program aims to return humans to the Moon by 2025.
    """
    
    # Generate chunks based on topics
    chunks = chunker.chunk(text)
    
    print("Topic-based Chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nTopic Chunk {i}:")
        print(chunk)
        print("-" * 50)

if __name__ == "__main__":
    topic_chunking_example()
