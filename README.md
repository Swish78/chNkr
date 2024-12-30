# ChNkr

**ChNkr** is a high-performance NLP library designed to enhance text chunking for large language models (LLMs) and retrieval-augmented generation (RAG) workflows. Built with speed and flexibility in mind, ChNkr supports multiple chunking styles, overlap options, and seamless integration with popular vector databases like Chroma, Pinecone, and Neo4j.

## Features

- **Flexible Chunking Styles**: Choose from fixed-token, semantic-aware, or custom chunking methods.
- **Overlap Support**: Ensure context continuity between chunks with adjustable overlap settings.
- **Vector Store Integration**: Direct support for Chroma, Pinecone, and Neo4j.
- **Blazing Fast Performance**: Built using Cython/C++ for maximum speed.
- **Ease of Use**: Python-friendly API with detailed documentation and examples.

## Installation

### Prerequisites
- Python 3.8+
- GCC/Clang (for building Cython/C++ components)
- Pip or equivalent package manager

### Install from PyPI
```bash
pip install chnkr
```

### Build from Source
```bash
git clone https://github.com/yourusername/chnkr.git
cd chnkr
pip install .
```

## Quick Start

### Basic Usage
```python
from chnkr import Chunker

# Initialize the Chunker with your preferred style
chunker = Chunker(style="semantic", overlap=50)

# Chunk a sample text
text = "Your large text input here..."
chunks = chunker.chunk(text)
print(chunks)
```

### Integration with Chroma
```python
from chnkr import Chunker
from chromadb.client import Client

# Initialize ChNkr
chunker = Chunker(style="fixed", max_tokens=100, overlap=20)

# Chunk your text
text = "Your document text here..."
chunks = chunker.chunk(text)

# Push to Chroma
db_client = Client()
collection = db_client.create_collection("my_collection")
for chunk in chunks:
    collection.add(document=chunk, metadata={"source": "doc_1"})
```

### Integration with Pinecone
```python
from chnkr import Chunker
import pinecone

# Initialize Pinecone
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
index = pinecone.Index("my-index")

# Initialize ChNkr
chunker = Chunker(style="semantic", overlap=50)

# Chunk your text
text = "Your document text here..."
chunks = chunker.chunk(text)

# Push to Pinecone
for chunk in chunks:
    index.upsert([(chunk.id, chunk.vector)])
```

### Integration with Neo4j
```python
from chnkr import Chunker
from neo4j import GraphDatabase

# Initialize Neo4j driver
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# Initialize ChNkr
chunker = Chunker(style="fixed", max_tokens=200)

# Chunk your text
text = "Your document text here..."
chunks = chunker.chunk(text)

# Push to Neo4j
def add_chunk(tx, chunk):
    tx.run("CREATE (c:Chunk {content: $content, metadata: $metadata})", content=chunk.content, metadata=chunk.metadata)

with driver.session() as session:
    for chunk in chunks:
        session.write_transaction(add_chunk, chunk)
```

## Configuration Options

| Parameter      | Description                                       | Default Value |
|----------------|---------------------------------------------------|---------------|
| `style`        | Chunking style (`fixed`, `semantic`, `custom`)    | `fixed`       |
| `max_tokens`   | Maximum tokens per chunk (for `fixed` style)      | `100`         |
| `overlap`      | Overlap size between chunks (in tokens)           | `0`           |
| `embedding_dim`| Embedding size for semantic chunking              | `768`         |

## Development

### Setting Up for Development
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chnkr.git
   cd chnkr
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Build the Cython/C++ components:
   ```bash
   python setup.py build_ext --inplace
   ```

### Running Tests
```bash
pytest tests/
```

## Roadmap

- Add support for more vector stores (e.g., Weaviate, Redis).
- Implement advanced chunking styles (e.g., topic-based chunking).
- Extend support for additional languages beyond English.
- Add CLI support for quick operations from the terminal.

## Contributing

We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

## License

ChNkr is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- [Faiss](https://github.com/facebookresearch/faiss) for semantic chunking.
- [Pinecone](https://www.pinecone.io/) and [Chroma](https://www.trychroma.com/) for vector store integration.
- The open-source NLP community for inspiration and support.
