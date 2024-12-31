ChNkr/
├── chnkr/                        # Cython modules
│   ├── __init__.py               # Package initialization
│   ├── chunker.pyx               # Core chunking logic in Cython
│   ├── utils.pyx                 # Utility functions for preprocessing, tokenization, etc.
│   ├── models.pyx                # Support for custom user-provided models
│   ├── vector_store.pyx          # Vector store integration classes (Chroma, Pinecone, Neo4j)
│   ├── embeddings.pyx            # Embedding generation logic (semantic chunking)
│   ├── exceptions.pyx            # Custom exceptions for library errors
├── tests/                        # Unit and integration tests
│   ├── test_chunker.py           # Testing chunking functionality
│   ├── test_vector_store.py      # Testing vector store integrations
│   ├── test_models.py            # Testing customizable models
│   ├── test_utils.py             # Testing utility functions
│   └── test_embeddings.py        # Testing semantic embedding logic
├── benchmarks/                   # Performance benchmarks
│   ├── benchmark_chunking.py     # Measure chunking performance for large-scale inputs
│   ├── benchmark_vector_store.py # Measure vector store integration accuracy/times
├── python_examples/              # Python examples for library usage
│   ├── example_chunking.py       # A demo for chunking text
│   ├── example_pinecone.py       # How to use ChNkr with Pinecone
│   ├── example_custom_model.py   # Using a user-provided semantic model
│   └── example_chroma.py         # Storing chunks with Chroma
├── docs/                         # Documentation folder
│   ├── overview.md               # Introduction to the library
│   ├── usage.md                  # Usage examples and API documentation
│   ├── installation.md           # Installation and dependencies guide
├── setup.py                      # For Python package installation
├── pyproject.toml                # Build system configuration for Cython
├── README.md                     # Main documentation and introduction to ChNkr
├── LICENSE                       # License for the library