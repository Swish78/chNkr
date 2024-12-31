# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA

cdef class Chunker:
    cdef str style
    cdef int max_tokens
    cdef int overlap
    cdef int embedding_dim
    cdef object tokenizer
    cdef object model
    cdef float similarity_threshold

    def __init__(self, style="fixed", max_tokens=100,topic_method='lda', overlap=0, embedding_dim=768, model_name="bert-base-uncased", similarity_threshold=0.8):
        """
        Initialize the Chunker with a specified chunking style, tokenizer/embedding model, and similarity threshold.
        """
        self.topic_method = topic_method
        self.style = style
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.embedding_dim = embedding_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold  # Cosine similarity threshold for grouping sentences

    def chunk(self, str text):
        """
        Perform chunking based on the selected style.
        """
        if self.style == "fixed":
            return self._fixed_chunk(text)
        elif self.style == "semantic":
            return self._semantic_chunk(text)
        elif self.style == "topic":
            return self._topic_chunk(text)
        elif self.style == "sentence":
            return self._sentence_chunk(text)
        elif self.style == "window":
            return self._window_chunk(text)
        else:
            raise ValueError("Invalid chunking style")

    cdef list _semantic_chunk(self, str text):
        """
        Semantic chunking using cosine similarity between sentence embeddings.
        """
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings and extra spaces
        embeddings = self.model.encode(sentences)
        cosine_sim = cosine_similarity(embeddings)
        chunks = []
        visited = [False] * len(sentences)

        for i in range(len(sentences)):
            if visited[i]:
                continue  # Skip already visited sentences
            current_chunk = [sentences[i]]
            visited[i] = True
            for j in range(i + 1, len(sentences)):
                if not visited[j] and cosine_sim[i][j] >= self.similarity_threshold:
                    current_chunk.append(sentences[j])
                    visited[j] = True

            chunks.append(" ".join(current_chunk))

        return chunks

    cdef list _fixed_chunk(self, str text):
        """
        Fixed-token chunking.
        """
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk = self.tokenizer.convert_tokens_to_string(tokens[i:i + self.max_tokens])
            chunks.append(chunk)
        return chunks

    cdef list _window_chunk(self, str text):
        """
        Sliding window chunking with overlapping tokens.
        """
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk = self.tokenizer.convert_tokens_to_string(tokens[start:end])
            chunks.append(chunk)
            start = start + self.max_tokens - self.overlap  # Advance by max_tokens - overlap
        return chunks

    cdef list _topic_chunk(self, str text):
        """
        Topic-based chunking using LDA or clustering.
        """
        sentences = text.split('. ')
        embeddings = self.model.encode(sentences)

        if self.topic_method == "lda":
            # Using LDA for topic modeling
            lda = LDA(n_components=max(1, len(sentences) // self.max_tokens), random_state=42)
            topics = lda.fit_transform(embeddings)
            topic_indices = np.argmax(topics, axis=1)
        elif self.topic_method == "kmeans":
            # Using KMeans clustering
            n_clusters = max(1, len(sentences) // self.max_tokens)
            kmeans = KMeans(n_clusters=n_clusters)
            topic_indices = kmeans.fit_predict(embeddings)
        else:
            raise ValueError("Invalid topic modeling method. Choose 'lda' or 'kmeans'.")

        # Create chunks based on topic clusters
        chunks = []
        for topic in set(topic_indices):
            chunk = " ".join([sentences[i] for i in range(len(sentences)) if topic_indices[i] == topic])
            chunks.append(chunk)
        return chunks

    cdef list _sentence_chunk(self, str text):
        """
        Sentence-based chunking with a token limit.
        """
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            tokenized_sentence = self.tokenizer.tokenize(sentence)
            if current_length + len(tokenized_sentence) > self.max_tokens:
                chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))
                current_chunk = tokenized_sentence
                current_length = len(tokenized_sentence)
            else:
                current_chunk.extend(tokenized_sentence)
                current_length += len(tokenized_sentence)
        if current_chunk:
            chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))
        return chunks
