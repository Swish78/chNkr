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
    cdef str topic_method

    def __init__(self, style="fixed", max_tokens=100, topic_method='lda', overlap=0, embedding_dim=768, model_name="bert-base-uncased", similarity_threshold=0.8):
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
        Semantic chunking based on sentence embeddings and similarity.
        Groups similar sentences together while respecting max_tokens.
        """
        sentences = text.split('. ')
        if not sentences:
            return []
            
        # Get embeddings for all sentences
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            tokens = len(self.tokenizer.encode(sentence))
            
            if not current_chunk:
                current_chunk.append(sentence)
                current_tokens = tokens
                continue
                
            # Calculate similarity with current chunk
            chunk_embedding = np.mean([embeddings[j] for j, s in enumerate(sentences) if s in current_chunk], axis=0)
            similarity = cosine_similarity([chunk_embedding], [embeddings[i]])[0][0]
            
            if similarity >= self.similarity_threshold and current_tokens + tokens <= self.max_tokens:
                current_chunk.append(sentence)
                current_tokens += tokens
            else:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_tokens = tokens
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            
        return chunks

    cdef list _topic_chunk(self, str text):
        """
        Topic-based chunking using LDA or KMeans clustering.
        """
        sentences = text.split('. ')
        if not sentences:
            return []
            
        # Get embeddings
        embeddings = self.model.encode(sentences)
        
        if self.topic_method == 'lda':
            # Use LDA for topic modeling
            lda = LDA(n_components=max(2, len(sentences) // self.max_tokens))
            sentence_topics = lda.fit_transform(embeddings)
            dominant_topics = sentence_topics.argmax(axis=1)
        else:
            # Use KMeans clustering
            n_clusters = max(2, len(sentences) // self.max_tokens)
            kmeans = KMeans(n_clusters=n_clusters)
            dominant_topics = kmeans.fit_predict(embeddings)
        
        # Group sentences by topic
        topic_groups = {}
        for i, topic in enumerate(dominant_topics):
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].append(sentences[i])
        
        # Create chunks respecting max_tokens
        chunks = []
        for topic_sentences in topic_groups.values():
            current_chunk = []
            current_tokens = 0
            
            for sentence in topic_sentences:
                tokens = len(self.tokenizer.encode(sentence))
                if current_tokens + tokens <= self.max_tokens:
                    current_chunk.append(sentence)
                    current_tokens += tokens
                else:
                    chunks.append('. '.join(current_chunk) + '.')
                    current_chunk = [sentence]
                    current_tokens = tokens
            
            if current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
        
        return chunks

    cdef list _window_chunk(self, str text):
        """
        Window-based chunking with configurable overlap.
        """
        tokens = self.tokenizer.encode(text)
        if not tokens:
            return []
            
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move window considering overlap
            start = end - self.overlap if self.overlap > 0 else end
            
        return chunks

    cdef list _fixed_chunk(self, str text):
        """
        Fixed-size chunking based on token count.
        """
        tokens = self.tokenizer.encode(text)
        if not tokens:
            return []
            
        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks

    cdef list _sentence_chunk(self, str text):
        """
        Sentence-based chunking with token limit and smart sentence grouping.
        """
        sentences = text.split('. ')
        if not sentences:
            return []
            
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            tokens = len(self.tokenizer.encode(sentence))
            
            if tokens > self.max_tokens:
                # Handle very long sentences by using window chunking
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                    current_chunk = []
                    current_tokens = 0
                
                window_chunks = self._window_chunk(sentence)
                chunks.extend(window_chunks)
                continue
            
            if current_tokens + tokens <= self.max_tokens:
                current_chunk.append(sentence)
                current_tokens += tokens
            else:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_tokens = tokens
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            
        return chunks
