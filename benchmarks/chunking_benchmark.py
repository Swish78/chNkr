import time
import numpy as np
from typing import List, Dict
from chnkr import Chunker
from sentence_transformers import SentenceTransformer

class ChunkingBenchmark:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunking_styles = ["fixed", "semantic", "topic", "window", "sentence"]
        self.results: Dict[str, Dict] = {}
        
    def generate_test_data(self, num_samples: int = 100, min_words: int = 50, max_words: int = 500) -> List[str]:
        """Generate synthetic test data."""
        words = ["machine", "learning", "artificial", "intelligence", "data", "science",
                "neural", "networks", "deep", "learning", "computer", "vision", "natural",
                "language", "processing", "transformer", "model", "training", "dataset",
                "algorithm", "optimization", "classification", "regression", "clustering"]
        
        texts = []
        for _ in range(num_samples):
            length = np.random.randint(min_words, max_words)
            text = " ".join(np.random.choice(words, size=length))
            texts.append(text)
            
        return texts
        
    def benchmark_chunking_style(self, style: str, texts: List[str], max_tokens: int = 100) -> Dict:
        """Benchmark a specific chunking style."""
        chunker = Chunker(style=style, max_tokens=max_tokens)
        
        # Measure chunking time
        start_time = time.time()
        all_chunks = []
        for text in texts:
            chunks = chunker.chunk(text)
            all_chunks.extend(chunks)
        chunking_time = time.time() - start_time
        
        # Calculate metrics
        avg_chunk_length = np.mean([len(chunk.split()) for chunk in all_chunks])
        chunk_length_std = np.std([len(chunk.split()) for chunk in all_chunks])
        
        # Calculate semantic coherence using embeddings
        start_time = time.time()
        embeddings = self.model.encode(all_chunks)
        similarities = []
        for i in range(len(embeddings)-1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
            similarities.append(sim)
        coherence_time = time.time() - start_time
        
        return {
            "chunking_time": chunking_time,
            "coherence_time": coherence_time,
            "num_chunks": len(all_chunks),
            "avg_chunk_length": avg_chunk_length,
            "chunk_length_std": chunk_length_std,
            "avg_coherence": np.mean(similarities) if similarities else 0,
            "coherence_std": np.std(similarities) if similarities else 0
        }
    
    def run_benchmarks(self, num_samples: int = 100):
        """Run benchmarks for all chunking styles."""
        print("Generating test data...")
        texts = self.generate_test_data(num_samples)
        
        for style in self.chunking_styles:
            print(f"\nBenchmarking {style} chunking style...")
            self.results[style] = self.benchmark_chunking_style(style, texts)
            
        self.print_results()
    
    def print_results(self):
        """Print benchmark results in a formatted table."""
        print("\nBenchmark Results:")
        print("-" * 100)
        headers = ["Style", "Chunk Time", "Coherence Time", "Num Chunks", "Avg Length", "Length STD", "Coherence", "Coherence STD"]
        print("{:<10} {:<12} {:<14} {:<12} {:<12} {:<12} {:<10} {:<12}".format(*headers))
        print("-" * 100)
        
        for style, metrics in self.results.items():
            row = [
                style,
                f"{metrics['chunking_time']:.3f}s",
                f"{metrics['coherence_time']:.3f}s",
                metrics['num_chunks'],
                f"{metrics['avg_chunk_length']:.1f}",
                f"{metrics['chunk_length_std']:.1f}",
                f"{metrics['avg_coherence']:.3f}",
                f"{metrics['coherence_std']:.3f}"
            ]
            print("{:<10} {:<12} {:<14} {:<12} {:<12} {:<12} {:<10} {:<12}".format(*row))

if __name__ == "__main__":
    benchmark = ChunkingBenchmark()
    benchmark.run_benchmarks(num_samples=50)  # Adjust sample size as needed
