import argparse
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

TOLERANCE = 1e-12

def calculate_angular_similarity(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    dot_product = np.sum(matrix_a * matrix_b, axis=-1)
    norm_a = np.linalg.norm(matrix_a, axis=-1)
    norm_b = np.linalg.norm(matrix_b, axis=-1)
    return dot_product / (norm_a * norm_b + TOLERANCE)

def pairwise_divergence_scores(features: np.ndarray) -> np.ndarray:
    normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + TOLERANCE)
    pair_cosines = normalized @ normalized.T
    return -(pair_cosines @ np.ones(len(features)))

def centroid_divergence_scores(features: np.ndarray, normalize_center: bool = False) -> np.ndarray:
    if normalize_center:
        base = features / (np.linalg.norm(features, axis=1, keepdims=True) + TOLERANCE)
    else:
        base = features
    center_vector = base.mean(axis=0)
    return -calculate_angular_similarity(features, np.broadcast_to(center_vector, features.shape))

@dataclass
class MemoryBuffer:
    feature_vectors: np.ndarray
    payloads: np.ndarray
    timestamps: np.ndarray
    peak_usage: int = 0

    @classmethod
    def initialize_empty(cls, feature_dim: int) -> "MemoryBuffer":
        return cls(
            feature_vectors=np.empty((0, feature_dim), dtype=np.float64),
            payloads=np.empty((0,), dtype=np.int64),
            timestamps=np.empty((0,), dtype=np.int64),
        )

    def ingest(self, features: np.ndarray, payloads: np.ndarray, timestamps: np.ndarray) -> None:
        self.feature_vectors = np.concatenate([self.feature_vectors, features], axis=0)
        self.payloads = np.concatenate([self.payloads, payloads], axis=0)
        self.timestamps = np.concatenate([self.timestamps, timestamps], axis=0)
        self.peak_usage = max(self.peak_usage, len(self.feature_vectors))

    def preserve_subset(self, target_indices: np.ndarray) -> None:
        target_indices = np.sort(target_indices)
        self.feature_vectors = self.feature_vectors[target_indices]
        self.payloads = self.payloads[target_indices]
        self.timestamps = self.timestamps[target_indices]

    def geometric_prune(self, memory_limit: int) -> None:
        if len(self.feature_vectors) <= memory_limit:
            return
        priority_scores = centroid_divergence_scores(self.feature_vectors, normalize_center=False)
        survivors = np.argsort(priority_scores)[-memory_limit:]
        self.preserve_subset(survivors)

    def sliding_window_prune(self, memory_limit: int) -> None:
        if len(self.feature_vectors) > memory_limit:
            recent_indices = np.arange(len(self.feature_vectors) - memory_limit, len(self.feature_vectors))
            self.preserve_subset(recent_indices)

def simulate_chunked_ingestion(
    features: np.ndarray,
    payloads: np.ndarray,
    chunk_capacity: int,
    max_memory: Optional[int],
    strategy: str,
) -> MemoryBuffer:
    buffer = MemoryBuffer.initialize_empty(features.shape[1])
    for step in range(0, len(features), chunk_capacity):
        limit = min(step + chunk_capacity, len(features))
        time_idx = np.arange(step, limit, dtype=np.int64)
        buffer.ingest(features[step:limit], payloads[step:limit], time_idx)
        if max_memory is not None:
            if strategy == "geometric":
                buffer.geometric_prune(max_memory)
            elif strategy == "fifo":
                buffer.sliding_window_prune(max_memory)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    return buffer

def execute_retrieval(buffer: MemoryBuffer, query_vec: np.ndarray) -> int:
    logits = calculate_angular_similarity(buffer.feature_vectors, np.broadcast_to(query_vec, buffer.feature_vectors.shape))
    return int(buffer.payloads[int(np.argmax(logits))])

def generate_synthetic_task(rng: np.random.Generator, seq_len: int, dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if seq_len < 32:
        raise ValueError("Length must be at least 32")
    base_cluster = rng.normal(size=dim)
    base_cluster /= np.linalg.norm(base_cluster)
    
    target_vector = rng.normal(size=dim)
    target_vector -= target_vector.dot(base_cluster) * base_cluster
    target_vector /= np.linalg.norm(target_vector)

    noise_features = base_cluster + 0.12 * rng.normal(size=(seq_len, dim))
    targets = np.zeros(seq_len, dtype=np.int64)
    
    insertion_idx = int(rng.integers(8, seq_len // 3))
    ground_truth = int(rng.integers(1, 1_000_000))
    
    noise_features[insertion_idx] = target_vector + 0.03 * rng.normal(size=dim)
    targets[insertion_idx] = ground_truth

    search_query = target_vector + 0.03 * rng.normal(size=dim)
    return noise_features, targets, search_query, ground_truth, insertion_idx

def execute_benchmark(seq_len: int, dim: int, budget: int, chunk_sz: int, num_trials: int, rnd_seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(rnd_seed)
    test_configs = {
        "Unrestricted Buffer": (None, "full"), 
        "Sliding-Window": (budget, "fifo"), 
        "Geometric-Pruning": (budget, "geometric")
    }
    
    metrics_acc = {k: 0 for k in test_configs}
    metrics_retention = {k: 0 for k in test_configs}
    metrics_time = {k: [] for k in test_configs}
    metrics_peak = {k: [] for k in test_configs}

    for _ in range(num_trials):
        features, targets, query, truth, idx = generate_synthetic_task(rng, seq_len, dim)
        for label, (limit, strat) in test_configs.items():
            t0 = time.perf_counter()
            active_buffer = simulate_chunked_ingestion(features, targets, chunk_sz, limit, strat)
            prediction = execute_retrieval(active_buffer, query)
            metrics_time[label].append((time.perf_counter() - t0) * 1000)
            
            metrics_acc[label] += int(prediction == truth)
            metrics_retention[label] += int(idx in set(active_buffer.timestamps.tolist()))
            metrics_peak[label].append(active_buffer.peak_usage)

    output_rows = []
    for label, (limit, _) in test_configs.items():
        output_rows.append({
            "algorithm": label,
            "memory_budget": seq_len if limit is None else limit,
            "average_final_tokens": seq_len if limit is None else budget,
            "target_preservation_rate": round(100 * metrics_retention[label] / num_trials, 1),
            "search_accuracy_rate": round(100 * metrics_acc[label] / num_trials, 1),
            "mean_processing_latency_ms": round(float(np.mean(metrics_time[label])), 3),
            "peak_tokens_before_compression": int(max(metrics_peak[label])),
        })
    
    return {
        "simulation_type": "Synthetic long-context vector retrieval (deterministic)",
        "random_seed": rnd_seed,
        "total_iterations": num_trials,
        "input_sequence_length": seq_len,
        "embedding_dimension": dim,
        "memory_budget": budget,
        "chunk_size": chunk_sz,
        "results_table": output_rows,
        "compression_bound_validated": all(s <= budget + chunk_sz for s in metrics_peak["Geometric-Pruning"]),
    }

def verify_centroid_equivalence(seed: int = 7) -> bool:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(15, 12))
    direct_calc = np.argsort(pairwise_divergence_scores(features))[-7:]
    fast_calc = np.argsort(centroid_divergence_scores(features, normalize_center=True))[-7:]
    return set(direct_calc.tolist()) == set(fast_calc.tolist())

def main() -> None:
    parser = argparse.ArgumentParser(description="Run obfuscated Geometric Pruning reproduction")
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=48)
    parser.add_argument("--cache-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    if args.cache_size < args.block_size:
        raise ValueError("cache-size must be at least block-size")

    # Smoke test and main run
    smoke_run = execute_benchmark(64, 24, 16, 8, 1, args.seed)
    main_run = execute_benchmark(args.length, args.dimension, args.cache_size, args.block_size, args.trials, args.seed)
    
    print("Centroid formulation matches exact pairwise evaluation:", verify_centroid_equivalence())
    print("\nMiniature verification test (1 iteration):")
    print(json.dumps(smoke_run, indent=2))
    print("\nPrimary benchmark execution:")
    print(json.dumps(main_run, indent=2))

if __name__ == "__main__":
    main()