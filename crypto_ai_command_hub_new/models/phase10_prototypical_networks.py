"""
Phase 10.2: Prototypical Networks for Meta-Learning
Implements metric learning approach to few-shot classification:
- Prototype computation (class centroids)
- Distance-based classification
- Support/query episode sampling
- Efficient adaptation through metric learning
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Metrics for prototype learning"""
    episode_id: int
    support_accuracy: float
    query_accuracy: float
    prototype_coherence: float  # Avg distance within class
    inter_class_separation: float
    
    def to_dict(self):
        return {
            'episode_id': self.episode_id,
            'support_accuracy': float(self.support_accuracy),
            'query_accuracy': float(self.query_accuracy),
            'prototype_coherence': float(self.prototype_coherence),
            'inter_class_separation': float(self.inter_class_separation)
        }


@dataclass
class PrototypicalConfig:
    """Configuration for prototypical networks"""
    num_episodes: int = 200
    num_ways: int = 5  # Number of classes per episode
    num_support_per_way: int = 5
    num_query_per_way: int = 5
    feature_dim: int = 64
    embedding_dim: int = 32
    learning_rate: float = 0.001
    distance_metric: str = "euclidean"  # or "cosine"
    

class EmbeddingNetwork:
    """Learns embedding space for metric learning"""
    
    def __init__(self, input_dim: int, embedding_dim: int):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Embedding network weights
        self.w1 = np.random.normal(0, 0.01, (input_dim, 64))
        self.b1 = np.zeros(64)
        self.w2 = np.random.normal(0, 0.01, (64, embedding_dim))
        self.b2 = np.zeros(embedding_dim)
    
    def embed(self, X: np.ndarray) -> np.ndarray:
        """Embed samples into learned space"""
        h = np.maximum(0, X @ self.w1 + self.b1)
        embedding = h @ self.w2 + self.b2
        return embedding
    
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings"""
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    
    def update_weights(self, gradient_w1: np.ndarray, gradient_w2: np.ndarray, learning_rate: float):
        """Update embedding network"""
        self.w1 -= learning_rate * gradient_w1
        self.w2 -= learning_rate * gradient_w2


class PrototypicalNetwork:
    """Prototypical Networks for few-shot learning"""
    
    def __init__(self, config: PrototypicalConfig):
        self.config = config
        self.embedding_network = EmbeddingNetwork(config.feature_dim, config.embedding_dim)
        self.episode_metrics: List[EpisodeMetrics] = []
        
    def compute_prototypes(self, support_embeddings: np.ndarray, support_labels: np.ndarray,
                          num_ways: int) -> np.ndarray:
        """Compute class prototypes as mean embeddings"""
        prototypes = []
        
        for class_id in range(num_ways):
            class_embeddings = support_embeddings[support_labels == class_id]
            prototype = np.mean(class_embeddings, axis=0)
            prototypes.append(prototype)
        
        return np.array(prototypes)
    
    def compute_distances(self, query_embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        """Compute distances from query samples to prototypes"""
        
        distances = np.zeros((query_embeddings.shape[0], len(prototypes)))
        
        for i, query_emb in enumerate(query_embeddings):
            for j, prototype in enumerate(prototypes):
                if self.config.distance_metric == "euclidean":
                    distances[i, j] = np.linalg.norm(query_emb - prototype)
                elif self.config.distance_metric == "cosine":
                    dot_product = np.dot(query_emb, prototype)
                    distances[i, j] = 1.0 - (dot_product / (np.linalg.norm(query_emb) * np.linalg.norm(prototype) + 1e-6))
        
        return distances
    
    def predict(self, query_embeddings: np.ndarray, prototypes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and confidences"""
        distances = self.compute_distances(query_embeddings, prototypes)
        
        # Predicted labels are closest prototypes
        predicted_labels = np.argmin(distances, axis=1)
        
        # Confidence is inverse of distance
        min_distances = np.min(distances, axis=1)
        confidences = 1.0 / (1.0 + min_distances)
        
        return predicted_labels, confidences
    
    def run_episode(self, support_data: Tuple[np.ndarray, np.ndarray],
                   query_data: Tuple[np.ndarray, np.ndarray],
                   episode_id: int) -> EpisodeMetrics:
        """Run one episode of few-shot learning"""
        
        X_support, y_support = support_data
        X_query, y_query = query_data
        
        # Embed samples
        support_embeddings = self.embedding_network.embed(X_support)
        query_embeddings = self.embedding_network.embed(X_query)
        
        # Normalize if cosine distance
        if self.config.distance_metric == "cosine":
            support_embeddings = self.embedding_network.normalize(support_embeddings)
            query_embeddings = self.embedding_network.normalize(query_embeddings)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, y_support, self.config.num_ways)
        
        # Predict on support set
        support_pred, _ = self.predict(support_embeddings, prototypes)
        support_accuracy = np.mean(support_pred == y_support)
        
        # Predict on query set
        query_pred, query_conf = self.predict(query_embeddings, prototypes)
        query_accuracy = np.mean(query_pred == y_query)
        
        # Prototype coherence (average intra-class distance)
        intra_class_distances = []
        for class_id in range(self.config.num_ways):
            class_embeddings = support_embeddings[y_support == class_id]
            if len(class_embeddings) > 1:
                distances = np.linalg.norm(class_embeddings[:, None, :] - class_embeddings[None, :, :], axis=2)
                avg_dist = np.mean(distances[np.triu_indices_from(distances, k=1)])
                intra_class_distances.append(avg_dist)
        
        prototype_coherence = np.mean(intra_class_distances) if intra_class_distances else 0.0
        
        # Inter-class separation
        inter_class_distances = []
        for i in range(len(prototypes)):
            for j in range(i + 1, len(prototypes)):
                dist = np.linalg.norm(prototypes[i] - prototypes[j])
                inter_class_distances.append(dist)
        
        inter_class_separation = np.mean(inter_class_distances) if inter_class_distances else 0.0
        
        metrics = EpisodeMetrics(
            episode_id=episode_id,
            support_accuracy=support_accuracy,
            query_accuracy=query_accuracy,
            prototype_coherence=prototype_coherence,
            inter_class_separation=inter_class_separation
        )
        
        return metrics
    
    def train(self) -> Dict:
        """Train prototypical networks"""
        
        logger.info(f"Training prototypical networks")
        logger.info(f"Episodes: {self.config.num_episodes}, Ways: {self.config.num_ways}, "
                   f"Support: {self.config.num_support_per_way}, Query: {self.config.num_query_per_way}")
        
        for episode_id in range(self.config.num_episodes):
            # Sample episode
            classes = np.random.choice(100, size=self.config.num_ways, replace=False)
            
            # Support set
            X_support_list = []
            y_support_list = []
            
            for way_id, class_id in enumerate(classes):
                samples = np.random.normal(class_id * 0.1, 1, (self.config.num_support_per_way, self.config.feature_dim))
                X_support_list.append(samples)
                y_support_list.extend([way_id] * self.config.num_support_per_way)
            
            X_support = np.vstack(X_support_list)
            y_support = np.array(y_support_list)
            
            # Query set
            X_query_list = []
            y_query_list = []
            
            for way_id, class_id in enumerate(classes):
                samples = np.random.normal(class_id * 0.1, 1, (self.config.num_query_per_way, self.config.feature_dim))
                X_query_list.append(samples)
                y_query_list.extend([way_id] * self.config.num_query_per_way)
            
            X_query = np.vstack(X_query_list)
            y_query = np.array(y_query_list)
            
            # Run episode
            metrics = self.run_episode(
                support_data=(X_support, y_support),
                query_data=(X_query, y_query),
                episode_id=episode_id + 1
            )
            
            self.episode_metrics.append(metrics)
            
            if (episode_id + 1) % 50 == 0:
                logger.info(f"Episode {episode_id + 1}: Support acc={metrics.support_accuracy:.3f}, "
                           f"Query acc={metrics.query_accuracy:.3f}")
        
        return self.get_final_results()
    
    def get_final_results(self) -> Dict:
        """Compile results"""
        
        support_accs = [m.support_accuracy for m in self.episode_metrics]
        query_accs = [m.query_accuracy for m in self.episode_metrics]
        
        return {
            'algorithm': 'Prototypical-Networks',
            'num_episodes': self.config.num_episodes,
            'num_ways': self.config.num_ways,
            'support_per_way': self.config.num_support_per_way,
            'query_per_way': self.config.num_query_per_way,
            'distance_metric': self.config.distance_metric,
            'avg_support_accuracy': float(np.mean(support_accs)),
            'avg_query_accuracy': float(np.mean(query_accs)),
            'final_query_accuracy': float(query_accs[-1]) if query_accs else 0.0,
            'episode_metrics': [m.to_dict() for m in self.episode_metrics]
        }


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 10.2: Prototypical Networks for Meta-Learning")
    logger.info("=" * 60)
    
    config = PrototypicalConfig(
        num_episodes=150,
        num_ways=5,
        num_support_per_way=5,
        num_query_per_way=5,
        feature_dim=64,
        embedding_dim=32,
        distance_metric="euclidean"
    )
    
    network = PrototypicalNetwork(config)
    results = network.train()
    
    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Algorithm: {results['algorithm']}")
    logger.info(f"Distance metric: {results['distance_metric']}")
    logger.info(f"Avg support accuracy: {results['avg_support_accuracy']:.3f}")
    logger.info(f"Avg query accuracy: {results['avg_query_accuracy']:.3f}")
    logger.info(f"Final query accuracy: {results['final_query_accuracy']:.3f}")
    
    # Save results
    with open('phase10_prototypical_networks_results.json', 'w') as f:
        json.dump(results, f, indent=2)
