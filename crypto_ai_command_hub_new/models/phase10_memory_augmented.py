"""Phase 10.5: Memory-Augmented Meta-Learning
Uses external memory for storing task-relevant information"""

import numpy as np, pandas as pd, json, logging
from dataclasses import dataclass
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryMetrics:
    episode_id: int; memory_usage: float; recall_accuracy: float
    def to_dict(self):
        return {'episode_id': self.episode_id, 'memory_usage': float(self.memory_usage),
                'recall_accuracy': float(self.recall_accuracy)}

class ExternalMemory:
    def __init__(self, memory_size: int = 100, embedding_dim: int = 32):
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.memory = np.zeros((memory_size, embedding_dim))
        self.memory_labels = np.zeros(memory_size, dtype=int)
        self.write_head = 0
    
    def write_to_memory(self, embedding: np.ndarray, label: int):
        idx = self.write_head % self.memory_size
        self.memory[idx] = embedding
        self.memory_labels[idx] = label
        self.write_head += 1
    
    def read_from_memory(self, query_embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.write_head == 0:
            return np.zeros_like(query_embedding), np.array([])
        
        valid_memory = self.memory[:min(self.write_head, self.memory_size)]
        valid_labels = self.memory_labels[:min(self.write_head, self.memory_size)]
        
        distances = np.linalg.norm(valid_memory - query_embedding, axis=1)
        top_k = min(5, len(distances))
        top_indices = np.argsort(distances)[:top_k]
        
        retrieved_embeddings = valid_memory[top_indices]
        retrieved_labels = valid_labels[top_indices]
        
        return retrieved_embeddings, retrieved_labels
    
    def memory_utilization(self) -> float:
        return min(self.write_head, self.memory_size) / self.memory_size

class MemoryAugmentedNetwork:
    def __init__(self, feature_dim: int = 64, embedding_dim: int = 32, num_episodes: int = 200):
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.num_episodes = num_episodes
        self.memory = ExternalMemory(memory_size=100, embedding_dim=embedding_dim)
        self.metrics_history = []
    
    def embed(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ np.random.normal(0, 0.01, (self.feature_dim, 32)))
        return h @ np.random.normal(0, 0.01, (32, self.embedding_dim))
    
    def predict_with_memory(self, query_emb: np.ndarray) -> int:
        retrieved_embs, retrieved_labels = self.memory.read_from_memory(query_emb)
        
        if len(retrieved_labels) == 0:
            return 0
        
        distances = np.linalg.norm(retrieved_embs - query_emb, axis=1)
        nearest_idx = np.argmin(distances)
        return retrieved_labels[nearest_idx]
    
    def train(self) -> Dict:
        logger.info(f"Training Memory-Augmented Networks: {self.num_episodes} episodes")
        
        for episode in range(self.num_episodes):
            classes = np.random.choice(100, 5, replace=False)
            X_sup_list, y_sup_list = [], []
            X_qry_list, y_qry_list = [], []
            
            for way, cls in enumerate(classes):
                sup = np.random.normal(cls * 0.1, 1, (5, self.feature_dim))
                qry = np.random.normal(cls * 0.1, 1, (5, self.feature_dim))
                X_sup_list.extend(sup); y_sup_list.extend([way] * 5)
                X_qry_list.extend(qry); y_qry_list.extend([way] * 5)
            
            X_sup, y_sup = np.array(X_sup_list), np.array(y_sup_list)
            X_qry, y_qry = np.array(X_qry_list), np.array(y_qry_list)
            
            sup_emb = self.embed(X_sup)
            qry_emb = self.embed(X_qry)
            
            for s_emb, s_label in zip(sup_emb, y_sup):
                self.memory.write_to_memory(s_emb, s_label)
            
            correct = 0
            for q_emb, q_label in zip(qry_emb, y_qry):
                pred = self.predict_with_memory(q_emb)
                if pred == q_label: correct += 1
            
            acc = correct / len(y_qry)
            mem_util = self.memory.memory_utilization()
            self.metrics_history.append(MemoryMetrics(episode + 1, mem_util, acc).to_dict())
            
            if (episode + 1) % 50 == 0:
                logger.info(f"Episode {episode + 1}: Accuracy={acc:.3f}, Memory util={mem_util:.1%}")
        
        accs = [m['recall_accuracy'] for m in self.metrics_history]
        return {
            'algorithm': 'Memory-Augmented-Networks',
            'num_episodes': self.num_episodes,
            'avg_accuracy': float(np.mean(accs)),
            'final_accuracy': float(accs[-1]),
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 10.5: Memory-Augmented Meta-Learning")
    logger.info("=" * 60)
    
    network = MemoryAugmentedNetwork(num_episodes=150)
    results = network.train()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Algorithm: {results['algorithm']}")
    logger.info(f"Avg accuracy: {results['avg_accuracy']:.3f}")
    logger.info(f"Final accuracy: {results['final_accuracy']:.3f}")
    
    with open('phase10_memory_augmented_results.json', 'w') as f:
        json.dump(results, f, indent=2)
