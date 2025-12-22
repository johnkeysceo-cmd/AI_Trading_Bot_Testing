"""Phase 10.3: Matching Networks for Few-Shot Learning
Attention-based metric learning for few-shot tasks"""

import numpy as np, pandas as pd, json, logging
from dataclasses import dataclass
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchingMetrics:
    episode_id: int; attention_entropy: float; matching_accuracy: float
    def to_dict(self):
        return {'episode_id': self.episode_id, 'attention_entropy': float(self.attention_entropy),
                'matching_accuracy': float(self.matching_accuracy)}

class AttentionModule:
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        self.w1 = np.random.normal(0, 0.01, (input_dim * 2, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.normal(0, 0.01, (hidden_dim, 1))
        self.b2 = np.zeros(1)
    
    def compute_attention(self, query: np.ndarray, support_batch: np.ndarray) -> np.ndarray:
        attention_weights = []
        for support_sample in support_batch:
            combined = np.concatenate([query, support_sample])
            h = np.maximum(0, combined @ self.w1 + self.b1)
            score = h @ self.w2 + self.b2
            attention_weights.append(score.flatten()[0])
        
        weights = np.array(attention_weights)
        weights = np.exp(weights) / (np.sum(np.exp(weights)) + 1e-6)
        return weights

class MatchingNetwork:
    def __init__(self, feature_dim: int = 64, embedding_dim: int = 32, num_episodes: int = 200):
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.num_episodes = num_episodes
        self.attention = AttentionModule(embedding_dim)
        self.metrics_history = []
    
    def embed_samples(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ np.random.normal(0, 0.01, (self.feature_dim, 32)))
        return h @ np.random.normal(0, 0.01, (32, self.embedding_dim))
    
    def match_query_to_support(self, query_embedding: np.ndarray, support_embeddings: np.ndarray,
                               support_labels: np.ndarray) -> int:
        attention_weights = self.attention.compute_attention(query_embedding, support_embeddings)
        weighted_labels = np.bincount(support_labels, weights=attention_weights, minlength=5)
        return np.argmax(weighted_labels)
    
    def train(self) -> Dict:
        logger.info(f"Training Matching Networks: {self.num_episodes} episodes")
        
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
            
            sup_emb = self.embed_samples(X_sup)
            qry_emb = self.embed_samples(X_qry)
            
            correct = 0
            entropy = []
            for q_emb, q_label in zip(qry_emb, y_qry):
                pred = self.match_query_to_support(q_emb, sup_emb, y_sup)
                if pred == q_label: correct += 1
                attn = self.attention.compute_attention(q_emb, sup_emb)
                ent = -np.sum(attn * np.log(attn + 1e-6))
                entropy.append(ent)
            
            acc = correct / len(y_qry)
            avg_ent = np.mean(entropy)
            self.metrics_history.append(MatchingMetrics(episode + 1, avg_ent, acc).to_dict())
            
            if (episode + 1) % 50 == 0:
                logger.info(f"Episode {episode + 1}: Accuracy={acc:.3f}, Entropy={avg_ent:.3f}")
        
        accs = [m['matching_accuracy'] for m in self.metrics_history]
        return {
            'algorithm': 'Matching-Networks',
            'num_episodes': self.num_episodes,
            'avg_accuracy': float(np.mean(accs)),
            'final_accuracy': float(accs[-1]),
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 10.3: Matching Networks for Few-Shot Learning")
    logger.info("=" * 60)
    
    network = MatchingNetwork(num_episodes=150)
    results = network.train()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Algorithm: {results['algorithm']}")
    logger.info(f"Avg accuracy: {results['avg_accuracy']:.3f}")
    logger.info(f"Final accuracy: {results['final_accuracy']:.3f}")
    
    with open('phase10_matching_networks_results.json', 'w') as f:
        json.dump(results, f, indent=2)
