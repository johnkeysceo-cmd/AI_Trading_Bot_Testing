"""Phase 10.4: Relation Networks for Meta-Learning
Learns task-aware metric for comparing query and support samples"""

import numpy as np, pandas as pd, json, logging
from dataclasses import dataclass
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RelationMetrics:
    episode_id: int; relation_scores: float; accuracy: float
    def to_dict(self):
        return {'episode_id': self.episode_id, 'avg_relation_score': float(self.relation_scores),
                'accuracy': float(self.accuracy)}

class RelationModule:
    def __init__(self, embedding_dim: int = 32):
        self.w1 = np.random.normal(0, 0.01, (embedding_dim * 2, 64))
        self.b1 = np.zeros(64)
        self.w2 = np.random.normal(0, 0.01, (64, 1))
        self.b2 = np.zeros(1)
    
    def compute_relation(self, query: np.ndarray, support: np.ndarray) -> float:
        combined = np.concatenate([query, support])
        h = np.maximum(0, combined @ self.w1 + self.b1)
        relation = 1.0 / (1.0 + np.exp(-(h @ self.w2 + self.b2)))
        return float(relation[0, 0])

class RelationNetwork:
    def __init__(self, feature_dim: int = 64, embedding_dim: int = 32, num_episodes: int = 200):
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.num_episodes = num_episodes
        self.relation_module = RelationModule(embedding_dim)
        self.metrics_history = []
    
    def embed(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ np.random.normal(0, 0.01, (self.feature_dim, 32)))
        return h @ np.random.normal(0, 0.01, (32, self.embedding_dim))
    
    def predict(self, query_emb: np.ndarray, support_embs: np.ndarray, support_labels: np.ndarray) -> int:
        relations = []
        for sup_emb in support_embs:
            relation = self.relation_module.compute_relation(query_emb, sup_emb)
            relations.append(relation)
        
        relations = np.array(relations)
        max_relation_idx = np.argmax(relations)
        return support_labels[max_relation_idx]
    
    def train(self) -> Dict:
        logger.info(f"Training Relation Networks: {self.num_episodes} episodes")
        
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
            
            correct, rel_scores = 0, []
            for q_emb, q_label in zip(qry_emb, y_qry):
                pred = self.predict(q_emb, sup_emb, y_sup)
                if pred == q_label: correct += 1
                relations = [self.relation_module.compute_relation(q_emb, s) for s in sup_emb]
                rel_scores.append(np.mean(relations))
            
            acc = correct / len(y_qry)
            avg_rel = np.mean(rel_scores)
            self.metrics_history.append(RelationMetrics(episode + 1, avg_rel, acc).to_dict())
            
            if (episode + 1) % 50 == 0:
                logger.info(f"Episode {episode + 1}: Accuracy={acc:.3f}, Avg relation={avg_rel:.3f}")
        
        accs = [m['accuracy'] for m in self.metrics_history]
        return {
            'algorithm': 'Relation-Networks',
            'num_episodes': self.num_episodes,
            'avg_accuracy': float(np.mean(accs)),
            'final_accuracy': float(accs[-1]),
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 10.4: Relation Networks for Meta-Learning")
    logger.info("=" * 60)
    
    network = RelationNetwork(num_episodes=150)
    results = network.train()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Algorithm: {results['algorithm']}")
    logger.info(f"Avg accuracy: {results['avg_accuracy']:.3f}")
    logger.info(f"Final accuracy: {results['final_accuracy']:.3f}")
    
    with open('phase10_relation_networks_results.json', 'w') as f:
        json.dump(results, f, indent=2)
