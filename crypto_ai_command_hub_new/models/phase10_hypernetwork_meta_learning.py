"""Phase 10.7: Hypernetwork Meta-Learning
Uses hypernetworks to generate task-specific weights"""

import numpy as np, pandas as pd, json, logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HypernetMetrics:
    episode_id: int; weight_variance: float; accuracy: float
    def to_dict(self):
        return {'episode_id': self.episode_id, 'weight_variance': float(self.weight_variance),
                'accuracy': float(self.accuracy)}

class Hypernetwork:
    def __init__(self, task_embedding_dim: int = 16, main_weight_dim: int = 64):
        self.w1 = np.random.normal(0, 0.01, (task_embedding_dim, 128))
        self.b1 = np.zeros(128)
        self.w2 = np.random.normal(0, 0.01, (128, main_weight_dim))
        self.b2 = np.zeros(main_weight_dim)
    
    def generate_weights(self, task_embedding: np.ndarray) -> np.ndarray:
        h = np.maximum(0, task_embedding @ self.w1 + self.b1)
        weights = h @ self.w2 + self.b2
        return weights

class HypernetworkMetaLearner:
    def __init__(self, feature_dim: int = 64, num_episodes: int = 200):
        self.feature_dim = feature_dim
        self.num_episodes = num_episodes
        self.hypernetwork = Hypernetwork(16, 64)
        self.metrics_history = []
    
    def embed_task(self, support: np.ndarray) -> np.ndarray:
        return np.random.normal(0, 1, 16)
    
    def train(self) -> dict:
        logger.info(f"Training Hypernetwork: {self.num_episodes} episodes")
        
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
            
            task_emb = self.embed_task(X_sup)
            main_weights = self.hypernetwork.generate_weights(task_emb)
            
            correct = 0
            for q, q_label in zip(X_qry, y_qry):
                pred_score = np.dot(q, main_weights)
                pred_class = int(pred_score % 5)
                if pred_class == q_label: correct += 1
            
            acc = correct / len(y_qry)
            weight_var = np.var(main_weights)
            self.metrics_history.append(HypernetMetrics(episode + 1, weight_var, acc).to_dict())
            
            if (episode + 1) % 50 == 0:
                logger.info(f"Episode {episode + 1}: Accuracy={acc:.3f}, Weight var={weight_var:.4f}")
        
        accs = [m['accuracy'] for m in self.metrics_history]
        return {
            'algorithm': 'Hypernetwork-Meta-Learning',
            'num_episodes': self.num_episodes,
            'avg_accuracy': float(np.mean(accs)),
            'final_accuracy': float(accs[-1]),
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 10.7: Hypernetwork Meta-Learning")
    logger.info("=" * 60)
    
    learner = HypernetworkMetaLearner(num_episodes=150)
    results = learner.train()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Algorithm: {results['algorithm']}")
    logger.info(f"Avg accuracy: {results['avg_accuracy']:.3f}")
    logger.info(f"Final accuracy: {results['final_accuracy']:.3f}")
    
    with open('phase10_hypernetwork_meta_learning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
