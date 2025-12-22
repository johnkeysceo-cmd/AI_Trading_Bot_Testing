"""
Phase 9.3: Personalized Federated Learning
Implements federated learning with client-specific models through:
- Per-client hyperparameter adaptation
- Personalization layers
- Client clustering for similar users
- Mixture of global and local models
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
import logging
from datetime import datetime
import scipy.stats as stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PersonalizationMetrics:
    """Metrics for personalized FL"""
    client_id: str
    personalization_level: float  # [0,1] - how much local vs global
    global_loss: float
    local_loss: float
    adaptation_rate: float
    cluster_id: int = -1
    
    def to_dict(self):
        return {
            'client_id': self.client_id,
            'personalization_level': float(self.personalization_level),
            'global_loss': float(self.global_loss),
            'local_loss': float(self.local_loss),
            'adaptation_rate': float(self.adaptation_rate),
            'cluster_id': int(self.cluster_id)
        }


@dataclass
class PersonalizedConfig:
    """Configuration for personalized federated learning"""
    num_clients: int = 10
    num_rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 32
    global_lr: float = 0.01
    local_lr: float = 0.05  # Local learning rate can differ
    personalization_alpha: float = 0.5  # Tradeoff: 0=global, 1=local
    num_clusters: int = 3
    similarity_threshold: float = 0.7
    

class PersonalizedClient:
    """Client with personalized model adaptation"""
    
    def __init__(self, client_id: str, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray, config: PersonalizedConfig):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        
        # Global and local models
        self.global_weights = None
        self.local_weights = None
        self.personalized_weights = None
        
        # Personalization tracking
        self.personalization_history: List[float] = []
        self.local_loss_history: List[float] = []
        self.global_loss_history: List[float] = []
        self.similarity_scores: List[float] = []
        
        # Cluster assignment
        self.cluster_id = -1
    
    def set_global_weights(self, weights: np.ndarray):
        """Update global weights from server"""
        self.global_weights = weights.copy()
        
        # Initialize local weights if needed
        if self.local_weights is None:
            self.local_weights = weights.copy()
        
        # Initialize personalized weights as weighted mixture
        self._update_personalized_weights()
    
    def _update_personalized_weights(self):
        """Compute personalized weights as mixture of global and local"""
        alpha = self.config.personalization_alpha
        
        if self.global_weights is not None and self.local_weights is not None:
            self.personalized_weights = (1 - alpha) * self.global_weights + alpha * self.local_weights
        elif self.global_weights is not None:
            self.personalized_weights = self.global_weights.copy()
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Compute MSE loss"""
        n_features = X.shape[1]
        y_pred = X @ weights[:n_features]
        if weights.shape[0] > n_features:
            y_pred += weights[-1]
        return np.mean((y_pred - y)**2)
    
    def local_personalized_training(self) -> PersonalizationMetrics:
        """Train personalized model on local data"""
        
        if self.personalized_weights is None:
            raise ValueError("Weights not initialized")
        
        weights = self.personalized_weights.copy()
        n_samples = self.X_train.shape[0]
        n_features = self.X_train.shape[1]
        
        for epoch in range(self.config.local_epochs):
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                X_batch = self.X_train[batch_idx]
                y_batch = self.y_train[batch_idx]
                
                # Forward pass
                y_pred = X_batch @ weights[:n_features]
                if weights.shape[0] > n_features:
                    y_pred += weights[-1]
                
                # Backward pass
                residual = y_pred - y_batch
                grad_w = (2.0 / len(X_batch)) * X_batch.T @ residual
                grad_b = (2.0 / len(X_batch)) * np.sum(residual)
                
                # Update with local learning rate
                weights[:n_features] -= self.config.local_lr * grad_w
                weights[-1] -= self.config.local_lr * grad_b
        
        # Update local weights
        self.local_weights = weights.copy()
        self._update_personalized_weights()
        
        # Compute metrics
        global_loss = self.compute_loss(self.X_test, self.y_test, self.global_weights)
        local_loss = self.compute_loss(self.X_test, self.y_test, self.local_weights)
        
        # Personalization level: how much local loss improved
        adaptation_rate = max(0, (global_loss - local_loss) / (global_loss + 1e-6))
        personalization_level = min(1.0, adaptation_rate)
        
        # Track history
        self.global_loss_history.append(global_loss)
        self.local_loss_history.append(local_loss)
        self.personalization_history.append(personalization_level)
        
        return PersonalizationMetrics(
            client_id=self.client_id,
            personalization_level=personalization_level,
            global_loss=global_loss,
            local_loss=local_loss,
            adaptation_rate=adaptation_rate,
            cluster_id=self.cluster_id
        )


class ClientClusterer:
    """Cluster similar clients for better personalization"""
    
    @staticmethod
    def compute_data_similarity(X1: np.ndarray, X2: np.ndarray) -> float:
        """Compute similarity between two datasets using correlation"""
        
        # Compute feature statistics
        stats1 = np.concatenate([np.mean(X1, axis=0), np.std(X1, axis=0)])
        stats2 = np.concatenate([np.mean(X2, axis=0), np.std(X2, axis=0)])
        
        # Normalize
        stats1 = (stats1 - np.mean(stats1)) / (np.std(stats1) + 1e-6)
        stats2 = (stats2 - np.mean(stats2)) / (np.std(stats2) + 1e-6)
        
        # Correlation
        similarity = np.abs(np.corrcoef(stats1, stats2)[0, 1])
        return np.clip(similarity, 0, 1)
    
    @staticmethod
    def kmeans_clustering(clients: List[PersonalizedClient], num_clusters: int) -> Dict[int, List[str]]:
        """K-means clustering of clients based on data characteristics"""
        
        # Extract client features (mean, std, skew of each feature)
        client_features = []
        client_ids = []
        
        for client in clients:
            # Compute statistics for each feature
            features = []
            for feat_idx in range(client.X_train.shape[1]):
                col = client.X_train[:, feat_idx]
                features.extend([np.mean(col), np.std(col), stats.skew(col)])
            
            client_features.append(np.array(features))
            client_ids.append(client.client_id)
        
        client_features = np.array(client_features)
        
        # Normalize
        feature_mean = np.mean(client_features, axis=0)
        feature_std = np.std(client_features, axis=0) + 1e-6
        client_features = (client_features - feature_mean) / feature_std
        
        # Initialize centroids
        centroid_indices = np.random.choice(len(clients), size=num_clusters, replace=False)
        centroids = client_features[centroid_indices].copy()
        
        # K-means iterations
        for iteration in range(10):
            # Assign to nearest centroid
            distances = np.zeros((len(clients), num_clusters))
            for i in range(len(clients)):
                for j in range(num_clusters):
                    distances[i, j] = np.linalg.norm(client_features[i] - centroids[j])
            
            assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(num_clusters):
                cluster_points = client_features[assignments == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[j] = centroids[j]
            
            centroids = new_centroids
        
        # Create cluster assignments
        clusters = {i: [] for i in range(num_clusters)}
        for client_id, cluster_id in zip(client_ids, assignments):
            clusters[cluster_id].append(client_id)
        
        return clusters


class PersonalizedFLServer:
    """Server coordinating personalized federated learning"""
    
    def __init__(self, config: PersonalizedConfig):
        self.config = config
        self.global_weights = np.random.normal(0, 0.1, 51)
        self.client_clusters: Dict[int, List[str]] = {}
        self.personalization_metrics: List[PersonalizationMetrics] = []
        self.round_results: List[Dict] = []
        
    def aggregate_personalized_updates(self, client_updates: Dict[str, np.ndarray]) -> np.ndarray:
        """Aggregate updates with personalization awareness"""
        
        if not client_updates:
            return self.global_weights.copy()
        
        # Weighted average (could use cluster-specific aggregation)
        updates = list(client_updates.values())
        aggregated = np.mean(updates, axis=0)
        
        return aggregated


class PersonalizedFLCoordinator:
    """Coordinates personalized federated learning"""
    
    def __init__(self, config: PersonalizedConfig, clients: List[PersonalizedClient]):
        self.config = config
        self.clients = clients
        self.server = PersonalizedFLServer(config)
        self.clusterer = ClientClusterer()
        
        # Assign cluster IDs
        self._cluster_clients()
    
    def _cluster_clients(self):
        """Cluster clients for better personalization"""
        
        logger.info(f"Clustering {len(self.clients)} clients into {self.config.num_clusters} clusters...")
        
        clusters = self.clusterer.kmeans_clustering(self.clients, self.config.num_clusters)
        
        for cluster_id, client_ids in clusters.items():
            for client in self.clients:
                if client.client_id in client_ids:
                    client.cluster_id = cluster_id
        
        self.server.client_clusters = clusters
        
        for cluster_id, client_ids in clusters.items():
            logger.info(f"Cluster {cluster_id}: {client_ids}")
    
    def run_personalized_fl(self) -> Dict:
        """Execute personalized federated learning"""
        
        logger.info(f"Starting personalized FL with {len(self.clients)} clients")
        
        for round_num in range(self.config.num_rounds):
            logger.info(f"\n--- Round {round_num + 1}/{self.config.num_rounds} ---")
            
            # Send global weights to all clients
            for client in self.clients:
                client.set_global_weights(self.server.global_weights)
            
            # Client personalized training
            client_updates = {}
            round_metrics = []
            
            for client in self.clients:
                metrics = client.local_personalized_training()
                round_metrics.append(metrics)
                client_updates[client.client_id] = client.personalized_weights
            
            # Server aggregation
            new_global_weights = self.server.aggregate_personalized_updates(client_updates)
            self.server.global_weights = new_global_weights
            
            # Log metrics
            avg_personalization = np.mean([m.personalization_level for m in round_metrics])
            avg_adaptation = np.mean([m.adaptation_rate for m in round_metrics])
            
            logger.info(f"Round {round_num + 1}: Avg personalization={avg_personalization:.3f}, "
                       f"Avg adaptation={avg_adaptation:.3f}")
            
            # Store round results
            round_result = {
                'round': round_num + 1,
                'avg_personalization': float(avg_personalization),
                'avg_adaptation': float(avg_adaptation),
                'client_metrics': [m.to_dict() for m in round_metrics]
            }
            self.server.round_results.append(round_result)
        
        return self.get_final_results()
    
    def get_final_results(self) -> Dict:
        """Compile final results"""
        
        personalization_levels = []
        adaptation_rates = []
        
        for client in self.clients:
            if client.personalization_history:
                personalization_levels.append(client.personalization_history[-1])
            if hasattr(client, 'adaptation_rate') or client.local_loss_history:
                if len(client.local_loss_history) > 0 and len(client.global_loss_history) > 0:
                    final_adaptation = (client.global_loss_history[-1] - client.local_loss_history[-1]) / (client.global_loss_history[-1] + 1e-6)
                    adaptation_rates.append(final_adaptation)
        
        return {
            'algorithm': 'Personalized-FedAvg',
            'num_rounds': self.config.num_rounds,
            'num_clients': len(self.clients),
            'num_clusters': self.config.num_clusters,
            'personalization_alpha': self.config.personalization_alpha,
            'avg_personalization_level': float(np.mean(personalization_levels)) if personalization_levels else 0.0,
            'avg_adaptation_rate': float(np.mean(adaptation_rates)) if adaptation_rates else 0.0,
            'round_results': self.server.round_results,
            'clusters': self.server.client_clusters
        }


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 9.3: Personalized Federated Learning")
    logger.info("=" * 60)
    
    # Generate diverse client data
    num_clients = 10
    n_samples = 1000
    n_features = 50
    
    # Create clients with different data distributions
    clients = []
    
    for i in range(num_clients):
        # Each client has slightly different feature distribution
        scale = 0.8 + (i * 0.02)
        shift = (i * 0.1) - 0.45
        
        X_train = np.random.normal(shift, scale, (n_samples, n_features))
        true_weights = np.random.normal(0, 1, n_features)
        y_train = X_train @ true_weights + 0.1 * np.random.normal(0, 1, n_samples)
        
        X_test = np.random.normal(shift, scale, (n_samples // 2, n_features))
        y_test = X_test @ true_weights + 0.1 * np.random.normal(0, 1, n_samples // 2)
        
        config = PersonalizedConfig(
            num_clients=num_clients,
            num_rounds=25,
            local_epochs=3,
            personalization_alpha=0.5,
            num_clusters=3
        )
        
        client = PersonalizedClient(
            client_id=f"client_{i}",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=config
        )
        clients.append(client)
    
    # Run personalized FL
    coordinator = PersonalizedFLCoordinator(config, clients)
    results = coordinator.run_personalized_fl()
    
    # Results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Algorithm: {results['algorithm']}")
    logger.info(f"Num clients: {results['num_clients']}")
    logger.info(f"Num clusters: {results['num_clusters']}")
    logger.info(f"Avg personalization level: {results['avg_personalization_level']:.4f}")
    logger.info(f"Avg adaptation rate: {results['avg_adaptation_rate']:.4f}")
    
    # Save results
    with open('phase9_personalized_fl_results.json', 'w') as f:
        json.dump(results, f, indent=2)
