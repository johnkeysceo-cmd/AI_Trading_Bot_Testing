"""
Phase 9.6: Byzantine-Robust Federated Learning
Implements aggregation methods resilient to malicious or faulty clients:
- Coordinate-wise median aggregation
- Krum aggregation (multi-Krum)
- Distance-based outlier removal
- Byzantine tolerance analysis
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
import logging
from datetime import datetime
from scipy.spatial.distance import euclidean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ByzantineRobustnessMetrics:
    """Metrics for Byzantine robustness"""
    round_num: int
    num_clients_total: int
    num_byzantine: int
    aggregation_method: str
    robustness_score: float  # [0,1] - resistance to Byzantine attacks
    detected_malicious: int
    weight_divergence: float
    convergence_metric: float
    
    def to_dict(self):
        return {
            'round': self.round_num,
            'total_clients': self.num_clients_total,
            'byzantine_clients': self.num_byzantine,
            'aggregation_method': self.aggregation_method,
            'robustness_score': float(self.robustness_score),
            'detected_malicious': self.detected_malicious,
            'weight_divergence': float(self.weight_divergence),
            'convergence_metric': float(self.convergence_metric)
        }


@dataclass
class ByzantineConfig:
    """Configuration for Byzantine-robust FL"""
    num_clients: int = 10
    num_byzantine_clients: int = 2  # Number of malicious clients
    num_rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    aggregation_method: str = "multi_krum"  # or "median", "distance_based"
    krum_f: int = 2  # Number of Byzantine clients to tolerate
    median_threshold: float = 0.5
    

class ByzantineAggregator:
    """Implements Byzantine-robust aggregation methods"""
    
    @staticmethod
    def coordinate_wise_median(updates: List[np.ndarray]) -> np.ndarray:
        """
        Compute element-wise median of updates
        
        Args:
            updates: List of weight vectors from clients
            
        Returns:
            Median weights
        """
        updates_array = np.array(updates)
        median_weights = np.median(updates_array, axis=0)
        return median_weights
    
    @staticmethod
    def krum_aggregation(updates: List[np.ndarray], f: int) -> Tuple[np.ndarray, List[int]]:
        """
        Krum aggregation: select update with smallest distance to others
        
        Args:
            updates: List of weight vectors
            f: Number of Byzantine clients to tolerate
            
        Returns:
            Selected weights and index, list of distances
        """
        n = len(updates)
        
        # For each update, compute sum of distances to n-f-2 nearest neighbors
        distances = []
        
        for i, update_i in enumerate(updates):
            # Compute distances to all other updates
            dist_to_others = []
            for j, update_j in enumerate(updates):
                if i != j:
                    dist = np.linalg.norm(update_i - update_j)
                    dist_to_others.append(dist)
            
            dist_to_others.sort()
            # Sum of distances to n-f-2 nearest neighbors
            sum_dist = np.sum(dist_to_others[:n - f - 2])
            distances.append(sum_dist)
        
        # Select update with minimum distance sum
        selected_idx = np.argmin(distances)
        selected_weights = updates[selected_idx]
        
        return selected_weights, distances
    
    @staticmethod
    def multi_krum_aggregation(updates: List[np.ndarray], f: int, m: int = None) -> Tuple[np.ndarray, List[int]]:
        """
        Multi-Krum: average m updates selected by Krum
        
        Args:
            updates: List of weight vectors
            f: Number of Byzantine clients to tolerate
            m: Number of updates to average (default: n-f)
            
        Returns:
            Aggregated weights and selected indices
        """
        n = len(updates)
        if m is None:
            m = n - f
        
        selected_indices = []
        remaining_updates = updates.copy()
        
        # Select m updates using Krum
        for _ in range(m):
            krum_weights, _ = ByzantineAggregator.krum_aggregation(remaining_updates, f)
            selected_idx = remaining_updates.index(krum_weights)
            selected_indices.append(selected_idx)
            remaining_updates.remove(krum_weights)
        
        # Average selected updates
        aggregated = np.mean([updates[i] for i in selected_indices], axis=0)
        
        return aggregated, selected_indices
    
    @staticmethod
    def distance_based_aggregation(updates: List[np.ndarray], threshold: float = 0.5) -> Tuple[np.ndarray, List[int]]:
        """
        Remove outlier updates by distance and average rest
        
        Args:
            updates: List of weight vectors
            threshold: Distance threshold for outlier detection
            
        Returns:
            Aggregated weights and list of kept indices
        """
        # Compute pairwise distances
        n = len(updates)
        updates_array = np.array(updates)
        mean_weights = np.mean(updates_array, axis=0)
        
        # Compute distance of each update from mean
        distances = []
        for update in updates:
            dist = np.linalg.norm(update - mean_weights)
            distances.append(dist)
        
        distances = np.array(distances)
        max_dist = np.max(distances)
        
        # Normalize distances
        if max_dist > 0:
            norm_distances = distances / max_dist
        else:
            norm_distances = distances
        
        # Keep updates with distance below threshold
        kept_indices = np.where(norm_distances < threshold)[0].tolist()
        
        # If all removed, keep all
        if not kept_indices:
            kept_indices = list(range(n))
        
        # Average kept updates
        kept_updates = [updates[i] for i in kept_indices]
        aggregated = np.mean(kept_updates, axis=0)
        
        return aggregated, kept_indices


class ByzantineClient:
    """Client that may be Byzantine (malicious)"""
    
    def __init__(self, client_id: str, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray, is_byzantine: bool = False,
                 config: ByzantineConfig = None):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.is_byzantine = is_byzantine
        self.config = config or ByzantineConfig()
        
        self.current_weights = None
        self.benign_loss_history: List[float] = []
        
    def set_weights(self, weights: np.ndarray):
        """Receive weights from server"""
        self.current_weights = weights.copy()
    
    def local_training(self) -> np.ndarray:
        """Train locally"""
        
        if self.current_weights is None:
            raise ValueError("Weights not initialized")
        
        weights = self.current_weights.copy()
        n_samples = self.X_train.shape[0]
        n_features = self.X_train.shape[1]
        
        for epoch in range(self.config.local_epochs):
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                X_batch = self.X_train[batch_idx]
                y_batch = self.y_train[batch_idx]
                
                # Forward and backward
                y_pred = X_batch @ weights[:n_features]
                if weights.shape[0] > n_features:
                    y_pred += weights[-1]
                
                residual = y_pred - y_batch
                grad = (2.0 / len(X_batch)) * X_batch.T @ residual
                grad_bias = (2.0 / len(X_batch)) * np.sum(residual)
                
                # Update
                weights[:n_features] -= self.config.learning_rate * grad
                weights[-1] -= self.config.learning_rate * grad_bias
        
        # Track loss
        y_test_pred = self.X_test @ weights[:n_features]
        if weights.shape[0] > n_features:
            y_test_pred += weights[-1]
        
        test_loss = np.mean((y_test_pred - self.y_test)**2)
        self.benign_loss_history.append(test_loss)
        
        return weights
    
    def attack_poisoning(self) -> np.ndarray:
        """Byzantine attack: send malicious update"""
        
        if not self.is_byzantine:
            return self.local_training()
        
        # Send random/malicious weights
        malicious_weights = np.random.normal(0, 10, self.current_weights.shape)
        
        logger.info(f"Client {self.client_id} sending Byzantine attack")
        
        return malicious_weights


class ByzantineRobustServer:
    """Server with Byzantine-robust aggregation"""
    
    def __init__(self, config: ByzantineConfig):
        self.config = config
        self.global_weights = np.random.normal(0, 0.1, 51)
        self.aggregator = ByzantineAggregator()
        self.round_results: List[ByzantineRobustnessMetrics] = []
        
    def aggregate_updates(self, client_updates: List[np.ndarray], round_num: int) -> Tuple[np.ndarray, ByzantineRobustnessMetrics]:
        """Aggregate with Byzantine robustness"""
        
        if not client_updates:
            return self.global_weights.copy(), None
        
        # Apply aggregation method
        if self.config.aggregation_method == "median":
            aggregated = self.aggregator.coordinate_wise_median(client_updates)
            detected = 0
        elif self.config.aggregation_method == "multi_krum":
            aggregated, selected_idx = self.aggregator.multi_krum_aggregation(
                client_updates, 
                f=self.config.num_byzantine_clients
            )
            detected = len(client_updates) - len(selected_idx)
        elif self.config.aggregation_method == "distance_based":
            aggregated, kept_idx = self.aggregator.distance_based_aggregation(
                client_updates,
                threshold=self.config.median_threshold
            )
            detected = len(client_updates) - len(kept_idx)
        else:
            aggregated = np.mean(client_updates, axis=0)
            detected = 0
        
        # Compute metrics
        weight_divergence = np.mean([np.linalg.norm(u - aggregated) for u in client_updates])
        robustness_score = 1.0 - (detected / len(client_updates))
        
        convergence_metric = np.linalg.norm(aggregated - self.global_weights)
        
        metrics = ByzantineRobustnessMetrics(
            round_num=round_num,
            num_clients_total=len(client_updates),
            num_byzantine=self.config.num_byzantine_clients,
            aggregation_method=self.config.aggregation_method,
            robustness_score=robustness_score,
            detected_malicious=detected,
            weight_divergence=weight_divergence,
            convergence_metric=convergence_metric
        )
        
        self.global_weights = aggregated
        return aggregated, metrics


class ByzantineRobustCoordinator:
    """Coordinates Byzantine-robust federated learning"""
    
    def __init__(self, config: ByzantineConfig, clients: List[ByzantineClient]):
        self.config = config
        self.clients = clients
        self.server = ByzantineRobustServer(config)
        
    def run_byzantine_robust_fl(self) -> Dict:
        """Execute Byzantine-robust FL"""
        
        logger.info(f"Starting Byzantine-robust FL")
        logger.info(f"Total clients: {len(self.clients)}, Byzantine clients: {self.config.num_byzantine_clients}")
        logger.info(f"Aggregation method: {self.config.aggregation_method}")
        
        for round_num in range(self.config.num_rounds):
            logger.info(f"\n--- Round {round_num + 1}/{self.config.num_rounds} ---")
            
            # Send weights
            for client in self.clients:
                client.set_weights(self.server.global_weights)
            
            # Client training/attack
            client_updates = []
            for client in self.clients:
                if client.is_byzantine:
                    update = client.attack_poisoning()
                else:
                    update = client.local_training()
                client_updates.append(update)
            
            # Server aggregation
            _, metrics = self.server.aggregate_updates(client_updates, round_num + 1)
            
            if metrics:
                self.server.round_results.append(metrics)
                
                logger.info(f"Round {round_num + 1}: Robustness={metrics.robustness_score:.3f}, "
                           f"Detected={metrics.detected_malicious}, "
                           f"Divergence={metrics.weight_divergence:.4f}")
        
        return self.get_final_results()
    
    def get_final_results(self) -> Dict:
        """Compile final results"""
        
        robustness_scores = [m.robustness_score for m in self.server.round_results]
        detected_counts = [m.detected_malicious for m in self.server.round_results]
        
        return {
            'algorithm': f'Byzantine-Robust-{self.config.aggregation_method}',
            'num_clients': len(self.clients),
            'num_byzantine': self.config.num_byzantine_clients,
            'aggregation_method': self.config.aggregation_method,
            'avg_robustness_score': float(np.mean(robustness_scores)) if robustness_scores else 0.0,
            'total_detected_attacks': int(np.sum(detected_counts)),
            'num_rounds': len(self.server.round_results),
            'round_results': [m.to_dict() for m in self.server.round_results]
        }


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 9.6: Byzantine-Robust Federated Learning")
    logger.info("=" * 60)
    
    # Test different aggregation methods
    aggregation_methods = ['median', 'multi_krum', 'distance_based']
    
    for method in aggregation_methods:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing aggregation method: {method.upper()}")
        logger.info(f"{'=' * 60}")
        
        config = ByzantineConfig(
            num_clients=10,
            num_byzantine_clients=2,
            num_rounds=20,
            aggregation_method=method,
            krum_f=2
        )
        
        # Generate data
        n_samples = 1000
        n_features = 50
        
        clients = []
        for i in range(config.num_clients):
            X_train = np.random.normal(0, 1, (n_samples, n_features))
            true_weights = np.random.normal(0, 1, n_features)
            y_train = X_train @ true_weights + 0.1 * np.random.normal(0, 1, n_samples)
            
            X_test = np.random.normal(0, 1, (n_samples // 2, n_features))
            y_test = X_test @ true_weights + 0.1 * np.random.normal(0, 1, n_samples // 2)
            
            is_byzantine = i < config.num_byzantine_clients
            
            client = ByzantineClient(
                client_id=f"client_{i}",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                is_byzantine=is_byzantine,
                config=config
            )
            clients.append(client)
        
        # Run Byzantine-robust FL
        coordinator = ByzantineRobustCoordinator(config, clients)
        results = coordinator.run_byzantine_robust_fl()
        
        # Log results
        logger.info(f"\n{'=' * 60}")
        logger.info(f"RESULTS - {method.upper()}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Aggregation method: {results['aggregation_method']}")
        logger.info(f"Avg robustness score: {results['avg_robustness_score']:.4f}")
        logger.info(f"Total detected attacks: {results['total_detected_attacks']}")
        
        # Save results
        with open(f'phase9_byzantine_robust_{method}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
