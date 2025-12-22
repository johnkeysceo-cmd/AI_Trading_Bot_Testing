"""
Phase 9.1: Federated Averaging (FedAvg) Protocol
Implements the canonical Federated Learning algorithm where multiple clients
aggregate model updates on a central server without sharing raw data.

Core Components:
- Client-side SGD training
- Server-side weight averaging
- Communication efficiency tracking
- Privacy guarantees through parameter aggregation
- Convergence analysis for non-IID data
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import json
import logging
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    """Represents a model update from a single client"""
    client_id: str
    weights: np.ndarray  # Flattened model weights
    samples_used: int
    loss: float
    accuracy: float
    timestamp: float = field(default_factory=datetime.now().timestamp)
    
    def to_dict(self):
        return {
            'client_id': self.client_id,
            'samples_used': self.samples_used,
            'loss': float(self.loss),
            'accuracy': float(self.accuracy),
            'timestamp': self.timestamp,
            'weight_hash': hashlib.md5(self.weights.tobytes()).hexdigest()
        }


@dataclass
class FedAvgConfig:
    """Configuration for FedAvg protocol"""
    num_clients: int = 10
    num_rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    fraction_fit: float = 1.0  # Fraction of clients to sample each round
    min_available_clients: int = 2
    initial_weights: Optional[np.ndarray] = None
    momentum: float = 0.9
    weight_decay: float = 0.0001
    

@dataclass
class RoundResults:
    """Results from a single federated learning round"""
    round_num: int
    num_clients_sampled: int
    num_clients_aggregated: int
    server_loss: float
    server_accuracy: float
    client_losses: List[float] = field(default_factory=list)
    client_accuracies: List[float] = field(default_factory=list)
    communication_cost: float = 0.0  # Bytes transmitted
    convergence_metric: float = 0.0  # Weight change magnitude
    
    def to_dict(self):
        return {
            'round': self.round_num,
            'sampled': self.num_clients_sampled,
            'aggregated': self.num_clients_aggregated,
            'server_loss': float(self.server_loss),
            'server_accuracy': float(self.server_accuracy),
            'avg_client_loss': float(np.mean(self.client_losses)) if self.client_losses else 0.0,
            'avg_client_accuracy': float(np.mean(self.client_accuracies)) if self.client_accuracies else 0.0,
            'communication_bytes': float(self.communication_cost),
            'convergence_metric': float(self.convergence_metric)
        }


class FedAvgClient:
    """Individual client in federated learning system"""
    
    def __init__(self, client_id: str, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray, model_architecture: str = "linear"):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_architecture = model_architecture
        self.current_weights = None
        self.weight_history = []
        self.local_losses = []
        
    def set_weights(self, weights: np.ndarray):
        """Update model weights from server"""
        self.current_weights = weights.copy()
        
    def local_sgd_update(self, num_epochs: int, batch_size: int, learning_rate: float,
                        momentum: float = 0.9, weight_decay: float = 0.0001) -> ClientUpdate:
        """
        Perform local SGD training on client data
        
        Args:
            num_epochs: Number of local epochs
            batch_size: Batch size for SGD
            learning_rate: SGD learning rate
            momentum: Momentum coefficient
            weight_decay: L2 regularization
            
        Returns:
            ClientUpdate with trained weights and metrics
        """
        if self.current_weights is None:
            raise ValueError("Weights not initialized")
        
        weights = self.current_weights.copy()
        velocity = np.zeros_like(weights)  # For momentum
        n_samples = self.X_train.shape[0]
        n_features = self.X_train.shape[1]
        
        for epoch in range(num_epochs):
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]
            
            # Mini-batch SGD
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Simple linear model: y = X @ w
                y_pred = X_batch @ weights[:n_features]
                if weights.shape[0] > n_features:  # Has bias
                    y_pred += weights[-1]
                
                # MSE loss and gradient
                residual = y_pred - y_batch
                loss = np.mean(residual**2)
                
                # Gradient computation
                grad = (2.0 / len(X_batch)) * X_batch.T @ residual
                grad_bias = (2.0 / len(X_batch)) * np.sum(residual)
                
                # L2 regularization gradient
                grad_w = grad + weight_decay * weights[:n_features]
                
                # Momentum update
                velocity[:n_features] = momentum * velocity[:n_features] - learning_rate * grad_w
                weights[:n_features] += velocity[:n_features]
                
                if weights.shape[0] > n_features:
                    weights[-1] -= learning_rate * grad_bias
            
            # Evaluate on local test set
            y_test_pred = self.X_test @ weights[:n_features]
            if weights.shape[0] > n_features:
                y_test_pred += weights[-1]
            
            test_loss = np.mean((y_test_pred - self.y_test)**2)
            self.local_losses.append(test_loss)
        
        # Final evaluation
        y_final_pred = self.X_test @ weights[:n_features]
        if weights.shape[0] > n_features:
            y_final_pred += weights[-1]
        
        final_loss = np.mean((y_final_pred - self.y_test)**2)
        # Accuracy as 1/(1 + MSE)
        final_accuracy = 1.0 / (1.0 + final_loss)
        
        update = ClientUpdate(
            client_id=self.client_id,
            weights=weights,
            samples_used=n_samples,
            loss=final_loss,
            accuracy=final_accuracy
        )
        
        self.weight_history.append(weights.copy())
        return update


class FedAvgServer:
    """Central server orchestrating federated learning"""
    
    def __init__(self, config: FedAvgConfig):
        self.config = config
        self.global_weights = config.initial_weights
        if self.global_weights is None:
            # Initialize with random weights (feature_dim + 1 for bias)
            self.global_weights = np.random.normal(0, 0.1, 51)  # 50 features + bias
        
        self.global_loss_history = []
        self.global_accuracy_history = []
        self.communication_history = []
        self.round_results: List[RoundResults] = []
        self.client_updates_history: List[List[ClientUpdate]] = []
        
    def aggregate_updates(self, client_updates: List[ClientUpdate]) -> np.ndarray:
        """
        Aggregate client updates using weighted averaging
        
        Args:
            client_updates: List of ClientUpdate from selected clients
            
        Returns:
            Aggregated weights
        """
        if not client_updates:
            return self.global_weights.copy()
        
        # Weighted average by number of samples
        total_samples = sum(update.samples_used for update in client_updates)
        
        aggregated_weights = np.zeros_like(self.global_weights)
        for update in client_updates:
            weight = update.samples_used / total_samples
            aggregated_weights += weight * update.weights
        
        return aggregated_weights
    
    def sample_clients(self, num_clients: int) -> List[int]:
        """Sample clients for current round"""
        num_to_sample = max(int(num_clients * self.config.fraction_fit), 
                           self.config.min_available_clients)
        return list(np.random.choice(num_clients, size=num_to_sample, replace=False))
    
    def aggregate_and_update(self, sampled_client_updates: List[ClientUpdate]) -> Dict:
        """Aggregate and update global model"""
        
        num_aggregated = len(sampled_client_updates)
        
        # Weighted FedAvg aggregation
        new_global_weights = self.aggregate_updates(sampled_client_updates)
        
        # Compute convergence metric (weight change)
        weight_change = np.linalg.norm(new_global_weights - self.global_weights)
        
        # Compute communication cost (bytes for weights)
        communication_cost = new_global_weights.nbytes * len(sampled_client_updates)
        
        # Update global model
        self.global_weights = new_global_weights
        
        # Compute aggregate metrics
        avg_loss = np.mean([u.loss for u in sampled_client_updates]) if sampled_client_updates else 0.0
        avg_accuracy = np.mean([u.accuracy for u in sampled_client_updates]) if sampled_client_updates else 0.0
        
        self.global_loss_history.append(avg_loss)
        self.global_accuracy_history.append(avg_accuracy)
        self.communication_history.append(communication_cost)
        
        return {
            'global_weights': new_global_weights,
            'weight_change': weight_change,
            'communication_cost': communication_cost,
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'num_aggregated': num_aggregated
        }


class FedAvgCoordinator:
    """Coordinates federated learning process"""
    
    def __init__(self, config: FedAvgConfig, clients: List[FedAvgClient]):
        self.config = config
        self.clients = clients
        self.server = FedAvgServer(config)
        self.rounds_completed = 0
        self.performance_history = []
        
    def run_federated_learning(self) -> Dict:
        """Execute federated learning for specified rounds"""
        
        logger.info(f"Starting FedAvg with {len(self.clients)} clients, {self.config.num_rounds} rounds")
        
        for round_num in range(self.config.num_rounds):
            logger.info(f"\n--- Round {round_num + 1}/{self.config.num_rounds} ---")
            
            # Step 1: Sample clients
            sampled_indices = self.server.sample_clients(len(self.clients))
            sampled_clients = [self.clients[i] for i in sampled_indices]
            
            logger.info(f"Sampled {len(sampled_clients)} clients")
            
            # Step 2: Send global weights to clients
            for client in sampled_clients:
                client.set_weights(self.server.global_weights)
            
            # Step 3: Client local training
            client_updates = []
            for client in sampled_clients:
                update = client.local_sgd_update(
                    num_epochs=self.config.local_epochs,
                    batch_size=self.config.batch_size,
                    learning_rate=self.config.learning_rate,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay
                )
                client_updates.append(update)
            
            # Step 4: Server aggregation
            agg_result = self.server.aggregate_and_update(client_updates)
            
            # Step 5: Collect metrics
            round_result = RoundResults(
                round_num=round_num + 1,
                num_clients_sampled=len(sampled_clients),
                num_clients_aggregated=len(client_updates),
                server_loss=agg_result['avg_loss'],
                server_accuracy=agg_result['avg_accuracy'],
                client_losses=[u.loss for u in client_updates],
                client_accuracies=[u.accuracy for u in client_updates],
                communication_cost=agg_result['communication_cost'],
                convergence_metric=agg_result['weight_change']
            )
            
            self.server.round_results.append(round_result)
            self.performance_history.append(round_result.to_dict())
            
            logger.info(f"Round {round_num + 1}: Loss={round_result.server_loss:.4f}, "
                       f"Accuracy={round_result.server_accuracy:.4f}, "
                       f"Convergence={round_result.convergence_metric:.6f}")
            
            # Early stopping if converged
            if round_num > 10 and round_result.convergence_metric < 1e-5:
                logger.info("Converged early, stopping training")
                break
            
            self.rounds_completed = round_num + 1
        
        return self.get_final_results()
    
    def get_final_results(self) -> Dict:
        """Return final federated learning results"""
        
        return {
            'rounds_completed': self.rounds_completed,
            'final_global_weights': self.server.global_weights,
            'final_loss': self.server.global_loss_history[-1] if self.server.global_loss_history else None,
            'final_accuracy': self.server.global_accuracy_history[-1] if self.server.global_accuracy_history else None,
            'loss_history': self.server.global_loss_history,
            'accuracy_history': self.server.global_accuracy_history,
            'total_communication_bytes': sum(self.server.communication_history),
            'performance_history': self.performance_history,
            'clients_count': len(self.clients),
            'algorithm': 'FedAvg'
        }
    
    def get_convergence_analysis(self) -> Dict:
        """Analyze convergence properties"""
        
        if not self.server.round_results:
            return {'status': 'no_results'}
        
        losses = self.server.global_loss_history
        convergence_metrics = [r.convergence_metric for r in self.server.round_results]
        
        # Convergence rate (loss reduction per round)
        if len(losses) > 1:
            convergence_rate = (losses[0] - losses[-1]) / losses[0]
        else:
            convergence_rate = 0.0
        
        return {
            'convergence_rate': convergence_rate,
            'final_loss_reduction': losses[0] - losses[-1],
            'avg_weight_change': np.mean(convergence_metrics),
            'max_weight_change': np.max(convergence_metrics),
            'min_weight_change': np.min(convergence_metrics),
            'total_rounds': len(losses),
            'steady_state': losses[-5:] if len(losses) >= 5 else losses
        }


def create_non_iid_data(num_clients: int, n_samples: int, n_features: int, 
                       iid_fraction: float = 0.1) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create non-IID distributed data across clients
    
    Args:
        num_clients: Number of federated clients
        n_samples: Total samples across all clients
        n_features: Feature dimension
        iid_fraction: Fraction of data that's IID (0=pure non-IID, 1=pure IID)
    """
    
    # Generate base dataset
    X_all = np.random.normal(0, 1, (n_samples * 2, n_features))  # Extra for test
    true_weights = np.random.normal(0, 1, n_features)
    y_all = X_all @ true_weights + 0.1 * np.random.normal(0, 1, n_samples * 2)
    
    # Split across clients with non-IID property
    samples_per_client = n_samples // num_clients
    client_data = []
    
    for client_id in range(num_clients):
        # Add some IID samples
        iid_size = int(samples_per_client * iid_fraction)
        iid_indices = np.random.choice(n_samples * 2, size=iid_size, replace=True)
        
        # Add client-specific (non-IID) samples
        start_idx = client_id * samples_per_client
        end_idx = start_idx + (samples_per_client - iid_size)
        non_iid_indices = np.arange(start_idx, min(end_idx, n_samples * 2))
        
        # Combine indices
        all_indices = np.concatenate([iid_indices, non_iid_indices])
        
        # Split into train/test
        n_train = int(0.8 * len(all_indices))
        train_indices = all_indices[:n_train]
        test_indices = all_indices[n_train:]
        
        X_train, y_train = X_all[train_indices], y_all[train_indices]
        X_test, y_test = X_all[test_indices], y_all[test_indices]
        
        client_data.append((X_train, y_train, X_test, y_test))
    
    return client_data


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 9.1: Federated Averaging (FedAvg) Protocol")
    logger.info("=" * 60)
    
    # Configuration
    num_clients = 10
    n_samples = 10000
    n_features = 50
    
    # Create non-IID distributed data
    logger.info(f"\nGenerating non-IID data for {num_clients} clients...")
    client_data = create_non_iid_data(
        num_clients=num_clients,
        n_samples=n_samples,
        n_features=n_features,
        iid_fraction=0.15  # 15% IID, 85% non-IID
    )
    
    # Create clients
    clients = []
    for i, (X_train, y_train, X_test, y_test) in enumerate(client_data):
        client = FedAvgClient(
            client_id=f"client_{i}",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        clients.append(client)
    
    # Configure FedAvg
    config = FedAvgConfig(
        num_clients=num_clients,
        num_rounds=30,
        local_epochs=3,
        batch_size=16,
        learning_rate=0.05,
        fraction_fit=0.8,  # Sample 80% of clients each round
        momentum=0.9,
        weight_decay=0.0001
    )
    
    # Run federated learning
    logger.info(f"\nStarting federated learning: {config.num_rounds} rounds, "
               f"fraction_fit={config.fraction_fit}")
    
    coordinator = FedAvgCoordinator(config, clients)
    results = coordinator.run_federated_learning()
    
    # Convergence analysis
    logger.info("\n" + "=" * 60)
    logger.info("CONVERGENCE ANALYSIS")
    logger.info("=" * 60)
    
    convergence = coordinator.get_convergence_analysis()
    for key, value in convergence.items():
        if key != 'steady_state':
            logger.info(f"{key}: {value}")
    
    # Final metrics
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Rounds Completed: {results['rounds_completed']}")
    logger.info(f"Final Loss: {results['final_loss']:.6f}")
    logger.info(f"Final Accuracy: {results['final_accuracy']:.6f}")
    logger.info(f"Total Communication: {results['total_communication_bytes'] / 1e6:.2f} MB")
    logger.info(f"Algorithm: {results['algorithm']}")
    
    # Save results
    results_path = 'phase9_fedavg_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'rounds_completed': results['rounds_completed'],
            'final_loss': float(results['final_loss']),
            'final_accuracy': float(results['final_accuracy']),
            'loss_history': [float(x) for x in results['loss_history']],
            'accuracy_history': [float(x) for x in results['accuracy_history']],
            'total_communication_bytes': float(results['total_communication_bytes']),
            'performance_history': results['performance_history'],
            'convergence_analysis': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                    for k, v in convergence.items()}
        }
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
