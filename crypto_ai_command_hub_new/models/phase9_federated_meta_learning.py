"""
Phase 9.5: Federated Meta-Learning
Combines federated learning with meta-learning to adapt quickly to new tasks:
- MAML-style few-shot adaptation
- Federated task distribution learning
- Client-specific meta-parameters
- Fast adaptation with minimal data
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetaTask:
    """Represents a task for meta-learning"""
    task_id: str
    X_support: np.ndarray  # Support set for few-shot learning
    y_support: np.ndarray
    X_query: np.ndarray    # Query set for evaluation
    y_query: np.ndarray
    client_id: str = ""
    

@dataclass
class AdaptationMetrics:
    """Metrics for meta-adaptation"""
    task_id: str
    inner_loss_initial: float
    inner_loss_final: float
    adaptation_efficiency: float  # How much loss reduced in inner loop
    num_inner_steps: int
    inner_learning_rate: float
    
    def to_dict(self):
        return {
            'task_id': self.task_id,
            'inner_loss_initial': float(self.inner_loss_initial),
            'inner_loss_final': float(self.inner_loss_final),
            'adaptation_efficiency': float(self.adaptation_efficiency),
            'num_inner_steps': int(self.num_inner_steps),
            'inner_learning_rate': float(self.inner_learning_rate)
        }


@dataclass
class FedMetaConfig:
    """Configuration for federated meta-learning"""
    num_clients: int = 10
    num_rounds: int = 50
    num_tasks_per_client: int = 5  # Tasks available to each client
    num_meta_epochs: int = 10
    inner_learning_rate: float = 0.1  # For task-specific adaptation
    outer_learning_rate: float = 0.01  # For meta-model update
    inner_steps: int = 3  # Gradient steps in inner loop
    num_support_samples: int = 5  # Few-shot: support set size
    num_query_samples: int = 20   # Query set size
    

class FedMetaClient:
    """Client performing federated meta-learning"""
    
    def __init__(self, client_id: str, tasks: List[MetaTask], config: FedMetaConfig):
        self.client_id = client_id
        self.tasks = tasks
        self.config = config
        
        self.meta_weights = None  # Global meta-model
        self.task_weights = {}    # Task-specific adaptations
        self.adaptation_history: List[AdaptationMetrics] = []
        
    def set_meta_weights(self, weights: np.ndarray):
        """Update meta-model weights from server"""
        self.meta_weights = weights.copy()
        
    def inner_loop_adaptation(self, task: MetaTask) -> Tuple[np.ndarray, AdaptationMetrics]:
        """
        MAML-style inner loop: adapt meta-weights to specific task
        
        Args:
            task: Task to adapt to
            
        Returns:
            Task-adapted weights and metrics
        """
        if self.meta_weights is None:
            raise ValueError("Meta weights not initialized")
        
        # Start with meta-weights
        task_weights = self.meta_weights.copy()
        n_features = task.X_support.shape[1]
        
        # Compute initial loss
        y_pred = task.X_support @ task_weights[:n_features]
        if task_weights.shape[0] > n_features:
            y_pred += task_weights[-1]
        initial_loss = np.mean((y_pred - task.y_support)**2)
        
        # Inner loop: adapt to this specific task
        for step in range(self.config.inner_steps):
            y_pred = task.X_support @ task_weights[:n_features]
            if task_weights.shape[0] > n_features:
                y_pred += task_weights[-1]
            
            # Compute gradient
            residual = y_pred - task.y_support
            grad = (2.0 / len(task.X_support)) * task.X_support.T @ residual
            grad_bias = (2.0 / len(task.X_support)) * np.sum(residual)
            
            # Gradient step
            task_weights[:n_features] -= self.config.inner_learning_rate * grad
            task_weights[-1] -= self.config.inner_learning_rate * grad_bias
        
        # Compute final loss on query set
        y_pred_query = task.X_query @ task_weights[:n_features]
        if task_weights.shape[0] > n_features:
            y_pred_query += task_weights[-1]
        
        final_loss = np.mean((y_pred_query - task.y_query)**2)
        
        # Adaptation efficiency
        adaptation_efficiency = max(0, (initial_loss - final_loss) / (initial_loss + 1e-6))
        
        metrics = AdaptationMetrics(
            task_id=task.task_id,
            inner_loss_initial=initial_loss,
            inner_loss_final=final_loss,
            adaptation_efficiency=adaptation_efficiency,
            num_inner_steps=self.config.inner_steps,
            inner_learning_rate=self.config.inner_learning_rate
        )
        
        return task_weights, metrics
    
    def meta_training_step(self) -> Tuple[np.ndarray, List[AdaptationMetrics]]:
        """
        Outer loop: update meta-weights based on task adaptation
        
        Returns:
            Gradient for server aggregation and metrics
        """
        # Compute gradients of adapted tasks
        meta_gradient = np.zeros_like(self.meta_weights)
        round_metrics = []
        
        for task in self.tasks:
            task_weights, metrics = self.inner_loop_adaptation(task)
            round_metrics.append(metrics)
            
            # Compute gradient of task loss w.r.t. meta-weights
            # This is second-order derivative (gradient of gradient)
            task_gradient = task_weights - self.meta_weights
            meta_gradient += task_gradient
        
        # Average across tasks
        meta_gradient /= len(self.tasks)
        
        # Store metrics
        self.adaptation_history.extend(round_metrics)
        
        return meta_gradient, round_metrics


class FedMetaServer:
    """Server for federated meta-learning"""
    
    def __init__(self, config: FedMetaConfig):
        self.config = config
        self.meta_weights = np.random.normal(0, 0.1, 51)
        self.round_results: List[Dict] = []
        self.task_performance: List[float] = []
        
    def aggregate_meta_gradients(self, client_gradients: List[np.ndarray]) -> None:
        """Update meta-weights with aggregated gradients"""
        
        if not client_gradients:
            return
        
        # Average gradients
        avg_gradient = np.mean(client_gradients, axis=0)
        
        # Update meta-weights
        self.meta_weights -= self.config.outer_learning_rate * avg_gradient


class FedMetaCoordinator:
    """Coordinates federated meta-learning"""
    
    def __init__(self, config: FedMetaConfig, clients: List[FedMetaClient]):
        self.config = config
        self.clients = clients
        self.server = FedMetaServer(config)
        
    def run_federated_meta_learning(self) -> Dict:
        """Execute federated meta-learning"""
        
        logger.info(f"Starting federated meta-learning with {len(self.clients)} clients")
        logger.info(f"Meta-learning configuration: inner_lr={self.config.inner_learning_rate}, "
                   f"outer_lr={self.config.outer_learning_rate}, inner_steps={self.config.inner_steps}")
        
        for round_num in range(self.config.num_rounds):
            logger.info(f"\n--- Meta-Round {round_num + 1}/{self.config.num_rounds} ---")
            
            # Distribute meta-weights to clients
            for client in self.clients:
                client.set_meta_weights(self.server.meta_weights)
            
            # Client meta-training
            client_gradients = []
            round_metrics = []
            
            for client in self.clients:
                gradient, metrics = client.meta_training_step()
                client_gradients.append(gradient)
                round_metrics.extend(metrics)
            
            # Server aggregation
            self.server.aggregate_meta_gradients(client_gradients)
            
            # Compute round statistics
            avg_adaptation = np.mean([m.adaptation_efficiency for m in round_metrics])
            avg_final_loss = np.mean([m.inner_loss_final for m in round_metrics])
            
            logger.info(f"Round {round_num + 1}: Avg adaptation={avg_adaptation:.4f}, "
                       f"Avg final loss={avg_final_loss:.6f}")
            
            round_result = {
                'round': round_num + 1,
                'avg_adaptation_efficiency': float(avg_adaptation),
                'avg_final_loss': float(avg_final_loss),
                'num_tasks_evaluated': len(round_metrics)
            }
            self.server.round_results.append(round_result)
        
        return self.get_final_results()
    
    def get_final_results(self) -> Dict:
        """Compile final results"""
        
        all_metrics = []
        for client in self.clients:
            all_metrics.extend(client.adaptation_history)
        
        if all_metrics:
            adaptation_efficiencies = [m.adaptation_efficiency for m in all_metrics]
            final_losses = [m.inner_loss_final for m in all_metrics]
        else:
            adaptation_efficiencies = []
            final_losses = []
        
        return {
            'algorithm': 'Federated-MAML',
            'num_rounds': self.config.num_rounds,
            'num_clients': len(self.clients),
            'inner_learning_rate': self.config.inner_learning_rate,
            'outer_learning_rate': self.config.outer_learning_rate,
            'avg_adaptation_efficiency': float(np.mean(adaptation_efficiencies)) if adaptation_efficiencies else 0.0,
            'avg_final_loss': float(np.mean(final_losses)) if final_losses else 0.0,
            'total_tasks_evaluated': len(all_metrics),
            'round_results': self.server.round_results
        }


def create_synthetic_tasks(num_tasks: int, support_size: int, query_size: int,
                          n_features: int = 50) -> List[MetaTask]:
    """Create synthetic meta-learning tasks"""
    
    tasks = []
    
    for task_id in range(num_tasks):
        # Each task has slightly different data distribution
        task_offset = np.random.normal(0, 0.5, n_features)
        
        # Support set (few-shot)
        X_support = np.random.normal(task_offset, 1, (support_size, n_features))
        true_weights = np.random.normal(0, 1, n_features)
        y_support = X_support @ true_weights + 0.05 * np.random.normal(0, 1, support_size)
        
        # Query set
        X_query = np.random.normal(task_offset, 1, (query_size, n_features))
        y_query = X_query @ true_weights + 0.05 * np.random.normal(0, 1, query_size)
        
        task = MetaTask(
            task_id=f"task_{task_id}",
            X_support=X_support,
            y_support=y_support,
            X_query=X_query,
            y_query=y_query
        )
        tasks.append(task)
    
    return tasks


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 9.5: Federated Meta-Learning")
    logger.info("=" * 60)
    
    # Configuration
    config = FedMetaConfig(
        num_clients=8,
        num_rounds=30,
        num_tasks_per_client=5,
        inner_learning_rate=0.1,
        outer_learning_rate=0.01,
        inner_steps=3,
        num_support_samples=5,
        num_query_samples=20
    )
    
    # Create clients with tasks
    clients = []
    for client_id in range(config.num_clients):
        # Create tasks for this client
        tasks = create_synthetic_tasks(
            num_tasks=config.num_tasks_per_client,
            support_size=config.num_support_samples,
            query_size=config.num_query_samples
        )
        
        client = FedMetaClient(f"client_{client_id}", tasks, config)
        clients.append(client)
    
    # Run federated meta-learning
    coordinator = FedMetaCoordinator(config, clients)
    results = coordinator.run_federated_meta_learning()
    
    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Algorithm: {results['algorithm']}")
    logger.info(f"Number of rounds: {results['num_rounds']}")
    logger.info(f"Number of clients: {results['num_clients']}")
    logger.info(f"Avg adaptation efficiency: {results['avg_adaptation_efficiency']:.4f}")
    logger.info(f"Avg final loss: {results['avg_final_loss']:.6f}")
    logger.info(f"Total tasks evaluated: {results['total_tasks_evaluated']}")
    
    # Save results
    with open('phase9_federated_meta_learning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
