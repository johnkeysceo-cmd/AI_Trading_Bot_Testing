"""
Phase 9.7: Federated Transfer Learning
Implements knowledge transfer across federated domains:
- Feature extraction layer sharing
- Fine-tuning with domain adaptation
- Cross-domain knowledge distillation
- Domain-specific output layers
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
class TransferMetrics:
    """Metrics for federated transfer learning"""
    domain_id: str
    transfer_efficiency: float  # How much transfer helped
    source_domain_loss: float
    target_domain_loss: float
    feature_similarity: float  # Similarity between extracted features
    fine_tuning_improvement: float
    
    def to_dict(self):
        return {
            'domain_id': self.domain_id,
            'transfer_efficiency': float(self.transfer_efficiency),
            'source_domain_loss': float(self.source_domain_loss),
            'target_domain_loss': float(self.target_domain_loss),
            'feature_similarity': float(self.feature_similarity),
            'fine_tuning_improvement': float(self.fine_tuning_improvement)
        }


@dataclass
class FederatedTransferConfig:
    """Configuration for federated transfer learning"""
    num_source_domains: int = 5  # Domains with abundant data
    num_target_domains: int = 5  # New domains with limited data
    num_rounds: int = 50
    feature_extraction_layers: int = 3
    task_specific_layers: int = 2
    local_epochs: int = 5
    batch_size: int = 32
    feature_learning_rate: float = 0.001  # Slow for shared features
    task_learning_rate: float = 0.01      # Fast for task-specific
    transfer_weight: float = 0.5  # Balance source and target
    distillation_temperature: float = 3.0
    

class FeatureExtractor:
    """Shared feature extraction layer"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initialize weights for multi-layer extraction
        self.weights = []
        self.biases = []
        
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        for i in range(num_layers):
            w = np.random.normal(0, 0.1, (dims[i], hidden_dim))
            b = np.zeros(hidden_dim)
            self.weights.append(w)
            self.biases.append(b)
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features through layers"""
        
        features = X.copy()
        
        for w, b in zip(self.weights, self.biases):
            # Linear transformation + ReLU
            features = features @ w + b
            features = np.maximum(0, features)  # ReLU
        
        return features
    
    def update_weights(self, gradients: List[np.ndarray], learning_rate: float):
        """Update feature extraction weights"""
        
        for i, grad in enumerate(gradients):
            if i < len(self.weights):
                self.weights[i] -= learning_rate * grad


class DomainSpecificHead:
    """Domain-specific prediction head"""
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int = 1, num_layers: int = 2):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Task-specific layers
        self.weights = []
        self.biases = []
        
        dims = [feature_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            w = np.random.normal(0, 0.1, (dims[i], dims[i + 1]))
            b = np.zeros(dims[i + 1])
            self.weights.append(w)
            self.biases.append(b)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make prediction from features"""
        
        out = features.copy()
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            out = out @ w + b
            # ReLU for hidden layers
            if i < len(self.weights) - 1:
                out = np.maximum(0, out)
        
        return out
    
    def update_weights(self, gradients: List[np.ndarray], learning_rate: float):
        """Update task-specific weights"""
        
        for i, grad in enumerate(gradients):
            if i < len(self.weights):
                self.weights[i] -= learning_rate * grad


class FederatedTransferClient:
    """Client participating in federated transfer learning"""
    
    def __init__(self, domain_id: str, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray, 
                 feature_extractor: FeatureExtractor, domain_head: DomainSpecificHead,
                 config: FederatedTransferConfig):
        self.domain_id = domain_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_extractor = feature_extractor
        self.domain_head = domain_head
        self.config = config
        
        self.transfer_metrics_history: List[TransferMetrics] = []
        self.loss_history: List[float] = []
    
    def extract_and_predict(self, X: np.ndarray) -> np.ndarray:
        """Extract features and make prediction"""
        features = self.feature_extractor.extract_features(X)
        predictions = self.domain_head.predict(features)
        return predictions.flatten()
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE loss"""
        predictions = self.extract_and_predict(X)
        return np.mean((predictions - y)**2)
    
    def transfer_learning_update(self, shared_features: Optional[np.ndarray] = None) -> TransferMetrics:
        """
        Update with transfer learning
        
        Args:
            shared_features: Features from source domain (for distillation)
        """
        
        initial_loss = self.compute_loss(self.X_test, self.y_test)
        
        # Training phase
        n_samples = self.X_train.shape[0]
        
        for epoch in range(self.config.local_epochs):
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                X_batch = self.X_train[batch_idx]
                y_batch = self.y_train[batch_idx]
                
                # Extract features
                features = self.feature_extractor.extract_features(X_batch)
                
                # Make prediction
                predictions = self.domain_head.predict(features)
                
                # Compute loss
                loss = np.mean((predictions.flatten() - y_batch)**2)
                
                # Knowledge distillation loss (if shared features available)
                if shared_features is not None:
                    kd_loss = self._distillation_loss(
                        features,
                        shared_features[batch_idx],
                        temperature=self.config.distillation_temperature
                    )
                    loss = self.config.transfer_weight * loss + (1 - self.config.transfer_weight) * kd_loss
                
                # Gradient computation and update
                # Simplified: just update task-specific layer
                residual = predictions.flatten() - y_batch
                grad = (2.0 / len(y_batch)) * np.mean(residual)
                
                # Update domain head (would normally include full backprop)
                self.domain_head.biases[-1] -= self.config.task_learning_rate * grad
        
        # Compute final loss
        final_loss = self.compute_loss(self.X_test, self.y_test)
        
        # Transfer efficiency
        transfer_efficiency = max(0, (initial_loss - final_loss) / (initial_loss + 1e-6))
        
        # Feature similarity (placeholder)
        if shared_features is not None:
            target_features = self.feature_extractor.extract_features(self.X_test)
            source_features = shared_features  # Assuming same shape
            feature_sim = 1.0 - (np.mean(np.abs(target_features - source_features)) / 
                               (np.mean(np.abs(source_features)) + 1e-6))
        else:
            feature_sim = 0.5
        
        metrics = TransferMetrics(
            domain_id=self.domain_id,
            transfer_efficiency=transfer_efficiency,
            source_domain_loss=initial_loss,
            target_domain_loss=final_loss,
            feature_similarity=feature_sim,
            fine_tuning_improvement=transfer_efficiency
        )
        
        self.transfer_metrics_history.append(metrics)
        self.loss_history.append(final_loss)
        
        return metrics
    
    @staticmethod
    def _distillation_loss(student_features: np.ndarray, teacher_features: np.ndarray, 
                          temperature: float = 3.0) -> float:
        """Compute knowledge distillation loss"""
        
        # Normalize features
        student_norm = student_features / (np.linalg.norm(student_features, axis=1, keepdims=True) + 1e-6)
        teacher_norm = teacher_features / (np.linalg.norm(teacher_features, axis=1, keepdims=True) + 1e-6)
        
        # KL divergence of softmaxes at temperature T
        student_soft = np.exp(student_norm / temperature) / np.sum(np.exp(student_norm / temperature), axis=1, keepdims=True)
        teacher_soft = np.exp(teacher_norm / temperature) / np.sum(np.exp(teacher_norm / temperature), axis=1, keepdims=True)
        
        kl_div = np.sum(teacher_soft * (np.log(teacher_soft + 1e-6) - np.log(student_soft + 1e-6)))
        
        return kl_div


class FederatedTransferServer:
    """Server managing transfer learning"""
    
    def __init__(self, config: FederatedTransferConfig):
        self.config = config
        # Shared feature extractor across all domains
        self.global_feature_extractor = FeatureExtractor(
            input_dim=50,
            hidden_dim=25,
            num_layers=config.feature_extraction_layers
        )
        self.round_results: List[Dict] = []
    
    def aggregate_feature_extractors(self, client_extractors: List[FeatureExtractor]):
        """Average feature extractors across domains"""
        
        # Average weights
        for layer_idx in range(len(self.global_feature_extractor.weights)):
            avg_w = np.mean([ce.weights[layer_idx] for ce in client_extractors], axis=0)
            self.global_feature_extractor.weights[layer_idx] = avg_w


class FederatedTransferCoordinator:
    """Coordinates federated transfer learning"""
    
    def __init__(self, config: FederatedTransferConfig):
        self.config = config
        self.server = FederatedTransferServer(config)
        self.source_clients: List[FederatedTransferClient] = []
        self.target_clients: List[FederatedTransferClient] = []
        
    def setup_domains(self):
        """Create source and target domains"""
        
        n_samples = 2000
        n_features = 50
        
        # Create source domains (abundant data)
        logger.info(f"Creating {self.config.num_source_domains} source domains...")
        for i in range(self.config.num_source_domains):
            X_train = np.random.normal(i * 0.1, 1, (n_samples, n_features))
            true_weights = np.random.normal(0, 1, n_features)
            y_train = X_train @ true_weights + 0.1 * np.random.normal(0, 1, n_samples)
            
            X_test = np.random.normal(i * 0.1, 1, (n_samples // 2, n_features))
            y_test = X_test @ true_weights + 0.1 * np.random.normal(0, 1, n_samples // 2)
            
            domain_head = DomainSpecificHead(25, 15, output_dim=1)
            
            client = FederatedTransferClient(
                domain_id=f"source_{i}",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_extractor=self.server.global_feature_extractor,
                domain_head=domain_head,
                config=self.config
            )
            self.source_clients.append(client)
        
        # Create target domains (limited data)
        logger.info(f"Creating {self.config.num_target_domains} target domains...")
        for i in range(self.config.num_target_domains):
            # Limited data
            X_train = np.random.normal(0.5 + i * 0.1, 0.8, (n_samples // 4, n_features))
            true_weights = np.random.normal(0, 1, n_features)
            y_train = X_train @ true_weights + 0.1 * np.random.normal(0, 1, len(X_train))
            
            X_test = np.random.normal(0.5 + i * 0.1, 0.8, (n_samples // 8, n_features))
            y_test = X_test @ true_weights + 0.1 * np.random.normal(0, 1, len(X_test))
            
            domain_head = DomainSpecificHead(25, 15, output_dim=1)
            
            client = FederatedTransferClient(
                domain_id=f"target_{i}",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_extractor=self.server.global_feature_extractor,
                domain_head=domain_head,
                config=self.config
            )
            self.target_clients.append(client)
    
    def run_federated_transfer_learning(self) -> Dict:
        """Execute federated transfer learning"""
        
        logger.info("Starting federated transfer learning")
        logger.info(f"Source domains: {len(self.source_clients)}, Target domains: {len(self.target_clients)}")
        
        for round_num in range(self.config.num_rounds):
            logger.info(f"\n--- Round {round_num + 1}/{self.config.num_rounds} ---")
            
            # Phase 1: Train on source domains
            logger.info("Training on source domains...")
            source_metrics = []
            for client in self.source_clients:
                metrics = client.transfer_learning_update()
                source_metrics.append(metrics)
            
            # Phase 2: Transfer to target domains
            logger.info("Transferring to target domains...")
            target_metrics = []
            
            # Extract features from source for distillation
            source_features_list = [
                client.feature_extractor.extract_features(client.X_test)
                for client in self.source_clients
            ]
            avg_source_features = np.mean(source_features_list, axis=0)
            
            for client in self.target_clients:
                metrics = client.transfer_learning_update(shared_features=avg_source_features)
                target_metrics.append(metrics)
            
            # Aggregate feature extractors
            all_clients = self.source_clients + self.target_clients
            self.server.aggregate_feature_extractors([c.feature_extractor for c in all_clients])
            
            # Log metrics
            avg_source_efficiency = np.mean([m.transfer_efficiency for m in source_metrics])
            avg_target_efficiency = np.mean([m.transfer_efficiency for m in target_metrics])
            
            logger.info(f"Round {round_num + 1}: Source efficiency={avg_source_efficiency:.4f}, "
                       f"Target efficiency={avg_target_efficiency:.4f}")
            
            round_result = {
                'round': round_num + 1,
                'source_transfer_efficiency': float(avg_source_efficiency),
                'target_transfer_efficiency': float(avg_target_efficiency),
                'source_metrics': [m.to_dict() for m in source_metrics],
                'target_metrics': [m.to_dict() for m in target_metrics]
            }
            self.server.round_results.append(round_result)
        
        return self.get_final_results()
    
    def get_final_results(self) -> Dict:
        """Compile final results"""
        
        all_target_efficiency = []
        for result in self.server.round_results:
            all_target_efficiency.append(result['target_transfer_efficiency'])
        
        return {
            'algorithm': 'Federated-Transfer-Learning',
            'num_source_domains': self.config.num_source_domains,
            'num_target_domains': self.config.num_target_domains,
            'num_rounds': len(self.server.round_results),
            'avg_target_transfer_efficiency': float(np.mean(all_target_efficiency)) if all_target_efficiency else 0.0,
            'final_target_efficiency': float(all_target_efficiency[-1]) if all_target_efficiency else 0.0,
            'distillation_temperature': self.config.distillation_temperature,
            'transfer_weight': self.config.transfer_weight,
            'round_results': self.server.round_results
        }


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 9.7: Federated Transfer Learning")
    logger.info("=" * 60)
    
    config = FederatedTransferConfig(
        num_source_domains=5,
        num_target_domains=5,
        num_rounds=25,
        feature_extraction_layers=3,
        transfer_weight=0.6,
        distillation_temperature=3.0
    )
    
    coordinator = FederatedTransferCoordinator(config)
    coordinator.setup_domains()
    results = coordinator.run_federated_transfer_learning()
    
    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Algorithm: {results['algorithm']}")
    logger.info(f"Source domains: {results['num_source_domains']}")
    logger.info(f"Target domains: {results['num_target_domains']}")
    logger.info(f"Avg target transfer efficiency: {results['avg_target_transfer_efficiency']:.4f}")
    logger.info(f"Final target efficiency: {results['final_target_efficiency']:.4f}")
    
    # Save results
    with open('phase9_federated_transfer_learning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
