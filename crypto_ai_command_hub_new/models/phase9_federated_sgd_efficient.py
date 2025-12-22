"""
Phase 9.2: Federated SGD with Communication Efficiency
Implements communication-efficient federated learning through:
- Gradient compression (quantization, sparsification)
- Local SGD with multiple epochs
- Adaptive learning rates
- Communication-computation tradeoff analysis
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import json
import logging
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CompressionStats:
    """Statistics for gradient compression"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    reconstruction_error: float
    sparsity: float  # Fraction of zeros
    
    def to_dict(self):
        return {
            'original_bytes': self.original_size,
            'compressed_bytes': self.compressed_size,
            'compression_ratio': float(self.compression_ratio),
            'reconstruction_error': float(self.reconstruction_error),
            'sparsity': float(self.sparsity)
        }


@dataclass
class CommunicationEfficientConfig:
    """Configuration for communication-efficient federated learning"""
    num_clients: int = 10
    num_rounds: int = 50
    local_epochs: int = 10  # More local epochs = less communication
    batch_size: int = 32
    learning_rate: float = 0.01
    gradient_compression: str = "quantization"  # or "sparsification", "none"
    quantization_bits: int = 8  # For quantization
    compression_ratio: float = 0.1  # For sparsification (keep top 10%)
    momentum: float = 0.9
    adaptive_lr: bool = True
    

class GradientCompressor:
    """Compress gradients for efficient communication"""
    
    @staticmethod
    def quantize_gradients(gradients: np.ndarray, num_bits: int = 8) -> Tuple[np.ndarray, Dict]:
        """
        Quantize gradients to fixed-point representation
        
        Args:
            gradients: Original gradient vector
            num_bits: Number of bits for quantization
            
        Returns:
            Quantized gradients and compression info
        """
        # Determine quantization range
        min_val = np.min(gradients)
        max_val = np.max(gradients)
        range_val = max_val - min_val
        
        if range_val == 0:
            quantized = np.zeros_like(gradients, dtype=np.uint8)
            error = 0.0
        else:
            # Scale to [0, 2^num_bits - 1]
            max_int = (1 << num_bits) - 1
            scaled = ((gradients - min_val) / range_val) * max_int
            quantized = np.round(scaled).astype(np.uint8)
            
            # Dequantize to compute error
            dequantized = (quantized.astype(np.float32) / max_int) * range_val + min_val
            error = np.mean(np.abs(dequantized - gradients))
        
        info = {
            'min_val': float(min_val),
            'max_val': float(max_val),
            'num_bits': num_bits,
            'reconstruction_error': float(error)
        }
        
        return quantized, info
    
    @staticmethod
    def dequantize_gradients(quantized: np.ndarray, info: Dict) -> np.ndarray:
        """Dequantize gradients"""
        max_int = (1 << info['num_bits']) - 1
        range_val = info['max_val'] - info['min_val']
        
        if range_val == 0:
            return np.full_like(quantized, info['min_val'], dtype=np.float32)
        
        dequantized = (quantized.astype(np.float32) / max_int) * range_val + info['min_val']
        return dequantized
    
    @staticmethod
    def sparsify_gradients(gradients: np.ndarray, compression_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Keep only top-k gradients by magnitude
        
        Args:
            gradients: Original gradient vector
            compression_ratio: Fraction to keep (e.g., 0.1 = keep top 10%)
            
        Returns:
            Sparse gradient mask and non-zero values
        """
        # Find top-k indices
        k = max(1, int(len(gradients) * compression_ratio))
        top_indices = np.argsort(np.abs(gradients))[-k:]
        
        # Create sparse representation
        sparse_mask = np.zeros_like(gradients, dtype=bool)
        sparse_mask[top_indices] = True
        sparse_values = gradients[sparse_mask]
        
        return sparse_mask, sparse_values
    
    @staticmethod
    def reconstruct_sparsified(sparse_mask: np.ndarray, sparse_values: np.ndarray) -> np.ndarray:
        """Reconstruct sparse gradients"""
        reconstructed = np.zeros(len(sparse_mask), dtype=np.float32)
        reconstructed[sparse_mask] = sparse_values
        return reconstructed


class CommunicationEfficientClient:
    """Client with communication-efficient training"""
    
    def __init__(self, client_id: str, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray, config: CommunicationEfficientConfig):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
        
        self.current_weights = None
        self.gradient_momentum = None
        self.accumulated_local_updates = 0
        self.communication_rounds = 0
        self.compression_stats_history: List[CompressionStats] = []
        
        # Initialize compressor
        self.compressor = GradientCompressor()
    
    def set_weights(self, weights: np.ndarray):
        """Update weights from server"""
        self.current_weights = weights.copy()
        if self.gradient_momentum is None:
            self.gradient_momentum = np.zeros_like(weights)
    
    def compute_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient for linear model"""
        n_features = X.shape[1]
        y_pred = X @ self.current_weights[:n_features]
        if self.current_weights.shape[0] > n_features:
            y_pred += self.current_weights[-1]
        
        residual = y_pred - y
        grad = (2.0 / len(X)) * X.T @ residual
        grad_bias = (2.0 / len(X)) * np.sum(residual)
        
        full_grad = np.concatenate([grad, [grad_bias]])
        return full_grad
    
    def local_training_with_compression(self) -> Tuple[np.ndarray, Dict]:
        """
        Local training with gradient compression
        
        Returns:
            Compressed update and compression info
        """
        weights = self.current_weights.copy()
        n_samples = self.X_train.shape[0]
        
        # Accumulate gradients over multiple epochs
        accumulated_grad = np.zeros_like(weights)
        
        for epoch in range(self.config.local_epochs):
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                X_batch = self.X_train[batch_indices]
                y_batch = self.y_train[batch_indices]
                
                # Compute gradient
                grad = self.compute_gradient(X_batch, y_batch)
                accumulated_grad += grad
                
                # Apply momentum
                if self.gradient_momentum is not None:
                    self.gradient_momentum = self.config.momentum * self.gradient_momentum - self.config.learning_rate * grad
                    weights += self.gradient_momentum
                else:
                    weights -= self.config.learning_rate * grad
        
        # Average accumulated gradient
        accumulated_grad /= (self.config.local_epochs * (n_samples // self.config.batch_size + 1))
        
        # Compress gradient
        compression_info = {'method': self.config.gradient_compression}
        
        if self.config.gradient_compression == 'quantization':
            compressed_grad, quant_info = self.compressor.quantize_gradients(
                accumulated_grad, 
                num_bits=self.config.quantization_bits
            )
            compression_info.update(quant_info)
            
            # Calculate stats
            original_size = accumulated_grad.nbytes
            compressed_size = compressed_grad.nbytes
            compression_ratio = original_size / max(compressed_size, 1)
            
            stats = CompressionStats(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                reconstruction_error=quant_info['reconstruction_error'],
                sparsity=0.0
            )
            
        elif self.config.gradient_compression == 'sparsification':
            sparse_mask, sparse_values = self.compressor.sparsify_gradients(
                accumulated_grad,
                compression_ratio=self.config.compression_ratio
            )
            compressed_grad = (sparse_mask, sparse_values)
            
            original_size = accumulated_grad.nbytes
            compressed_size = sparse_values.nbytes
            sparsity = 1.0 - (np.sum(sparse_mask) / len(accumulated_grad))
            
            stats = CompressionStats(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=original_size / max(compressed_size, 1),
                reconstruction_error=0.0,
                sparsity=sparsity
            )
        else:
            compressed_grad = accumulated_grad
            stats = CompressionStats(
                original_size=accumulated_grad.nbytes,
                compressed_size=accumulated_grad.nbytes,
                compression_ratio=1.0,
                reconstruction_error=0.0,
                sparsity=0.0
            )
        
        self.compression_stats_history.append(stats)
        self.communication_rounds += 1
        
        # Update weights for next round
        self.current_weights = weights
        
        return compressed_grad, compression_info


class CommunicationEfficientServer:
    """Server handling communication-efficient aggregation"""
    
    def __init__(self, config: CommunicationEfficientConfig):
        self.config = config
        self.global_weights = np.random.normal(0, 0.1, 51)
        self.round_results: List[Dict] = []
        self.total_communication_rounds = 0
        self.total_data_transmitted = 0
        
    def aggregate_compressed_updates(self, updates: List[Tuple[np.ndarray, Dict]]) -> np.ndarray:
        """Aggregate potentially compressed updates"""
        
        all_updates = []
        
        for update, info in updates:
            if info['method'] == 'quantization':
                # Dequantize
                dequantized = self.config.gradient_compressor.dequantize_gradients(update, info)
                all_updates.append(dequantized)
            elif info['method'] == 'sparsification':
                # Reconstruct
                reconstructed = self.config.gradient_compressor.reconstruct_sparsified(
                    update[0], update[1]
                )
                all_updates.append(reconstructed)
            else:
                all_updates.append(update)
        
        # Average updates
        aggregated = np.mean(all_updates, axis=0)
        self.global_weights -= self.config.learning_rate * aggregated
        
        return aggregated


class CommunicationEfficientCoordinator:
    """Coordinates communication-efficient federated learning"""
    
    def __init__(self, config: CommunicationEfficientConfig, clients: List[CommunicationEfficientClient]):
        self.config = config
        self.clients = clients
        self.server = CommunicationEfficientServer(config)
        self.config.gradient_compressor = GradientCompressor()  # Attach to config
        self.performance_history: List[Dict] = []
        
    def run_training(self) -> Dict:
        """Execute communication-efficient federated learning"""
        
        logger.info(f"Starting communication-efficient FL with {len(self.clients)} clients")
        logger.info(f"Compression method: {self.config.gradient_compression}")
        
        all_compression_stats = defaultdict(list)
        
        for round_num in range(self.config.num_rounds):
            logger.info(f"\n--- Round {round_num + 1}/{self.config.num_rounds} ---")
            
            # Send weights to all clients
            for client in self.clients:
                client.set_weights(self.server.global_weights)
            
            # Client local training with compression
            updates = []
            for client in self.clients:
                compressed_update, info = client.local_training_with_compression()
                updates.append((compressed_update, info))
            
            # Server aggregation
            aggregated_gradient = self.server.aggregate_compressed_updates(updates)
            
            # Collect statistics
            round_stats = {
                'round': round_num + 1,
                'num_clients': len(self.clients),
                'gradient_norm': float(np.linalg.norm(aggregated_gradient)),
                'compression_method': self.config.gradient_compression
            }
            
            # Collect compression stats
            for client in self.clients:
                if client.compression_stats_history:
                    latest_stat = client.compression_stats_history[-1]
                    all_compression_stats['compression_ratios'].append(latest_stat.compression_ratio)
                    all_compression_stats['errors'].append(latest_stat.reconstruction_error)
                    all_compression_stats['sparsities'].append(latest_stat.sparsity)
            
            self.performance_history.append(round_stats)
            
            logger.info(f"Round {round_num + 1}: Gradient norm={round_stats['gradient_norm']:.6f}, "
                       f"Compression method={self.config.gradient_compression}")
        
        return self.get_final_results(all_compression_stats)
    
    def get_final_results(self, compression_stats: Dict) -> Dict:
        """Compile final results"""
        
        total_comm_bytes = sum(sum(c.compression_stats_history[i].compressed_size 
                                   for i in range(min(5, len(c.compression_stats_history))))
                              for c in self.clients)
        
        return {
            'algorithm': 'FedSGD-CommEfficient',
            'num_rounds': self.config.num_rounds,
            'num_clients': len(self.clients),
            'compression_method': self.config.gradient_compression,
            'compression_stats': {
                'avg_ratio': float(np.mean(compression_stats['compression_ratios'])) if compression_stats['compression_ratios'] else 1.0,
                'avg_error': float(np.mean(compression_stats['errors'])) if compression_stats['errors'] else 0.0,
                'avg_sparsity': float(np.mean(compression_stats['sparsities'])) if compression_stats['sparsities'] else 0.0
            },
            'performance_history': self.performance_history,
            'total_data_transmitted_bytes': float(total_comm_bytes)
        }


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 9.2: Federated SGD with Communication Efficiency")
    logger.info("=" * 60)
    
    # Generate data
    num_clients = 10
    n_samples = 5000
    n_features = 50
    
    X_all = np.random.normal(0, 1, (n_samples * 2, n_features))
    true_weights = np.random.normal(0, 1, n_features)
    y_all = X_all @ true_weights + 0.1 * np.random.normal(0, 1, n_samples * 2)
    
    # Create clients with different compression methods
    compression_methods = ['quantization', 'sparsification', 'none']
    
    for comp_method in compression_methods:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing compression method: {comp_method}")
        logger.info(f"{'=' * 60}")
        
        # Split data
        clients = []
        samples_per_client = n_samples // num_clients
        
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client
            X_train = X_all[start:end]
            y_train = y_all[start:end]
            
            test_start = n_samples + i * (n_samples // num_clients)
            test_end = test_start + (n_samples // num_clients)
            X_test = X_all[test_start:test_end]
            y_test = y_all[test_start:test_end]
            
            # Create config
            config = CommunicationEfficientConfig(
                num_clients=num_clients,
                num_rounds=20,
                local_epochs=5,
                gradient_compression=comp_method,
                quantization_bits=8,
                compression_ratio=0.1
            )
            
            client = CommunicationEfficientClient(
                client_id=f"client_{i}",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                config=config
            )
            clients.append(client)
        
        # Run training
        coordinator = CommunicationEfficientCoordinator(config, clients)
        results = coordinator.run_training()
        
        # Log results
        logger.info(f"\n{'=' * 60}")
        logger.info(f"RESULTS - {comp_method.upper()}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Compression method: {results['compression_method']}")
        logger.info(f"Avg compression ratio: {results['compression_stats']['avg_ratio']:.2f}x")
        logger.info(f"Avg reconstruction error: {results['compression_stats']['avg_error']:.6f}")
        logger.info(f"Avg sparsity: {results['compression_stats']['avg_sparsity']:.2%}")
        logger.info(f"Total data transmitted: {results['total_data_transmitted_bytes'] / 1e6:.2f} MB")
        
        # Save results
        with open(f'phase9_fedsgd_{comp_method}_results.json', 'w') as f:
            json_results = {
                'method': comp_method,
                'compression_stats': results['compression_stats'],
                'total_data_transmitted_bytes': float(results['total_data_transmitted_bytes']),
                'num_rounds': results['num_rounds']
            }
            json.dump(json_results, f, indent=2)
