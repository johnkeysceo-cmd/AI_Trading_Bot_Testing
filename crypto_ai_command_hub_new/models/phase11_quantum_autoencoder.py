"""Phase 11.7: Quantum Autoencoder (QAE)
Quantum-classical hybrid autoencoder for dimensionality reduction"""

import numpy as np, json, logging
from dataclasses import dataclass
from typing import Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QAEMetrics:
    epoch: int; reconstruction_loss: float; latent_dim_utilization: float
    def to_dict(self):
        return {'epoch': self.epoch, 'reconstruction_loss': float(self.reconstruction_loss),
                'latent_dim_utilization': float(self.latent_dim_utilization)}

class QuantumEncoder:
    def __init__(self, input_dim: int = 8, latent_dim: int = 2):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.weights = np.random.normal(0, 0.1, (input_dim, latent_dim))
        self.bias = np.zeros(latent_dim)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        # Classical encoding layer
        h = x @ self.weights + self.bias
        
        # Quantum encoding: amplitude encoding
        h = np.tanh(h)
        h = h / (np.linalg.norm(h) + 1e-8)
        
        return h
    
    def compute_entanglement(self, encoded: np.ndarray) -> float:
        # Measure entanglement of latent state
        entropy = -np.sum(encoded**2 * np.log(encoded**2 + 1e-8))
        return float(entropy)

class QuantumDecoder:
    def __init__(self, latent_dim: int = 2, output_dim: int = 8):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.weights = np.random.normal(0, 0.1, (latent_dim, output_dim))
        self.bias = np.zeros(output_dim)
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        # Quantum phase to classical mapping
        phases = np.angle(z + 1j * 1e-8)
        
        # Classical decoding layer
        x_recon = np.sin(phases @ self.weights + self.bias)
        
        return x_recon

class QuantumAutoencoder:
    def __init__(self, input_dim: int = 8, latent_dim: int = 2, num_epochs: int = 100):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.encoder = QuantumEncoder(input_dim, latent_dim)
        self.decoder = QuantumDecoder(latent_dim, input_dim)
        self.metrics_history = []
        self.latent_states = []
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = self.encoder.encode(x)
        x_recon = self.decoder.decode(z)
        return x_recon, z
    
    def train(self, X: np.ndarray = None) -> dict:
        if X is None:
            X = np.random.normal(0, 1, (50, self.input_dim))
        
        logger.info(f"Training QAE: {self.num_epochs} epochs on {X.shape[0]} samples")
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            entanglements = []
            
            for x_sample in X:
                # Forward pass
                x_recon, z = self.forward(x_sample)
                
                # Compute reconstruction loss
                loss = np.mean((x_sample - x_recon)**2)
                epoch_loss += loss
                
                # Track entanglement
                entanglement = self.encoder.compute_entanglement(z)
                entanglements.append(entanglement)
                self.latent_states.append(z)
                
                # Backprop: update encoder
                grad_encoder = 2 * (x_recon - x_sample) @ x_sample.reshape(-1, 1)
                self.encoder.weights -= 0.01 * np.mean(grad_encoder, axis=0).reshape(self.input_dim, 1)
                
                # Backprop: update decoder
                grad_decoder = 2 * (x_recon - x_sample) @ z.reshape(-1, 1)
                self.decoder.weights -= 0.01 * np.mean(grad_decoder, axis=0).reshape(self.latent_dim, 1)
            
            avg_loss = epoch_loss / len(X)
            avg_entanglement = np.mean(entanglements)
            
            self.metrics_history.append(
                QAEMetrics(epoch + 1, avg_loss, avg_entanglement).to_dict()
            )
            
            if (epoch + 1) % 25 == 0:
                logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Entanglement={avg_entanglement:.4f}")
        
        return {
            'algorithm': 'Quantum-Autoencoder',
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'num_epochs': self.num_epochs,
            'final_loss': float(self.metrics_history[-1]['reconstruction_loss']),
            'avg_loss': float(np.mean([m['reconstruction_loss'] for m in self.metrics_history])),
            'avg_entanglement': float(np.mean([m['latent_dim_utilization'] for m in self.metrics_history])),
            'compression_ratio': float(self.input_dim / self.latent_dim),
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 11.7: Quantum Autoencoder (QAE)")
    logger.info("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.normal(0, 1, (50, 8))
    
    qae = QuantumAutoencoder(input_dim=8, latent_dim=2, num_epochs=100)
    results = qae.train(X_train)
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Final reconstruction loss: {results['final_loss']:.4f}")
    logger.info(f"Average loss: {results['avg_loss']:.4f}")
    logger.info(f"Average entanglement: {results['avg_entanglement']:.4f}")
    logger.info(f"Compression ratio: {results['compression_ratio']:.1f}x")
    
    with open('phase11_quantum_autoencoder_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✓ Results saved to phase11_quantum_autoencoder_results.json")
