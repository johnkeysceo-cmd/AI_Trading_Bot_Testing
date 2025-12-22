"""Phase 11.4: Quantum Support Vector Machine (QSVM)
Quantum kernel methods for classification"""

import numpy as np, json, logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QSVMMetrics:
    iteration: int; kernel_value: float; classification_accuracy: float
    def to_dict(self):
        return {'iteration': self.iteration, 'kernel_value': float(self.kernel_value),
                'classification_accuracy': float(self.classification_accuracy)}

class QuantumKernel:
    def __init__(self, feature_dim: int = 16):
        self.feature_dim = feature_dim
        self.rotation_params = np.random.uniform(0, 2*np.pi, feature_dim)
    
    def compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        phase1 = np.sum(x1 * self.rotation_params)
        phase2 = np.sum(x2 * self.rotation_params)
        kernel_val = np.abs(np.cos(phase1 - phase2))**2
        return kernel_val

class QuantumSVM:
    def __init__(self, feature_dim: int = 16, num_samples: int = 50, num_iterations: int = 100):
        self.feature_dim = feature_dim
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.kernel = QuantumKernel(feature_dim)
        self.metrics_history = []
    
    def train(self) -> dict:
        logger.info(f"Training QSVM: {self.num_iterations} iterations")
        
        for iteration in range(self.num_iterations):
            X = np.random.normal(0, 1, (self.num_samples, self.feature_dim))
            y = np.random.randint(0, 2, self.num_samples)
            
            gram_matrix = np.zeros((self.num_samples, self.num_samples))
            for i in range(self.num_samples):
                for j in range(self.num_samples):
                    gram_matrix[i, j] = self.kernel.compute_kernel(X[i], X[j])
            
            avg_kernel = np.mean(gram_matrix)
            
            predictions = np.sign(np.sum(gram_matrix, axis=1) - self.num_samples / 2)
            predictions = (predictions + 1) / 2
            accuracy = np.mean(predictions == y)
            
            self.metrics_history.append(QSVMMetrics(iteration + 1, avg_kernel, accuracy).to_dict())
            
            self.kernel.rotation_params -= 0.01 * np.random.normal(0, 0.1, self.feature_dim)
            
            if (iteration + 1) % 25 == 0:
                logger.info(f"Iteration {iteration + 1}: Kernel={avg_kernel:.4f}, Acc={accuracy:.3f}")
        
        accs = [m['classification_accuracy'] for m in self.metrics_history]
        
        return {
            'algorithm': 'Quantum-SVM',
            'feature_dim': self.feature_dim,
            'num_samples': self.num_samples,
            'num_iterations': self.num_iterations,
            'final_accuracy': float(accs[-1]),
            'avg_accuracy': float(np.mean(accs)),
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 11.4: Quantum Support Vector Machine (QSVM)")
    logger.info("=" * 60)
    
    qsvm = QuantumSVM(feature_dim=16, num_samples=50, num_iterations=80)
    results = qsvm.train()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Final accuracy: {results['final_accuracy']:.3f}")
    logger.info(f"Avg accuracy: {results['avg_accuracy']:.3f}")
    
    with open('phase11_quantum_svm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
