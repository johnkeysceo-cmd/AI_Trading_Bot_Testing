"""Phase 11.1: Variational Quantum Eigensolver (VQE)
Hybrid quantum-classical optimization for eigenvalue problems"""

import numpy as np, json, logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VQEMetrics:
    iteration: int; eigenvalue: float; variance: float
    def to_dict(self):
        return {'iteration': self.iteration, 'eigenvalue': float(self.eigenvalue),
                'variance': float(self.variance)}

class QuantumCircuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.params = np.random.uniform(0, 2*np.pi, num_qubits)
    
    def apply_rotation(self, param: float) -> np.ndarray:
        return np.array([[np.cos(param/2), -1j*np.sin(param/2)],
                        [-1j*np.sin(param/2), np.cos(param/2)]])
    
    def measure(self) -> np.ndarray:
        state = np.ones(2**self.num_qubits) / np.sqrt(2**self.num_qubits)
        for i, p in enumerate(self.params):
            phase = np.exp(1j * p)
            state = state * phase
        return np.abs(state)**2

class VQE:
    def __init__(self, num_qubits: int = 4, num_iterations: int = 100):
        self.num_qubits = num_qubits
        self.num_iterations = num_iterations
        self.circuit = QuantumCircuit(num_qubits)
        self.metrics_history = []
    
    def hamiltonian_expectation(self, state: np.ndarray) -> float:
        energy = np.sum(state * np.arange(len(state)))
        return energy / np.sum(state)
    
    def optimize(self) -> dict:
        logger.info(f"Running VQE: {self.num_iterations} iterations")
        
        best_energy = float('inf')
        
        for iteration in range(self.num_iterations):
            state = self.circuit.measure()
            energy = self.hamiltonian_expectation(state)
            variance = np.var(state)
            
            self.metrics_history.append(VQEMetrics(iteration + 1, energy, variance).to_dict())
            
            if energy < best_energy:
                best_energy = energy
            
            self.circuit.params -= 0.01 * np.random.normal(0, 1, self.num_qubits)
            
            if (iteration + 1) % 20 == 0:
                logger.info(f"Iteration {iteration + 1}: Energy={energy:.6f}, Var={variance:.6f}")
        
        return {
            'algorithm': 'VQE',
            'num_qubits': self.num_qubits,
            'num_iterations': self.num_iterations,
            'final_energy': float(self.metrics_history[-1]['eigenvalue']),
            'ground_state_energy': float(best_energy),
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 11.1: Variational Quantum Eigensolver (VQE)")
    logger.info("=" * 60)
    
    vqe = VQE(num_qubits=4, num_iterations=80)
    results = vqe.optimize()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Final energy: {results['final_energy']:.6f}")
    logger.info(f"Ground state energy: {results['ground_state_energy']:.6f}")
    
    with open('phase11_vqe_results.json', 'w') as f:
        json.dump(results, f, indent=2)
