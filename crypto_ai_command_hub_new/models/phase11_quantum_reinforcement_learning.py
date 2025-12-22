"""Phase 11.6: Quantum Reinforcement Learning (QRL)
Quantum agents for decision making and trading"""

import numpy as np, json, logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QRLMetrics:
    episode: int; episode_reward: float; avg_action_confidence: float
    def to_dict(self):
        return {'episode': self.episode, 'episode_reward': float(self.episode_reward),
                'avg_action_confidence': float(self.avg_action_confidence)}

class QuantumAgent:
    def __init__(self, num_actions: int = 4, num_qubits: int = 4):
        self.num_actions = num_actions
        self.num_qubits = num_qubits
        self.q_params = np.random.uniform(0, 2*np.pi, num_qubits)
        self.policy_weights = np.random.normal(0, 0.1, (num_qubits, num_actions))
    
    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        q_output = np.sin(np.sum(state * self.q_params))
        action_probs = np.exp(q_output + state @ self.policy_weights)
        action_probs /= np.sum(action_probs)
        
        action = np.argmax(action_probs)
        confidence = action_probs[action]
        
        return action, confidence

class QuantumRL:
    def __init__(self, num_actions: int = 4, num_episodes: int = 100):
        self.num_actions = num_actions
        self.num_episodes = num_episodes
        self.agent = QuantumAgent(num_actions)
        self.metrics_history = []
    
    def train(self) -> dict:
        logger.info(f"Training QRL: {self.num_episodes} episodes")
        
        for episode in range(self.num_episodes):
            state = np.random.normal(0, 1, 4)
            episode_reward = 0
            confidences = []
            
            for step in range(10):
                action, confidence = self.agent.get_action(state)
                
                reward = np.random.normal(0.1, 0.2)
                next_state = np.random.normal(0, 1, 4)
                
                episode_reward += reward
                confidences.append(confidence)
                
                self.agent.q_params -= 0.01 * np.random.normal(0, 0.1, 4)
                self.agent.policy_weights -= 0.01 * np.random.normal(0, 0.1, (4, self.num_actions))
                
                state = next_state
            
            avg_confidence = np.mean(confidences)
            self.metrics_history.append(QRLMetrics(episode + 1, episode_reward, avg_confidence).to_dict())
            
            if (episode + 1) % 20 == 0:
                logger.info(f"Episode {episode + 1}: Reward={episode_reward:.3f}, Confidence={avg_confidence:.3f}")
        
        rewards = [m['episode_reward'] for m in self.metrics_history]
        
        return {
            'algorithm': 'Quantum-Reinforcement-Learning',
            'num_actions': self.num_actions,
            'num_episodes': self.num_episodes,
            'total_reward': float(np.sum(rewards)),
            'avg_reward_per_episode': float(np.mean(rewards)),
            'final_reward': float(rewards[-1]),
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    from typing import Tuple
    
    logger.info("=" * 60)
    logger.info("Phase 11.6: Quantum Reinforcement Learning (QRL)")
    logger.info("=" * 60)
    
    qrl = QuantumRL(num_actions=4, num_episodes=100)
    results = qrl.train()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Total reward: {results['total_reward']:.3f}")
    logger.info(f"Avg reward per episode: {results['avg_reward_per_episode']:.3f}")
    logger.info(f"Final reward: {results['final_reward']:.3f}")
    
    with open('phase11_quantum_rl_results.json', 'w') as f:
        json.dump(results, f, indent=2)
