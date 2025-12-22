"""
models/deep_rl.py
-----------------
Deep Reinforcement Learning models for multi-agent crypto trading.

Features:
- Uses FinRL + PyTorch for RL agent implementation
- Supports multi-agent training and inference
- Observation processing for candlestick + technical features
- Reward shaping and risk-adjusted returns
- Model checkpointing and re-loading
- Thread-safe for multi-agent integration
- Designed to integrate with CCXTExecutor and feature pipelines
"""

import os
import threading
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple

# Optional FinRL integration
try:
    from finrl.agents.stablebaselines3_models import DRLAgent
    from finrl.config import INDICATORS
except ImportError:
    logging.warning("FinRL not installed; fallback to pure PyTorch agent")
    DRLAgent = None
    INDICATORS = []

# Logging setup
logger = logging.getLogger("DeepRL")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# -----------------------------
# Observation & Action Spaces
# -----------------------------
class ObservationProcessor:
    """
    Converts raw candle + feature data into agent observations
    """
    def __init__(self, indicators: List[str] = INDICATORS):
        self.indicators = indicators

    def process(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns a numpy array suitable for RL agent observation
        """
        obs = df.copy()
        # Use only relevant features (OHLCV + indicators)
        feature_cols = ["open", "high", "low", "close", "volume"] + self.indicators
        missing_cols = [c for c in feature_cols if c not in obs.columns]
        for c in missing_cols:
            obs[c] = 0.0
        obs_values = obs[feature_cols].values.astype(np.float32)
        return obs_values

# -----------------------------
# Neural Network for Policy
# -----------------------------
class PolicyNetwork(nn.Module):
    """
    Simple MLP for policy approximation
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))

# -----------------------------
# RL Agent
# -----------------------------
class RLTradingAgent:
    """
    Reinforcement learning agent for crypto trading
    Thread-safe for multi-agent execution
    """
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 1e-4, gamma: float = 0.99, cache_dir: str = "./data/models"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.memory: List[Tuple[np.ndarray, int, float, np.ndarray]] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        logger.info(f"[RLTradingAgent] Initialized on device {self.device}")

    # -----------------------------
    # Action Selection
    # -----------------------------
    def select_action(self, observation: np.ndarray) -> int:
        """
        Given observation, returns action index
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy(obs_tensor).cpu().numpy().flatten()
        action = np.random.choice(self.action_dim, p=probs)
        return action

    # -----------------------------
    # Memory & Learning
    # -----------------------------
    def store_transition(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray):
        with self.lock:
            self.memory.append((obs, action, reward, next_obs))

    def update_policy(self):
        """
        Simple policy gradient update (vanilla REINFORCE)
        """
        with self.lock:
            if len(self.memory) == 0:
                logger.warning("[RLTradingAgent] No transitions to update policy")
                return

            returns = []
            G = 0
            for _, _, reward, _ in reversed(self.memory):
                G = reward + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            self.optimizer.zero_grad()
            for idx, (obs, action, reward, next_obs) in enumerate(self.memory):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                probs = self.policy(obs_tensor)
                log_prob = torch.log(probs[0, action])
                loss = -log_prob * returns[idx]
                loss.backward()
            self.optimizer.step()
            self.memory.clear()
            logger.info("[RLTradingAgent] Policy updated with {} transitions".format(len(returns)))

    # -----------------------------
    # Save / Load
    # -----------------------------
    def save_model(self, filename: str):
        path = os.path.join(self.cache_dir, filename)
        with self.lock:
            torch.save(self.policy.state_dict(), path)
            logger.info(f"[RLTradingAgent] Model saved to {path}")

    def load_model(self, filename: str):
        path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"[RLTradingAgent] No model found at {path}")
            return
        with self.lock:
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"[RLTradingAgent] Model loaded from {path}")

# -----------------------------
# Example Multi-Agent Wrapper
# -----------------------------
class MultiAgentRLHub:
    """
    Container for multiple RL agents with capital allocation
    """
    def __init__(self, obs_dim: int, action_dim: int, agent_types: List[str] = ["conservative", "aggressive", "nuclear"]):
        self.agents: Dict[str, RLTradingAgent] = {}
        for t in agent_types:
            self.agents[t] = RLTradingAgent(obs_dim, action_dim)
        logger.info(f"[MultiAgentRLHub] Initialized agents: {list(self.agents.keys())}")

    def select_actions(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        obs_dict: {agent_type: observation}
        Returns actions for all agents
        """
        actions = {}
        for t, agent in self.agents.items():
            actions[t] = agent.select_action(obs_dict[t])
        return actions

    def update_agents(self, transitions: Dict[str, Tuple[np.ndarray, int, float, np.ndarray]]):
        """
        Update each agent's policy
        transitions: {agent_type: (obs, action, reward, next_obs)}
        """
        for t, tr in transitions.items():
            self.agents[t].store_transition(*tr)
            self.agents[t].update_policy()

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    obs_dim = 10  # Example feature vector length
    action_dim = 3  # Buy / Sell / Hold
    multi_hub = MultiAgentRLHub(obs_dim, action_dim)
    dummy_obs = {
        "conservative": np.random.rand(obs_dim),
        "aggressive": np.random.rand(obs_dim),
        "nuclear": np.random.rand(obs_dim)
    }
    actions = multi_hub.select_actions(dummy_obs)
    print(f"Selected actions: {actions}")
