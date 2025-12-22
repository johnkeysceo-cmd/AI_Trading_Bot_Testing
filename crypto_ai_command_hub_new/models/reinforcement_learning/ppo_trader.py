"""PPO trader wrapper for Phase 2.

Scaffolded implementation for a PPO-based trading agent. Replace the
placeholder environment with a `gym`-like trading environment for
real training. The wrapper detects `stable_baselines3` if installed
and uses it; otherwise methods raise informative errors.
"""
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    _HAS_SB3 = True
except Exception:
    _HAS_SB3 = False


class PPOTrader:
    """Wrapper to train and run a PPO trading agent.

    Usage: instantiate with a `gym.Env` trading environment and call
    `build_model`, `train`, `act`, `save`, `load` as needed.
    """

    def __init__(self, env=None, model_path: Optional[str] = None):
        self.env = env
        self.model_path = model_path
        self.model = None

    def build_model(self, **kwargs):
        if not _HAS_SB3:
            logger.warning("stable_baselines3 not installed — build is a noop")
            return
        vec_env = DummyVecEnv([lambda: self.env])
        self.model = PPO('MlpPolicy', vec_env, verbose=0, **kwargs)

    def train(self, timesteps: int = 10000):
        if not _HAS_SB3 or self.model is None:
            raise RuntimeError("PPO not configured. Install stable_baselines3 and call build_model().")
        self.model.learn(total_timesteps=timesteps)

    def act(self, obs):
        if not _HAS_SB3 or self.model is None:
            raise RuntimeError("PPO model not loaded")
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def save(self, path: str):
        if not _HAS_SB3 or self.model is None:
            raise RuntimeError("Nothing to save")
        self.model.save(path)

    def load(self, path: str):
        if not _HAS_SB3:
            raise RuntimeError("stable_baselines3 not installed")
        self.model = PPO.load(path)


if __name__ == "__main__":
    print("PPOTrader scaffold. Replace env with a real trading env.")
