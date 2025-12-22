import random
from agents.external._common import TradeSignal

class FinRLAgent:
    """
    Sample RL agent producing TradeSignals.
    """
    def __init__(self, name="FinRL"):
        self.name = name

    def generate_signals(self):
        # Random signals for demo
        signals = []
        symbols = ["BTC/USDT", "ETH/USDT"]
        for s in symbols:
            side = random.choice(["buy", "sell"])
            amount = random.uniform(0.001, 0.01)
            confidence = random.uniform(0.3, 1.0)
            signals.append(TradeSignal(s, side, amount, confidence, agent_name=self.name))
        return signals
