# execution/adapters/octobot_adapter.py
from agents.external.OctoBot import octobot
from execution.ccxt_executor import SafeExecutor
from agents.external._common import TradeSignal
import random

class OctoBotAdapter:
    def __init__(self, safe_executor: SafeExecutor):
        self.executor = safe_executor

    def generate_signals(self):
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        signals = []
        for s in symbols:
            side = random.choice(["buy", "sell"])
            amount = round(random.uniform(0.001, 0.02), 6)
            confidence = round(random.uniform(0.3, 1.0), 2)
            signals.append(TradeSignal(symbol=s, side=side, amount=amount, confidence=confidence, agent_name="OctoBot"))
        return signals

    def execute_signal(self, signal: TradeSignal):
        return self.executor.execute_trade("binance", signal.symbol, signal.side, signal.amount)
