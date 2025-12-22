# execution/adapters/jesse_adapter.py
from agents.external.jesse import jesse  # cloned repo
from execution.ccxt_executor import SafeExecutor
from agents.external._common import TradeSignal
import random

class JesseAdapter:
    def __init__(self, safe_executor: SafeExecutor):
        self.executor = safe_executor
        # Initialize Jesse bot internally if needed
        # self.bot = jesse.Broker(...)

    def generate_signals(self):
        """
        Generate a list of TradeSignal objects from Jesse strategies
        """
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        signals = []
        for s in symbols:
            side = random.choice(["buy", "sell"])
            amount = round(random.uniform(0.001, 0.01), 6)
            confidence = round(random.uniform(0.5, 1.0), 2)
            signals.append(TradeSignal(symbol=s, side=side, amount=amount, confidence=confidence, agent_name="Jesse"))
        return signals

    def execute_signal(self, signal: TradeSignal):
        return self.executor.execute_trade("binance", signal.symbol, signal.side, signal.amount)
