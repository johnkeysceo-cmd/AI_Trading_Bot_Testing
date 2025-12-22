from agents.external._common import TradeSignal
from execution.ccxt_executor import SafeExecutor
import random

class FreqtradeAdapter:
    def __init__(self, safe_executor: SafeExecutor):
        self.executor = safe_executor
        self.name = "Freqtrade"

    def generate_signals(self):
        # Placeholder logic: replace with actual Freqtrade outputs
        signals = []
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            side = random.choice(["buy", "sell"])
            amount = random.uniform(0.001, 0.01)
            confidence = random.uniform(0.3, 1.0)
            signals.append(TradeSignal(symbol, side, amount, confidence, agent_name=self.name))
        return signals

    def execute_signal(self, signal: TradeSignal):
        return self.executor.execute_trade("binance", signal.symbol, signal.side, signal.amount)
