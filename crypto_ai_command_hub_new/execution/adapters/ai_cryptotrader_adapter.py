# execution/adapters/ai_cryptotrader_adapter.py
from agents.external.AI_CryptoTrader import ai_cryptotrader
from execution.ccxt_executor import SafeExecutor
from agents.external._common import TradeSignal
import random

class AICryptoTraderAdapter:
    def __init__(self, safe_executor: SafeExecutor):
        self.executor = safe_executor

    def generate_signals(self):
        symbols = ["BTC/USDT", "ETH/USDT", "DOT/USDT"]
        signals = []
        for s in symbols:
            side = random.choice(["buy", "sell"])
            amount = round(random.uniform(0.001, 0.015), 6)
            confidence = round(random.uniform(0.5, 0.95), 2)
            signals.append(TradeSignal(symbol=s, side=side, amount=amount, confidence=confidence, agent_name="AI-CryptoTrader"))
        return signals

    def execute_signal(self, signal: TradeSignal):
        return self.executor.execute_trade("binance", signal.symbol, signal.side, signal.amount)
