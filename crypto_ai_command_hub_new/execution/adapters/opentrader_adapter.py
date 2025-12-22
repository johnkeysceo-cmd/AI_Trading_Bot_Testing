# execution/adapters/opentrader_adapter.py
from agents.external.opentrader import opentrader
from execution.ccxt_executor import SafeExecutor
from agents.external._common import TradeSignal
import random

class OpenTraderAdapter:
    def __init__(self, safe_executor: SafeExecutor):
        self.executor = safe_executor

    def generate_signals(self):
        symbols = ["BTC/USDT", "SOL/USDT", "MATIC/USDT"]
        signals = []
        for s in symbols:
            side = random.choice(["buy", "sell"])
            amount = round(random.uniform(0.002, 0.03), 6)
            confidence = round(random.uniform(0.4, 1.0), 2)
            signals.append(TradeSignal(symbol=s, side=side, amount=amount, confidence=confidence, agent_name="OpenTrader"))
        return signals

    def execute_signal(self, signal: TradeSignal):
        return self.executor.execute_trade("binance", signal.symbol, signal.side, signal.amount)
