# execution/adapters/rltrader_adapter.py
from agents.external.RLTrader import rl_trader
from execution.ccxt_executor import SafeExecutor
from agents.external._common import TradeSignal
import random

class RLTraderAdapter:
    def __init__(self, safe_executor: SafeExecutor):
        self.executor = safe_executor

    def generate_signals(self):
        symbols = ["BTC/USDT", "ETH/USDT", "LINK/USDT"]
        signals = []
        for s in symbols:
            side = random.choice(["buy", "sell"])
            amount = round(random.uniform(0.0005, 0.02), 6)
            confidence = round(random.uniform(0.6, 1.0), 2)
            signals.append(TradeSignal(symbol=s, side=side, amount=amount, confidence=confidence, agent_name="RLTrader"))
        return signals

    def execute_signal(self, signal: TradeSignal):
        return self.executor.execute_trade("binance", signal.symbol, signal.side, signal.amount)
