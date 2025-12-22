from agents.external.crypto_trading_bot_machine_learning import ml_bot
from execution.ccxt_executor import SafeExecutor
from agents.external._common import TradeSignal

class CryptoMLAdapter:
    def __init__(self, safe_executor: SafeExecutor):
        self.executor = safe_executor

    def predict_signal(self, market_data) -> TradeSignal:
        symbol = "BTC/USDT"
        side = "sell"
        amount = 0.0015
        confidence = 0.88
        return TradeSignal(symbol, side, amount, confidence)

    def execute_signal(self, signal: TradeSignal):
        return self.executor.execute_trade("binance", signal.symbol, signal.side, signal.amount)
