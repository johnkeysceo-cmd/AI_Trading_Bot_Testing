from typing import Dict
from agents.external._common import TradeSignal, RiskFlag

class PortfolioRisk:
    """
    Track positions and ensure allocations are within limits.
    """

    def __init__(self, capital_pool: float):
        self.capital_pool = capital_pool
        self.positions: Dict[str, float] = {}  # symbol -> amount

    def update_position(self, signal: TradeSignal):
        self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) + signal.amount

    def get_total_exposure(self):
        return sum(self.positions.values())

    def check_signal(self, signal: TradeSignal) -> RiskFlag:
        projected_exposure = self.get_total_exposure() + signal.amount
        if projected_exposure > self.capital_pool:
            return RiskFlag("high", f"Exceeds capital pool: {projected_exposure}", 10)
        return RiskFlag("low", "Safe trade", 1)
