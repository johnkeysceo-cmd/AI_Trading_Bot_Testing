# risk_manager.py
# -----------------
# Centralized risk management for all agents.
# Evaluates TradeSignals before execution and blocks risky trades.

from agents.external._common import TradeSignal, RiskFlag
import logging

logger = logging.getLogger("RiskManager")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Example configurable thresholds
MAX_POSITION_SIZE = 0.1          # Max fraction of total capital per trade
MAX_EXPOSURE = 0.5               # Max fraction of total capital per agent
STOP_LOSS_THRESHOLD = 0.2        # Max allowable loss fraction
MAX_DRAWUP_PER_TRADE = 0.2       # Max gain per trade to avoid over-leveraging

class RiskManager:
    def __init__(self, capital_pool: float = 10000):
        self.capital_pool = capital_pool
        self.agent_exposure = {}  # Track exposure per agent

    def evaluate(self, signal: TradeSignal) -> bool:
        """
        Returns True if trade passes risk checks, False otherwise.
        """
        tier = signal.agent_name or "unknown"
        exposure = self.agent_exposure.get(tier, 0.0)

        # Check position size
        if signal.amount > MAX_POSITION_SIZE * self.capital_pool:
            logger.warning(f"Trade blocked: {tier} position too large ({signal.amount})")
            return False

        # Check exposure per agent
        if exposure + signal.amount > MAX_EXPOSURE * self.capital_pool:
            logger.warning(f"Trade blocked: {tier} exposure limit reached ({exposure + signal.amount})")
            return False

        # Check confidence
        if signal.confidence < 0.2:
            logger.warning(f"Trade blocked: {tier} confidence too low ({signal.confidence})")
            return False

        # Passed all checks
        self.agent_exposure[tier] = exposure + signal.amount
        logger.info(f"Trade approved for {tier}: {signal}")
        return True

    def reset_exposures(self):
        """Reset all tracked exposures."""
        self.agent_exposure.clear()
        logger.debug("Agent exposures reset.")
