"""
_common.py
-----------
Shared utilities, classes, and constants for all AI crypto agents.
This module standardizes signals, logging, risk flags, and feature helpers.
All adapters and external agents import this module.
"""

import logging
import time
from typing import List, Optional, Dict

# -------------------------------
# Logging Setup (shared across agents)
# -------------------------------
logger = logging.getLogger("CryptoAICommon")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# -------------------------------
# Constants
# -------------------------------
DEFAULT_CURRENCY = "USD"
MIN_TRADE_AMOUNT = 0.0001  # Minimum trade size
MAX_CONFIDENCE = 1.0
MIN_CONFIDENCE = 0.0

# -------------------------------
# Trade Signal Class
# -------------------------------
class TradeSignal:
    """
    Standardized trade signal class for all agents.

    Attributes:
        symbol (str): Trading pair symbol, e.g., "BTC/USDT"
        side (str): "buy" or "sell"
        amount (float): Amount to trade
        confidence (float): Confidence level (0.0 - 1.0)
        price (Optional[float]): Optional limit price for orders
        agent_name (Optional[str]): Name of agent producing this signal
        timestamp (float): Epoch time of signal generation
    """
    def __init__(self, symbol: str, side: str, amount: float, 
                 confidence: float = 1.0, price: Optional[float] = None, 
                 agent_name: Optional[str] = None):
        self.symbol = symbol
        self.side = side.lower()
        self.amount = amount
        self.confidence = min(max(confidence, MIN_CONFIDENCE), MAX_CONFIDENCE)
        self.price = price
        self.agent_name = agent_name
        self.timestamp = time.time()

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "amount": self.amount,
            "confidence": self.confidence,
            "price": self.price,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp
        }

    def __repr__(self):
        return f"<TradeSignal {self.side.upper()} {self.amount} {self.symbol} | conf={self.confidence:.2f} | agent={self.agent_name}>"

# -------------------------------
# Risk / Capital Flags
# -------------------------------
class RiskFlag:
    """
    Shared risk signal for agents and meta-agent
    """
    def __init__(self, risk_level: str, description: str, severity: int):
        """
        Args:
            risk_level (str): "low", "medium", "high", "nuclear"
            description (str): Description of risk
            severity (int): Integer scale 1-10
        """
        self.risk_level = risk_level
        self.description = description
        self.severity = severity
        self.timestamp = time.time()

    def to_dict(self) -> Dict:
        return {
            "risk_level": self.risk_level,
            "description": self.description,
            "severity": self.severity,
            "timestamp": self.timestamp
        }

    def __repr__(self):
        return f"<RiskFlag {self.risk_level.upper()} | severity={self.severity} | desc={self.description}>"

# -------------------------------
# Utility Functions
# -------------------------------
def normalize_symbol(symbol: str) -> str:
    """Standardize trading pair symbols."""
    return symbol.replace("_", "/").upper()

def clamp_confidence(confidence: float) -> float:
    """Clamp confidence between 0.0 and 1.0."""
    return max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))

def calculate_order_size(balance: float, risk_fraction: float, price: float) -> float:
    """
    Compute position size based on balance and risk fraction.
    """
    amount = (balance * risk_fraction) / price
    return max(amount, MIN_TRADE_AMOUNT)

def current_timestamp() -> float:
    return time.time()

# -------------------------------
# Signal Aggregator
# -------------------------------
class SignalAggregator:
    """
    Collects and filters signals from multiple agents
    """
    def __init__(self):
        self.signals: List[TradeSignal] = []

    def add_signal(self, signal: TradeSignal):
        self.signals.append(signal)
        logger.debug(f"Added signal: {signal}")

    def filter_by_confidence(self, threshold: float = 0.5) -> List[TradeSignal]:
        filtered = [s for s in self.signals if s.confidence >= threshold]
        logger.debug(f"Filtered signals (conf >= {threshold}): {len(filtered)} / {len(self.signals)}")
        return filtered

    def clear(self):
        self.signals.clear()
        logger.debug("Signal aggregator cleared.")

# -------------------------------
# Example Helper: Risk Evaluation
# -------------------------------
def evaluate_risk(signal: TradeSignal) -> RiskFlag:
    """
    Very simple risk mapping based on confidence and side.
    """
    if signal.confidence < 0.2:
        return RiskFlag("high", "Very low confidence trade", 9)
    elif signal.confidence < 0.5:
        return RiskFlag("medium", "Moderate confidence", 5)
    elif signal.confidence < 0.8:
        return RiskFlag("low", "High confidence", 2)
    else:
        return RiskFlag("low", "Very high confidence", 1)
