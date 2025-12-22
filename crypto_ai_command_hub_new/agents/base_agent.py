"""
base_agent.py
--------------
Core AI agent class for the multi-agent crypto trading hub.

Features:
- Multi-tier agent types (Survival, Aggressive, Nuclear, Arbitrage)
- Signal computation (price prediction + feature evaluation)
- Capital allocation integration with CCXT executor
- Risk management and stop-loss enforcement
- Logging and monitoring
- Backtesting-ready structure
- Multi-agent coordination hooks

Notes:
- Designed for multi-agent ecosystem
- Extensible for RL agents, LLM reasoning, sentiment analysis, and arbitrage
- Each agent can operate independently or under meta-agent supervision
"""

import time
import logging
import random
from typing import Dict, Any, List, Optional
import numpy as np

# Import execution engine
from execution.ccxt_executor import SafeExecutor, ExchangeManager

# Import ML modules (placeholders for now)
from models.classical_ml import MLModelManager  # To be implemented
from feature_store.technical import TechnicalFeaturePipeline  # To be implemented

# Logging setup
logger = logging.getLogger("BaseAgent")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Supported agent tiers
AGENT_TIERS = ["survival", "aggressive", "nuclear", "arbitrage"]

# Risk parameters per tier (configurable)
RISK_CONFIG = {
    "survival": {"max_alloc": 0.25, "stop_loss_pct": 0.01, "description": "Conservative steady growth"},
    "aggressive": {"max_alloc": 0.5, "stop_loss_pct": 0.05, "description": "Medium risk, medium reward"},
    "nuclear": {"max_alloc": 1.0, "stop_loss_pct": 0.2, "description": "High risk, extreme reward"},
    "arbitrage": {"max_alloc": 0.75, "stop_loss_pct": 0.01, "description": "Market-neutral spreads"}
}


class BaseAgent:
    """
    Single AI agent within the multi-agent ecosystem.
    Handles decision-making, capital allocation, and execution.
    """

    def __init__(
        self,
        name: str,
        tier: str = "survival",
        initial_capital: float = 1000.0,
        safe_executor: Optional[SafeExecutor] = None
    ):
        if tier not in AGENT_TIERS:
            raise ValueError(f"Invalid agent tier {tier}")

        self.name = name
        self.tier = tier
        self.capital = initial_capital
        self.executor = safe_executor or SafeExecutor(ExchangeManager())
        self.max_alloc = RISK_CONFIG[tier]["max_alloc"]
        self.stop_loss_pct = RISK_CONFIG[tier]["stop_loss_pct"]
        self.description = RISK_CONFIG[tier]["description"]

        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position details
        self.trade_history: List[Dict[str, Any]] = []

        # Feature pipeline and ML model placeholders
        self.feature_pipeline = TechnicalFeaturePipeline()
        self.ml_model_manager = MLModelManager()

        # Multi-agent coordination hooks
        self.meta_agent = None  # To be assigned later

        logger.info(f"[{self.name}] Initialized agent (tier={self.tier}, capital=${self.capital})")

    # -----------------------------
    # Capital Allocation Methods
    # -----------------------------
    def allocate_capital(self, trade_amount: float) -> float:
        """
        Compute allowed capital per trade based on tier and current holdings
        """
        alloc = min(trade_amount, self.capital * self.max_alloc)
        logger.debug(f"[{self.name}] Allocating ${alloc:.2f} for trade (tier={self.tier})")
        return alloc

    def update_capital(self, pnl: float):
        self.capital += pnl
        logger.debug(f"[{self.name}] Updated capital: ${self.capital:.2f} (PnL={pnl:.2f})")

    # -----------------------------
    # Signal Generation Methods
    # -----------------------------
    def compute_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Compute buy/sell/hold signals for a symbol
        Combines:
        - ML prediction (price movement)
        - Technical features
        - Tier-based risk weighting
        """
        features = self.feature_pipeline.compute_features(symbol)
        predicted_movement = self.ml_model_manager.predict(symbol, features)

        # Convert ML prediction to trade signal
        signal_strength = np.clip(predicted_movement, -1, 1)  # -1=sell, 1=buy
        trade_signal = {
            "symbol": symbol,
            "side": "buy" if signal_strength > 0 else "sell",
            "confidence": abs(signal_strength),
            "amount_pct": abs(signal_strength) * self.max_alloc
        }

        logger.debug(f"[{self.name}] Signal for {symbol}: {trade_signal}")
        return trade_signal

    # -----------------------------
    # Execution Methods
    # -----------------------------
    def execute_signal(self, signal: Dict[str, Any], exchange_name: str):
        """
        Execute a trade based on computed signal
        """
        amount = self.allocate_capital(self.capital * signal["amount_pct"])
        side = signal["side"]
        symbol = signal["symbol"]

        if amount <= 0:
            logger.warning(f"[{self.name}] Trade amount too low, skipping execution")
            return

        result = self.executor.execute_trade(exchange_name, symbol, side, amount)
        if result["status"] == "success":
            self.positions[symbol] = {
                "side": side,
                "amount": amount,
                "timestamp": time.time()
            }
            self.trade_history.append(result["order"])
            logger.info(f"[{self.name}] Trade executed: {result['order']}")
        else:
            logger.error(f"[{self.name}] Trade failed: {result}")

    # -----------------------------
    # Risk Management
    # -----------------------------
    def check_stop_loss(self, current_prices: Dict[str, float]):
        """
        Check if positions need to be closed due to stop loss
        """
        for symbol, pos in list(self.positions.items()):
            price_now = current_prices.get(symbol)
            if not price_now:
                continue
            entry_price = pos.get("price", price_now)
            if pos["side"] == "buy":
                loss_pct = (entry_price - price_now) / entry_price
            else:
                loss_pct = (price_now - entry_price) / entry_price

            if loss_pct > self.stop_loss_pct:
                logger.warning(f"[{self.name}] Stop loss triggered for {symbol} (loss={loss_pct:.2%})")
                # Close position (simplified for example)
                self.executor.execute_trade("binance", symbol, "sell" if pos["side"] == "buy" else "buy", pos["amount"])
                self.positions.pop(symbol)

    # -----------------------------
    # Backtesting Hooks
    # -----------------------------
    def backtest_step(self, historical_data: Dict[str, Any], exchange_name: str):
        """
        Single backtesting step
        """
        for symbol, price in historical_data.items():
            signal = self.compute_signal(symbol)
            # Override execution to simulate PnL
            pnl = (random.random() - 0.5) * 2 * self.capital * 0.01  # ±1% random
            self.update_capital(pnl)
            self.trade_history.append({
                "symbol": symbol,
                "pnl": pnl,
                "timestamp": time.time()
            })

    # -----------------------------
    # Meta-Agent Hooks
    # -----------------------------
    def assign_meta_agent(self, meta_agent):
        """
        Link this agent to meta-agent controller
        """
        self.meta_agent = meta_agent
        logger.info(f"[{self.name}] Meta-agent assigned")

    def report_status(self):
        """
        Returns a summary of agent's capital and active positions
        """
        status = {
            "name": self.name,
            "tier": self.tier,
            "capital": self.capital,
            "positions": self.positions,
            "trade_count": len(self.trade_history)
        }
        logger.info(f"[{self.name}] Status report: {status}")
        return status

# -----------------------------
# Example instantiation
# -----------------------------
if __name__ == "__main__":
    safe_executor = SafeExecutor(ExchangeManager())
    agent = BaseAgent("ConservativeAgent", tier="survival", initial_capital=1000, safe_executor=safe_executor)

    # Simulate signal and execution
    fake_signal = {"symbol": "BTC/USDT", "side": "buy", "confidence": 0.7, "amount_pct": 0.2}
    agent.execute_signal(fake_signal, exchange_name="binance")
    agent.report_status()
