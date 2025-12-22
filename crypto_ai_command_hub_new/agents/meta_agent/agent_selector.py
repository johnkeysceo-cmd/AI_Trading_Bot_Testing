# agent_selector.py
# -----------------
# Meta-Agent controller: orchestrates all pre-built agents,
# allocates capital, dispatches TradeSignals, and evaluates risk before execution.

from typing import List, Dict
from agents.external._common import TradeSignal, SignalAggregator, calculate_order_size
from execution.ccxt_executor import SafeExecutor, ExchangeManager
from risk.risk_manager import RiskManager
import logging

# -------------------------------
# Logging
# -------------------------------
logger = logging.getLogger("MetaAgent")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# -------------------------------
# Initialize Exchange Manager, SafeExecutor & RiskManager
# -------------------------------
manager = ExchangeManager()
executor = SafeExecutor(manager)
risk_manager = RiskManager()

# -------------------------------
# Agent Imports (adapters for pre-built agents)
# -------------------------------
from execution.adapters.freqtrade_adapter import FreqtradeAdapter
from execution.adapters.jesse_adapter import JesseAdapter
from execution.adapters.octobot_adapter import OctoBotAdapter
from execution.adapters.opentrader_adapter import OpenTraderAdapter
from execution.adapters.ai_cryptotrader_adapter import AICryptoTraderAdapter
from execution.adapters.ml_trader_adapter import MLTraderAdapter
from execution.adapters.rl_trader_adapter import RLTraderAdapter

# -------------------------------
# Meta-Agent Configuration
# -------------------------------
CAPITAL_POOL = 10000.0
ALLOCATION_TIERS = {
    "survival": 0.7,
    "aggressive": 0.2,
    "very_aggressive": 0.08,
    "nuclear": 0.02
}

# -------------------------------
# Initialize Adapters
# -------------------------------
agents = {
    "survival": FreqtradeAdapter(executor),
    "aggressive": JesseAdapter(executor),
    "very_aggressive": MLTraderAdapter(executor),
    "nuclear": RLTraderAdapter(executor),
    "arbitrage": OctoBotAdapter(executor),
    "experimental": AICryptoTraderAdapter(executor)
}

# -------------------------------
# Signal Aggregator
# -------------------------------
aggregator = SignalAggregator()

# -------------------------------
# Core Helper Functions
# -------------------------------
def allocate_capital(tier: str) -> float:
    return CAPITAL_POOL * ALLOCATION_TIERS.get(tier, 0)

# -------------------------------
# Main Signal Dispatch Logic
# -------------------------------
def dispatch_signals(market_data: Dict = None):
    logger.info("Meta-Agent: Dispatching signals from all agents...")

    for tier, adapter in agents.items():
        logger.debug(f"Collecting signals from {tier} agent")
        try:
            signals: List[TradeSignal] = adapter.generate_signals(market_data)
        except Exception as e:
            logger.error(f"Adapter {tier} failed to generate signals: {e}")
            continue

        for signal in signals:
            # Ensure price is available
            if signal.price is None:
                ticker = executor.manager.connectors["binance"].fetch_ticker(signal.symbol)
                signal.price = ticker.get("last", None)
                if signal.price is None:
                    logger.warning(f"Price not available for {signal.symbol}, skipping")
                    continue

            # Calculate position sizing
            tier_capital = allocate_capital(tier)
            signal.amount = calculate_order_size(tier_capital, signal.confidence, signal.price)

            # Aggregate signal
            aggregator.add_signal(signal)
            logger.info(f"{tier.upper()} agent produced signal: {signal}")

    # Filter signals
    valid_signals = aggregator.filter_by_confidence(threshold=0.3)

    # --- RISK CHECK BEFORE EXECUTION ---
    for signal in valid_signals:
        if not risk_manager.evaluate(signal):
            logger.warning(f"RiskManager blocked trade: {signal}")
            continue

        try:
            executor.execute_trade(
                exchange_name="binance",
                symbol=signal.symbol,
                side=signal.side,
                amount=signal.amount,
                price=signal.price
            )
            risk_manager.register_trade(signal)
        except Exception as e:
            logger.error(f"Error executing signal {signal}: {e}")

    aggregator.clear()

# -------------------------------
# Entry Point for Periodic Execution
# -------------------------------
def run_meta_agent_loop(interval_sec: int = 60, market_data: Dict = None):
    import time
    logger.info("Meta-Agent loop starting...")
    while True:
        try:
            dispatch_signals(market_data)
            logger.info(f"Sleeping {interval_sec} seconds before next iteration...")
            time.sleep(interval_sec)
        except KeyboardInterrupt:
            logger.info("Meta-Agent loop terminated by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in meta-agent loop: {e}")
