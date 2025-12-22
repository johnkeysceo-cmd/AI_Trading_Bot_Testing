"""
controller.py
--------------
Meta-Agent Controller for multi-agent AI crypto trading hub.

Responsibilities:
- Coordinate multiple agents (survival, aggressive, nuclear, arbitrage)
- Dynamically allocate capital between agents
- Monitor agent performance and risk
- Decide which agent executes trades on which exchange
- Logging and dashboards for visibility
- Extensible for RL, arbitrage, and emergent multi-agent strategies
"""

import logging
import time
from typing import List, Dict, Optional
import random

from agents.base_agent import BaseAgent, AGENT_TIERS
from execution.ccxt_executor import SafeExecutor, ExchangeManager

# Logging setup
logger = logging.getLogger("MetaAgentController")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# -----------------------------
# Meta-Agent Controller Class
# -----------------------------
class MetaAgentController:
    """
    The Meta-Agent coordinates all individual trading agents.
    """

    def __init__(self, initial_capital: float = 1000.0):
        self.agents: List[BaseAgent] = []
        self.total_capital = initial_capital
        self.executor = SafeExecutor(ExchangeManager())
        self.agent_allocation: Dict[str, float] = {}  # name -> allocated capital
        self.agent_performance: Dict[str, float] = {}  # name -> cumulative PnL

        # Create one agent per tier by default
        for tier in AGENT_TIERS:
            agent_name = f"{tier.capitalize()}Agent"
            agent = BaseAgent(name=agent_name, tier=tier, initial_capital=initial_capital * 0.25, safe_executor=self.executor)
            agent.assign_meta_agent(self)
            self.agents.append(agent)
            self.agent_allocation[agent_name] = agent.capital
            self.agent_performance[agent_name] = 0.0

        logger.info(f"[MetaAgent] Initialized with agents: {', '.join([a.name for a in self.agents])}")

    # -----------------------------
    # Capital Management
    # -----------------------------
    def redistribute_capital(self):
        """
        Dynamically redistribute capital based on performance and risk
        - Survival agents get steady capital
        - High-performing aggressive/nuclear agents get more allocation
        """
        logger.debug("[MetaAgent] Redistributing capital among agents")
        total_cap = sum(agent.capital for agent in self.agents)
        for agent in self.agents:
            if agent.tier == "survival":
                self.agent_allocation[agent.name] = agent.capital  # Keep steady
            else:
                performance_factor = max(self.agent_performance.get(agent.name, 0), 0.1)
                alloc = agent.capital + performance_factor * 0.1 * total_cap
                self.agent_allocation[agent.name] = min(alloc, agent.capital * agent.max_alloc)
            logger.debug(f"[MetaAgent] {agent.name} allocation set to {self.agent_allocation[agent.name]:.2f}")

    # -----------------------------
    # Agent Selection Logic
    # -----------------------------
    def select_agent_for_trade(self) -> BaseAgent:
        """
        Selects which agent should execute the next trade
        - Weighted by tier and performance
        """
        weights = []
        for agent in self.agents:
            base_weight = {"survival": 0.2, "aggressive": 0.3, "nuclear": 0.4, "arbitrage": 0.1}[agent.tier]
            perf_bonus = max(self.agent_performance.get(agent.name, 0), 0)
            weights.append(base_weight + perf_bonus)
        # Normalize weights
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        selected_agent = random.choices(self.agents, probabilities, k=1)[0]
        logger.info(f"[MetaAgent] Selected {selected_agent.name} for next trade")
        return selected_agent

    # -----------------------------
    # Trade Execution
    # -----------------------------
    def execute_trade_cycle(self, symbol: str, exchange_name: str = "binance"):
        """
        Executes one trading cycle:
        - Select agent
        - Compute signal
        - Execute trade
        - Update performance tracking
        """
        agent = self.select_agent_for_trade()
        signal = agent.compute_signal(symbol)
        agent.execute_signal(signal, exchange_name)

        # Simulate PnL tracking
        pnl = (random.random() - 0.5) * 2 * agent.capital * 0.01  # ±1% random for demo
        agent.update_capital(pnl)
        self.agent_performance[agent.name] += pnl
        logger.debug(f"[MetaAgent] Updated {agent.name} performance: {self.agent_performance[agent.name]:.2f}")

        # Redistribute capital dynamically
        self.redistribute_capital()

    # -----------------------------
    # Monitoring & Dashboard
    # -----------------------------
    def report_status(self):
        """
        Aggregate report for all agents
        """
        report = {}
        for agent in self.agents:
            status = agent.report_status()
            report[agent.name] = status
        logger.info(f"[MetaAgent] Aggregate Status Report: {report}")
        return report

    # -----------------------------
    # Backtesting Mode
    # -----------------------------
    def backtest(self, historical_data: Dict[str, List[float]], symbol: str):
        """
        Run a backtest across all agents
        historical_data: dict of {symbol: price_series}
        """
        logger.info(f"[MetaAgent] Starting backtest for {symbol}")
        for t, price in enumerate(historical_data.get(symbol, [])):
            for agent in self.agents:
                # Simulate step with price data
                agent.backtest_step({symbol: price}, exchange_name="paper")
        logger.info(f"[MetaAgent] Backtest completed")

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Initialize meta-agent with $1000 capital
    meta_agent = MetaAgentController(initial_capital=1000)

    # Simulate a few trade cycles
    for i in range(10):
        meta_agent.execute_trade_cycle(symbol="BTC/USDT")
        time.sleep(0.5)

    # Report status
    meta_agent.report_status()
