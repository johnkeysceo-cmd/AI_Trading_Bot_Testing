# monte_carlo.py
# -----------------
# Simulates random market scenarios to test capital survival.

import random
from agents.meta_agent.agent_selector import agents, CAPITAL_POOL
from agents.external._common import TradeSignal

def monte_carlo_test(iterations: int = 1000):
    results = []
    for i in range(iterations):
        capital = CAPITAL_POOL
        for tier, adapter in agents.items():
            signals = adapter.generate_signals()
            for s in signals:
                # Random PnL simulation
                pnl_factor = random.uniform(-0.1, 0.1)  # +/-10% per trade
                capital += s.amount * pnl_factor
        results.append(capital)
    avg_result = sum(results) / len(results)
    print(f"Monte Carlo: avg final capital = {avg_result:.2f}")
    return results
