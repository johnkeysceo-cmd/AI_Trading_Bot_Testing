# stress_tests.py
# -----------------
# Simulates extreme market events: crashes, spikes, sudden volatility.

from agents.meta_agent.agent_selector import agents, CAPITAL_POOL
from agents.external._common import TradeSignal

def stress_test():
    scenarios = [
        {"desc": "Flash crash", "factor": -0.5},
        {"desc": "Pump spike", "factor": 0.5},
        {"desc": "High volatility", "factor": 0.2}
    ]
    for s in scenarios:
        capital = CAPITAL_POOL
        print(f"Running scenario: {s['desc']}")
        for tier, adapter in agents.items():
            signals = adapter.generate_signals()
            for sig in signals:
                capital += sig.amount * s["factor"]
        print(f"End capital after {s['desc']}: {capital:.2f}")
