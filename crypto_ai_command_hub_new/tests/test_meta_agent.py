# tests/test_meta_agent.py

from agents.meta_agent.agent_selector import MetaAgent

def run_test():
    meta = MetaAgent()
    result = meta.route_signal("freqtrade", market_data=None)
    print(result)

if __name__ == "__main__":
    run_test()
