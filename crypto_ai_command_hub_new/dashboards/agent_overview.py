def display_agent_status(agent_signals):
    for tier, signals in agent_signals.items():
        print(f"{tier.upper()} agent signals:")
        for s in signals:
            print(f"  {s}")
