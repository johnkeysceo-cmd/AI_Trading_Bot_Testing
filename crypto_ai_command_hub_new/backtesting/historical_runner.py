"""
historical_runner.py
--------------------
Run historical backtests for all agents.
"""

from agents.meta_agent.agent_selector import agents
from agents.external._common import TradeSignal, SignalAggregator
import logging

logger = logging.getLogger("HistoricalRunner")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

def run_historical_backtest(price_series: list):
    aggregator = SignalAggregator()
    for price in price_series:
        market_data = {"BTC/USDT": price}  # extendable for multi-symbol
        for name, agent in agents.items():
            try:
                signals = agent.generate_signals(market_data)
                for s in signals:
                    aggregator.add_signal(s)
            except Exception as e:
                logger.error(f"Agent {name} failed during backtest: {e}")

    valid_signals = aggregator.filter_by_confidence(threshold=0.3)
    logger.info(f"Backtest completed, total valid signals: {len(valid_signals)}")
    return valid_signals
