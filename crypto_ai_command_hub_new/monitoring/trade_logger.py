import logging

logger = logging.getLogger("TradeLogger")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

def log_trade(signal):
    logger.info(f"Executed trade: {signal}")
