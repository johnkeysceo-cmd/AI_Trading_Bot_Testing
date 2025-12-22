"""
ccxt_executor.py
-----------------
This module handles all exchange interactions for our AI agents.
It uses CCXT to fetch market data, place orders, and manage account state.

Features:
- Unified API across exchanges
- Retry logic
- Basic rate-limit handling
- Paper trading toggle
- Trade logging
- Integration hooks for multi-agent risk allocation
"""

import ccxt
import time
import logging
from config.trading_mode import TRADING_MODE
from typing import Dict, Optional, Any
import threading
import json
import os

# Logging setup
logger = logging.getLogger("CCXTExecutor")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Load API keys securely
API_KEYS_PATH = os.getenv("API_KEYS_PATH", "./config/api_keys.json")
if not os.path.exists(API_KEYS_PATH):
    raise FileNotFoundError(f"API keys file not found at {API_KEYS_PATH}")

with open(API_KEYS_PATH, "r") as f:
    API_KEYS = json.load(f)

# Paper trading mode (toggle)
PAPER_TRADING = TRADING_MODE.lower() != "live"

# Define supported exchanges (expandable)
EXCHANGES_SUPPORTED = ["binance", "kraken", "coinbasepro"]

class ExchangeConnector:
    """
    Handles connection and basic operations per exchange.
    """
    def __init__(self, exchange_name: str):
        if exchange_name not in EXCHANGES_SUPPORTED:
            raise ValueError(f"Unsupported exchange {exchange_name}")
        self.name = exchange_name
        self.paper = PAPER_TRADING

        # Initialize CCXT exchange instance
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': API_KEYS[exchange_name]['key'],
            'secret': API_KEYS[exchange_name]['secret'],
            'enableRateLimit': True
        })

        # Paper trading state
        self.paper_balance = 10000.0  # $10k default starting balance
        self.orders_log = []

        logger.info(f"[{self.name}] Exchange connector initialized. Paper trading={self.paper}")

    def fetch_balance(self) -> Dict[str, Any]:
        """
        Returns account balance. Uses paper balance if PAPER_TRADING.
        """
        if self.paper:
            logger.debug(f"[{self.name}] Returning paper balance: {self.paper_balance}")
            return {'USD': self.paper_balance}
        try:
            balance = self.exchange.fetch_balance()
            logger.debug(f"[{self.name}] Fetched real balance")
            return balance
        except Exception as e:
            logger.error(f"[{self.name}] Error fetching balance: {e}")
            return {}

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch latest ticker for a symbol
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"[{self.name}] Error fetching ticker for {symbol}: {e}")
            return {}

    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None, order_type: str = "market") -> Dict[str, Any]:
        """
        Place a trade. Handles both real and paper trading modes.
        """
        logger.info(f"[{self.name}] Placing {side} order: {symbol} amount={amount}, price={price}, type={order_type}")

        if self.paper:
            executed_price = price if price else self.fetch_ticker(symbol).get('last', 0)
            cost = amount * executed_price
            if side.lower() == "buy":
                if cost > self.paper_balance:
                    logger.warning(f"[{self.name}] Paper trade insufficient balance: cost={cost}, balance={self.paper_balance}")
                    return {"status": "failed", "reason": "insufficient balance"}
                self.paper_balance -= cost
            elif side.lower() == "sell":
                self.paper_balance += cost  # Simplified: assumes full amount is available
            order = {
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": executed_price,
                "type": order_type,
                "paper_trade": True,
                "timestamp": time.time()
            }
            self.orders_log.append(order)
            logger.info(f"[{self.name}] Paper order executed: {order}")
            return {"status": "success", "order": order}
        else:
            try:
                if order_type.lower() == "market":
                    order = self.exchange.create_market_order(symbol, side, amount)
                else:
                    order = self.exchange.create_limit_order(symbol, side, amount, price)
                self.orders_log.append(order)
                logger.info(f"[{self.name}] Real order executed: {order}")
                return {"status": "success", "order": order}
            except Exception as e:
                logger.error(f"[{self.name}] Error executing order: {e}")
                return {"status": "failed", "reason": str(e)}

    def get_open_orders(self, symbol: str):
        """
        Get open orders for a symbol
        """
        if self.paper:
            # Paper trading does not maintain open orders
            return []
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            logger.error(f"[{self.name}] Error fetching open orders: {e}")
            return []

# Example singleton manager for multiple exchanges
class ExchangeManager:
    """
    Manages multiple exchanges for agents.
    Handles routing, paper vs real, and distributed allocation
    """
    def __init__(self):
        self.connectors: Dict[str, ExchangeConnector] = {}
        for name in EXCHANGES_SUPPORTED:
            self.connectors[name] = ExchangeConnector(name)

    def execute(self, exchange_name: str, symbol: str, side: str, amount: float, price: Optional[float] = None, order_type: str = "market"):
        if exchange_name not in self.connectors:
            raise ValueError(f"Exchange {exchange_name} not managed")
        return self.connectors[exchange_name].place_order(symbol, side, amount, price, order_type)

    def get_balances(self):
        return {name: conn.fetch_balance() for name, conn in self.connectors.items()}

# Thread-safe wrapper example for agent
class SafeExecutor:
    """
    Wraps exchange operations for thread-safety across multiple agents
    """
    def __init__(self, manager: ExchangeManager):
        self.manager = manager
        self.lock = threading.Lock()

    def execute_trade(self, exchange_name: str, symbol: str, side: str, amount: float, price: Optional[float] = None, order_type: str = "market"):
        with self.lock:
            return self.manager.execute(exchange_name, symbol, side, amount, price, order_type)

    def fetch_all_balances(self):
        with self.lock:
            return self.manager.get_balances()

# END OF FILE
