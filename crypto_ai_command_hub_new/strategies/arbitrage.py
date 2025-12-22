"""Multi-exchange arbitrage engine for Phase 3.

Detects and executes cross-exchange arbitrage opportunities with
minimal latency. Includes profit calculation, execution routing,
and position unwinding.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging
from abc import ABC, abstractmethod
import asyncio
import time

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity."""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    buy_volume: float
    sell_volume: float
    profit_pct: float
    execution_time_ms: float
    min_position_size: float
    max_position_size: float
    timestamp: float


class ExchangeAdapter(ABC):
    """Abstract adapter for exchange APIs."""
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.last_prices = {}
    
    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 5) -> Dict:
        """Get order book with specified depth."""
        pass
    
    @abstractmethod
    async def execute_buy(self, symbol: str, amount: float, price: Optional[float] = None) -> Dict:
        """Execute market buy order."""
        pass
    
    @abstractmethod
    async def execute_sell(self, symbol: str, amount: float, price: Optional[float] = None) -> Dict:
        """Execute market sell order."""
        pass


class MockExchangeAdapter(ExchangeAdapter):
    """Mock exchange for simulation."""
    
    def __init__(self, exchange_name: str, base_prices: Dict[str, float]):
        super().__init__(exchange_name)
        self.base_prices = base_prices
        self.orders = []
    
    async def get_order_book(self, symbol: str, depth: int = 5) -> Dict:
        """Return simulated order book."""
        base = self.base_prices.get(symbol, 30000)
        
        # Simulate realistic spread
        spread_pct = 0.002 + np.random.uniform(0, 0.002)
        bid_price = base * (1 - spread_pct/2)
        ask_price = base * (1 + spread_pct/2)
        
        bids = []
        asks = []
        
        for i in range(depth):
            bid = bid_price * (1 - i * 0.0001)
            ask = ask_price * (1 + i * 0.0001)
            volume = np.random.uniform(1, 10)
            
            bids.append([bid, volume])
            asks.append([ask, volume])
        
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": time.time()
        }
    
    async def execute_buy(self, symbol: str, amount: float, price: Optional[float] = None) -> Dict:
        """Simulate buy execution."""
        base = self.base_prices.get(symbol, 30000)
        execution_price = base * (1 + 0.001)  # 0.1% slippage
        
        order = {
            "order_id": f"{self.exchange_name}_{len(self.orders)}",
            "symbol": symbol,
            "side": "buy",
            "amount": amount,
            "price": execution_price,
            "cost": amount * execution_price,
            "timestamp": time.time()
        }
        self.orders.append(order)
        return order
    
    async def execute_sell(self, symbol: str, amount: float, price: Optional[float] = None) -> Dict:
        """Simulate sell execution."""
        base = self.base_prices.get(symbol, 30000)
        execution_price = base * (1 - 0.001)  # 0.1% slippage
        
        order = {
            "order_id": f"{self.exchange_name}_{len(self.orders)}",
            "symbol": symbol,
            "side": "sell",
            "amount": amount,
            "price": execution_price,
            "cost": amount * execution_price,
            "timestamp": time.time()
        }
        self.orders.append(order)
        return order


class ArbitrageEngine:
    """Detect and execute cross-exchange arbitrage."""
    
    def __init__(self, exchanges: Dict[str, ExchangeAdapter], 
                 min_profit_pct: float = 0.5, max_fee_pct: float = 0.2):
        """
        Args:
            exchanges: Dict of exchange_name -> ExchangeAdapter
            min_profit_pct: Minimum profit % after fees to execute
            max_fee_pct: Maximum fee % to consider
        """
        self.exchanges = exchanges
        self.min_profit_pct = min_profit_pct
        self.max_fee_pct = max_fee_pct
        
        self.opportunities_detected = 0
        self.opportunities_executed = 0
        self.total_pnl = 0.0
        self.active_positions = {}
    
    async def scan_all_pairs(self, symbols: List[str]) -> List[ArbitrageOpportunity]:
        """Scan all symbol pairs across exchanges."""
        opportunities = []
        
        for symbol in symbols:
            opp = await self.detect_opportunity(symbol)
            if opp:
                opportunities.append(opp)
                self.opportunities_detected += 1
        
        return opportunities
    
    async def detect_opportunity(self, symbol: str) -> Optional[ArbitrageOpportunity]:
        """Detect arbitrage for a single symbol."""
        
        start_time = time.time()
        exchange_names = list(self.exchanges.keys())
        
        # Get order books from all exchanges
        books = {}
        for ex_name in exchange_names:
            try:
                books[ex_name] = await self.exchanges[ex_name].get_order_book(symbol)
            except Exception as e:
                logger.warning(f"Failed to get book from {ex_name}: {e}")
                return None
        
        if len(books) < 2:
            return None
        
        # Check all exchange pairs for arbitrage
        best_opp = None
        
        for i, ex1_name in enumerate(exchange_names):
            for ex2_name in exchange_names[i+1:]:
                book1 = books[ex1_name]
                book2 = books[ex2_name]
                
                # Opportunity 1: Buy on ex1, sell on ex2
                opp1 = self._check_arbitrage(
                    symbol, ex1_name, ex2_name, book1, book2, "BUY_SELL"
                )
                
                # Opportunity 2: Buy on ex2, sell on ex1
                opp2 = self._check_arbitrage(
                    symbol, ex2_name, ex1_name, book2, book1, "SELL_BUY"
                )
                
                for opp in [opp1, opp2]:
                    if opp and (not best_opp or opp.profit_pct > best_opp.profit_pct):
                        best_opp = opp
        
        if best_opp:
            best_opp.execution_time_ms = (time.time() - start_time) * 1000
        
        return best_opp
    
    def _check_arbitrage(self, symbol: str, buy_ex: str, sell_ex: str,
                         buy_book: Dict, sell_book: Dict, 
                         direction: str) -> Optional[ArbitrageOpportunity]:
        """Check arbitrage between two exchanges."""
        
        # Get best bid/ask
        buy_ask = buy_book["asks"][0][0] if buy_book["asks"] else None
        buy_volume = buy_book["asks"][0][1] if buy_book["asks"] else 0
        
        sell_bid = sell_book["bids"][0][0] if sell_book["bids"] else None
        sell_volume = sell_book["bids"][0][1] if sell_book["bids"] else 0
        
        if not buy_ask or not sell_bid:
            return None
        
        # Calculate profit
        gross_profit_pct = (sell_bid - buy_ask) / buy_ask * 100
        
        # Deduct fees (0.1% per side = 0.2% total)
        fees_pct = self.max_fee_pct
        net_profit_pct = gross_profit_pct - fees_pct
        
        # Check if profitable
        if net_profit_pct < self.min_profit_pct:
            return None
        
        # Calculate position sizes
        min_size = max(0.01, buy_ask * 100)  # Min $100
        max_size = min(buy_volume, sell_volume) * 0.5  # Half of available volume
        
        if max_size < min_size:
            return None
        
        return ArbitrageOpportunity(
            symbol=symbol,
            buy_exchange=buy_ex,
            sell_exchange=sell_ex,
            buy_price=buy_ask,
            sell_price=sell_bid,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            profit_pct=net_profit_pct,
            execution_time_ms=0,
            min_position_size=min_size,
            max_position_size=max_size,
            timestamp=time.time()
        )
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity,
                                position_size: float) -> Dict:
        """Execute arbitrage trade."""
        
        if position_size < opportunity.min_position_size:
            logger.warning(f"Position size {position_size} below minimum {opportunity.min_position_size}")
            return {"success": False, "reason": "position_too_small"}
        
        if position_size > opportunity.max_position_size:
            position_size = opportunity.max_position_size
            logger.info(f"Capped position size to {position_size}")
        
        try:
            # Execute buy on first exchange
            buy_order = await self.exchanges[opportunity.buy_exchange].execute_buy(
                opportunity.symbol,
                position_size,
                opportunity.buy_price
            )
            
            # Execute sell on second exchange
            sell_order = await self.exchanges[opportunity.sell_exchange].execute_sell(
                opportunity.symbol,
                position_size,
                opportunity.sell_price
            )
            
            # Calculate PnL
            buy_cost = buy_order["cost"]
            sell_revenue = sell_order["cost"]
            pnl = sell_revenue - buy_cost
            pnl_pct = pnl / buy_cost * 100
            
            # Track position
            trade_id = f"ARB_{time.time()}"
            self.active_positions[trade_id] = {
                "buy_order": buy_order,
                "sell_order": sell_order,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "status": "closed"  # Arbitrage is immediately closed
            }
            
            self.total_pnl += pnl
            self.opportunities_executed += 1
            
            logger.info(f"Arbitrage executed: {opportunity.symbol} "
                       f"buy@{buy_order['price']:.2f} sell@{sell_order['price']:.2f} "
                       f"PnL: ${pnl:.2f} ({pnl_pct:.3f}%)")
            
            return {
                "success": True,
                "trade_id": trade_id,
                "buy_order": buy_order,
                "sell_order": sell_order,
                "pnl": pnl,
                "pnl_pct": pnl_pct
            }
        
        except Exception as e:
            logger.error(f"Arbitrage execution failed: {e}")
            return {"success": False, "reason": str(e)}
    
    def get_statistics(self) -> Dict:
        """Get arbitrage statistics."""
        return {
            "opportunities_detected": self.opportunities_detected,
            "opportunities_executed": self.opportunities_executed,
            "execution_rate": self.opportunities_executed / max(1, self.opportunities_detected),
            "total_pnl": self.total_pnl,
            "avg_pnl_per_trade": self.total_pnl / max(1, self.opportunities_executed)
        }


if __name__ == "__main__":
    async def demo():
        # Create mock exchanges
        exchanges = {
            "binance": MockExchangeAdapter("binance", {"BTC/USDT": 30000, "ETH/USDT": 1800}),
            "kraken": MockExchangeAdapter("kraken", {"BTC/USDT": 30015, "ETH/USDT": 1805}),
            "coinbase": MockExchangeAdapter("coinbase", {"BTC/USDT": 29985, "ETH/USDT": 1795})
        }
        
        # Create arbitrage engine
        arb = ArbitrageEngine(exchanges, min_profit_pct=0.3)
        
        # Scan for opportunities
        opportunities = await arb.scan_all_pairs(["BTC/USDT", "ETH/USDT"])
        
        print(f"\nFound {len(opportunities)} opportunities:")
        for opp in opportunities:
            print(f"  {opp.symbol}: {opp.buy_exchange} → {opp.sell_exchange} "
                  f"+{opp.profit_pct:.2f}% ({opp.execution_time_ms:.1f}ms)")
            
            # Execute best opportunity
            result = await arb.execute_arbitrage(opp, opp.max_position_size * 0.8)
            if result["success"]:
                print(f"    Executed: ${result['pnl']:.2f} profit")
        
        # Print stats
        stats = arb.get_statistics()
        print(f"\nArbitrage Statistics:")
        print(f"  Detected: {stats['opportunities_detected']}")
        print(f"  Executed: {stats['opportunities_executed']}")
        print(f"  Total PnL: ${stats['total_pnl']:.2f}")
    
    asyncio.run(demo())
