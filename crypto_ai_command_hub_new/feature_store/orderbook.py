"""Order book analysis for order flow signals and microstructure alpha.

Analyzes bid/ask dynamics, volume imbalance, and order clustering to
generate alpha signals for entry/exit optimization.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class OrderFlowSignal(Enum):
    """Order flow signal types."""
    STRONG_BUY = 3
    BUY = 2
    NEUTRAL = 1
    SELL = 0
    STRONG_SELL = -1


@dataclass
class OrderBookMetrics:
    """Metrics derived from order book."""
    timestamp: float
    symbol: str
    bid_ask_spread_pct: float
    bid_ask_spread_abs: float
    bid_volume: float
    ask_volume: float
    volume_imbalance: float  # -1 to +1 (positive = buy pressure)
    weighted_mid: float
    vortex_oscillator: float  # -1 to +1
    order_clustering: float  # 0 to 1, higher = more clustered
    large_order_ratio: float  # ratio of large orders
    signal: OrderFlowSignal


class OrderBookAnalyzer:
    """Analyze order book microstructure."""
    
    def __init__(self, lookback_levels: int = 20):
        """
        Args:
            lookback_levels: How many price levels to analyze
        """
        self.lookback_levels = lookback_levels
        self.historical_books = []
        self.signals_generated = 0
    
    def analyze(self, order_book: Dict, symbol: str = "BTC/USDT") -> OrderBookMetrics:
        """Analyze order book and generate signal."""
        
        bids = np.array(order_book.get("bids", []))
        asks = np.array(order_book.get("asks", []))
        
        if len(bids) == 0 or len(asks) == 0:
            return None
        
        # Limit to lookback levels
        bids = bids[:self.lookback_levels]
        asks = asks[:self.lookback_levels]
        
        # Basic metrics
        bid_prices = bids[:, 0]
        bid_volumes = bids[:, 1]
        ask_prices = asks[:, 0]
        ask_volumes = asks[:, 1]
        
        bid_ask_spread_abs = ask_prices[0] - bid_prices[0]
        bid_ask_spread_pct = bid_ask_spread_abs / bid_prices[0] * 100
        
        # Volume metrics
        bid_volume_total = np.sum(bid_volumes)
        ask_volume_total = np.sum(ask_volumes)
        
        volume_imbalance = (bid_volume_total - ask_volume_total) / (bid_volume_total + ask_volume_total)
        
        # Weighted mid price (closer orders weighted more)
        bid_weights = bid_volumes / np.sum(bid_volumes)
        ask_weights = ask_volumes / np.sum(ask_volumes)
        
        weighted_bid = np.sum(bid_prices * bid_weights)
        weighted_ask = np.sum(ask_prices * ask_weights)
        weighted_mid = (weighted_bid + weighted_ask) / 2
        
        # Vortex oscillator (relative strength of bids vs asks)
        bid_strength = np.sum(bid_volumes[:3]) / max(np.sum(bid_volumes), 1e-6)
        ask_strength = np.sum(ask_volumes[:3]) / max(np.sum(ask_volumes), 1e-6)
        vortex = (bid_strength - ask_strength) / (bid_strength + ask_strength)
        
        # Order clustering (concentration at best prices)
        bid_clustering = bid_volumes[0] / np.sum(bid_volumes) if len(bid_volumes) > 0 else 0
        ask_clustering = ask_volumes[0] / np.sum(ask_volumes) if len(ask_volumes) > 0 else 0
        order_clustering = (bid_clustering + ask_clustering) / 2
        
        # Large order ratio
        large_order_threshold = np.percentile(
            np.concatenate([bid_volumes, ask_volumes]), 75
        )
        large_bid_count = np.sum(bid_volumes > large_order_threshold)
        large_ask_count = np.sum(ask_volumes > large_order_threshold)
        large_order_ratio = (large_bid_count + large_ask_count) / (len(bids) + len(asks))
        
        # Generate signal
        signal = self._generate_signal(
            volume_imbalance, vortex, order_clustering, bid_ask_spread_pct
        )
        
        metrics = OrderBookMetrics(
            timestamp=order_book.get("timestamp", 0),
            symbol=symbol,
            bid_ask_spread_pct=bid_ask_spread_pct,
            bid_ask_spread_abs=bid_ask_spread_abs,
            bid_volume=bid_volume_total,
            ask_volume=ask_volume_total,
            volume_imbalance=volume_imbalance,
            weighted_mid=weighted_mid,
            vortex_oscillator=vortex,
            order_clustering=order_clustering,
            large_order_ratio=large_order_ratio,
            signal=signal
        )
        
        self.signals_generated += 1
        self.historical_books.append(metrics)
        
        return metrics
    
    def _generate_signal(self, imbalance: float, vortex: float, 
                        clustering: float, spread_pct: float) -> OrderFlowSignal:
        """Generate signal from microstructure metrics."""
        
        # Weight factors
        imbalance_weight = 0.4
        vortex_weight = 0.3
        clustering_weight = 0.2
        spread_weight = 0.1
        
        # Normalize and score
        imbalance_score = imbalance  # Already -1 to +1
        vortex_score = vortex  # Already -1 to +1
        
        # Clustering: high clustering (>0.4) = concentration = stronger signal
        clustering_score = (clustering - 0.5) * 2  # -1 to +1
        
        # Spread: tight spread = good, but too tight (bot activity) = bad
        if spread_pct < 0.05:
            spread_score = -0.5  # Too tight, maybe bots
        elif spread_pct < 0.1:
            spread_score = 0.2  # Good spread
        elif spread_pct < 0.2:
            spread_score = 0.0
        else:
            spread_score = -0.3  # Wide spread, uncertainty
        
        # Composite score
        composite = (
            imbalance_weight * imbalance_score +
            vortex_weight * vortex_score +
            clustering_weight * clustering_score +
            spread_weight * spread_score
        )
        
        # Map to signal
        if composite > 0.6:
            return OrderFlowSignal.STRONG_BUY
        elif composite > 0.2:
            return OrderFlowSignal.BUY
        elif composite < -0.6:
            return OrderFlowSignal.STRONG_SELL
        elif composite < -0.2:
            return OrderFlowSignal.SELL
        else:
            return OrderFlowSignal.NEUTRAL
    
    def get_cumulative_delta(self) -> float:
        """Get cumulative order flow delta from history."""
        if not self.historical_books:
            return 0.0
        
        return np.mean([m.volume_imbalance for m in self.historical_books[-50:]])
    
    def detect_wall(self, book: Dict, volume_threshold: float = 10.0) -> Dict:
        """Detect large buy/sell walls."""
        
        bids = np.array(book.get("bids", []))
        asks = np.array(book.get("asks", []))
        
        results = {"buy_wall": None, "sell_wall": None}
        
        if len(bids) > 0:
            bid_volumes = bids[:, 1]
            avg_volume = np.mean(bid_volumes[1:])  # Exclude best bid
            
            for i, (price, volume) in enumerate(bids):
                if volume > avg_volume * volume_threshold:
                    results["buy_wall"] = {"price": price, "volume": volume, "level": i}
                    break
        
        if len(asks) > 0:
            ask_volumes = asks[:, 1]
            avg_volume = np.mean(ask_volumes[1:])  # Exclude best ask
            
            for i, (price, volume) in enumerate(asks):
                if volume > avg_volume * volume_threshold:
                    results["sell_wall"] = {"price": price, "volume": volume, "level": i}
                    break
        
        return results


class OrderFlowTrader:
    """Trade based on order flow signals."""
    
    def __init__(self, analyzer: OrderBookAnalyzer):
        self.analyzer = analyzer
        self.recent_signals = []
        self.max_history = 20
    
    def should_enter_long(self, book: Dict) -> Tuple[bool, float]:
        """Determine if should enter long based on order flow."""
        
        metrics = self.analyzer.analyze(book)
        if not metrics:
            return False, 0.0
        
        self.recent_signals.append(metrics.signal.value)
        self.recent_signals = self.recent_signals[-self.max_history:]
        
        # Signal consensus: need multiple strong buy signals
        recent_avg = np.mean(self.recent_signals)
        confidence = abs(recent_avg) / 3.0  # Normalize to 0-1
        
        return metrics.signal in [OrderFlowSignal.BUY, OrderFlowSignal.STRONG_BUY], confidence
    
    def should_exit_long(self, book: Dict) -> Tuple[bool, float]:
        """Determine if should exit long."""
        
        metrics = self.analyzer.analyze(book)
        if not metrics:
            return False, 0.0
        
        # Exit on sell signals
        exit = metrics.signal in [OrderFlowSignal.SELL, OrderFlowSignal.STRONG_SELL]
        confidence = abs(metrics.signal.value) / 3.0
        
        return exit, confidence


if __name__ == "__main__":
    # Demo
    analyzer = OrderBookAnalyzer()
    
    # Simulate order book
    mock_book = {
        "bids": [
            [29990, 5.0],   # Best bid
            [29989, 3.5],
            [29988, 2.1],
            [29987, 4.2],
            [29986, 1.8]
        ],
        "asks": [
            [30010, 2.0],   # Best ask
            [30011, 3.1],
            [30012, 1.5],
            [30013, 2.8],
            [30014, 0.9]
        ],
        "timestamp": 0
    }
    
    metrics = analyzer.analyze(mock_book)
    
    print(f"\nOrder Book Metrics:")
    print(f"  Spread: {metrics.bid_ask_spread_pct:.3f}%")
    print(f"  Bid Volume: {metrics.bid_volume:.2f}")
    print(f"  Ask Volume: {metrics.ask_volume:.2f}")
    print(f"  Imbalance: {metrics.volume_imbalance:.3f}")
    print(f"  Vortex: {metrics.vortex_oscillator:.3f}")
    print(f"  Clustering: {metrics.order_clustering:.3f}")
    print(f"  Signal: {metrics.signal.name} ({metrics.signal.value})")
    
    trader = OrderFlowTrader(analyzer)
    should_buy, confidence = trader.should_enter_long(mock_book)
    print(f"\nTrade Decision:")
    print(f"  Should Enter Long: {should_buy}")
    print(f"  Confidence: {confidence:.2%}")
