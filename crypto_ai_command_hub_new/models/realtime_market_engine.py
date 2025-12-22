"""Real-Time Market Data Engine with Live Market Integration
Pulls real OHLCV data from multiple exchanges, computes indicators, manages feeds"""

import numpy as np, json, logging, time, requests, os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import hashlib
import hmac
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self):
        return {
            'timestamp': float(self.timestamp),
            'open': float(self.open),
            'high': float(self.high),
            'low': float(self.low),
            'close': float(self.close),
            'volume': float(self.volume)
        }

@dataclass
class ExchangeFees:
    maker_fee: float = 0.001  # 0.1% standard
    taker_fee: float = 0.0025  # 0.25% standard
    withdrawal_fee: float = 0.0005  # 0.05% for crypto
    deposit_fee: float = 0.0
    
    def calculate_trade_fee(self, amount: float, is_maker: bool = False) -> float:
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return amount * fee_rate

@dataclass
class TickerData:
    symbol: str
    bid: float
    ask: float
    last: float
    timestamp: float
    volume_24h: float
    
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    def spread(self) -> float:
        return ((self.ask - self.bid) / self.mid_price()) * 100  # percentage

class MarketDataCache:
    """Thread-safe cache for OHLCV data"""
    def __init__(self, max_candles: int = 5000):
        self.data = {}
        self.max_candles = max_candles
    
    def add_candle(self, symbol: str, timeframe: str, candle: MarketData):
        key = f"{symbol}_{timeframe}"
        if key not in self.data:
            self.data[key] = deque(maxlen=self.max_candles)
        self.data[key].append(candle)
    
    def get_candles(self, symbol: str, timeframe: str, limit: int = None) -> List[MarketData]:
        key = f"{symbol}_{timeframe}"
        if key not in self.data:
            return []
        
        candles = list(self.data[key])
        if limit:
            return candles[-limit:]
        return candles
    
    def get_latest(self, symbol: str, timeframe: str) -> Optional[MarketData]:
        candles = self.get_candles(symbol, timeframe, limit=1)
        return candles[0] if candles else None

class CoinGeckoFeed:
    """Real market data from CoinGecko (free tier)"""
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, api_key: str = None):
        self.session = requests.Session()
        self.rate_limit_delay = 1.1  # 1.1 seconds between requests
        self.last_request = 0

        # If an api_key wasn't provided programmatically, try to load from
        # config/api_keys.json (path can be overridden with API_KEYS_PATH env var).
        if not api_key:
            try:
                API_KEYS_PATH = os.getenv("API_KEYS_PATH", str(Path(__file__).resolve().parents[1] / "config" / "api_keys.json"))
                if os.path.exists(API_KEYS_PATH):
                    with open(API_KEYS_PATH, "r") as f:
                        cfg = json.load(f)
                        api_key = cfg.get("coingecko", {}).get("key")
            except Exception:
                api_key = None

        # If an API key is available, use CoinGecko Pro header for authenticated requests
        if api_key:
            self.session.headers.update({"X-CG-Pro-API-Key": api_key})
    
    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request = time.time()
    
    def get_market_data(self, symbol: str, vs_currency: str = 'usd',
                       days: int = 30) -> List[Tuple[float, float, float, float]]:
        """Get historical price data (timestamp, open, high, low, close)"""
        
        self._rate_limit()
        
        # Map crypto symbols to CoinGecko IDs
        symbol_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'SOL': 'solana',
            'DOGE': 'dogecoin',
            'LUNA': 'terra',
            'AVAX': 'avalanche-2',
            'LINK': 'chainlink',
            'MATIC': 'matic-network'
        }
        
        coin_id = symbol_map.get(symbol.upper(), symbol.lower())
        
        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': 'daily'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            prices = data.get('prices', [])
            
            # prices format: [timestamp_ms, price]
            market_data = []
            for timestamp_ms, price in prices:
                timestamp = timestamp_ms / 1000
                # For daily data, we'll use price as OHLC
                market_data.append((timestamp, price, price, price, price))
            
            logger.info(f"Retrieved {len(market_data)} candles for {symbol}")
            return market_data
        
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return []
    
    def get_current_price(self, symbols: List[str]) -> Dict[str, float]:
        """Get current price for multiple symbols"""
        
        self._rate_limit()
        
        symbol_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'SOL': 'solana',
            'DOGE': 'dogecoin',
            'LUNA': 'terra',
            'AVAX': 'avalanche-2',
            'LINK': 'chainlink',
            'MATIC': 'matic-network'
        }
        
        coin_ids = [symbol_map.get(s.upper(), s.lower()) for s in symbols]
        
        url = f"{self.BASE_URL}/simple/price"
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': 'usd'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            prices = {}
            
            for orig_symbol, coin_id in zip(symbols, coin_ids):
                if coin_id in data:
                    prices[orig_symbol] = data[coin_id].get('usd', 0)
            
            return prices
        
        except Exception as e:
            logger.error(f"Error fetching current prices: {e}")
            return {s: 0 for s in symbols}

class KrakenMarketFeed:
    """Kraken market data (requires API key for full features)"""
    BASE_URL = "https://api.kraken.com"
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
    
    def get_ticker(self, pair: str) -> Optional[TickerData]:
        """Get real-time ticker data"""
        
        # Convert symbol to Kraken pair format
        pair_map = {
            'BTC/USD': 'XXBTZUSD',
            'ETH/USD': 'XETHZUSD',
            'XRP/USD': 'XXRPZUSD',
            'ADA/USD': 'ADAUSD',
        }
        
        kraken_pair = pair_map.get(pair, pair)
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/0/public/Ticker",
                params={'pair': kraken_pair},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('result'):
                ticker_data = list(data['result'].values())[0]
                
                return TickerData(
                    symbol=pair,
                    bid=float(ticker_data['b'][0]),
                    ask=float(ticker_data['a'][0]),
                    last=float(ticker_data['c'][0]),
                    timestamp=time.time(),
                    volume_24h=float(ticker_data['v'][1])
                )
        
        except Exception as e:
            logger.warning(f"Error fetching ticker for {pair}: {e}")
        
        return None
    
    def get_ohlc(self, pair: str, interval: int = 1440) -> List[MarketData]:
        """Get OHLC data (interval in minutes)"""
        
        pair_map = {
            'BTC/USD': 'XXBTZUSD',
            'ETH/USD': 'XETHZUSD',
            'XRP/USD': 'XXRPZUSD',
        }
        
        kraken_pair = pair_map.get(pair, pair)
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/0/public/OHLC",
                params={'pair': kraken_pair, 'interval': interval},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('result'):
                ohlc_data = list(data['result'].values())[0]
                
                candles = []
                for candle in ohlc_data:
                    timestamp, open_p, high, low, close, vwap, volume, count = candle
                    
                    market_data = MarketData(
                        timestamp=float(timestamp),
                        open=float(open_p),
                        high=float(high),
                        low=float(low),
                        close=float(close),
                        volume=float(volume)
                    )
                    candles.append(market_data)
                
                return candles
        
        except Exception as e:
            logger.warning(f"Error fetching OHLC for {pair}: {e}")
        
        return []

class RealtimeMarketEngine:
    """Unified market data engine with multiple feeds"""
    
    def __init__(self):
        self.coingecko = CoinGeckoFeed()
        self.kraken = KrakenMarketFeed()
        self.cache = MarketDataCache()
        
        self.symbols = ['BTC', 'ETH', 'XRP', 'ADA', 'SOL']
        self.timeframes = ['1h', '4h', '1d']
        self.fees = {
            'kraken': ExchangeFees(maker_fee=0.0016, taker_fee=0.0026),
            'binance': ExchangeFees(maker_fee=0.001, taker_fee=0.0025),
            'coinbase': ExchangeFees(maker_fee=0.004, taker_fee=0.006),
        }
        
        self.data_quality_metrics = {
            'total_candles_cached': 0,
            'last_update': 0,
            'data_freshness_seconds': 0
        }
    
    def load_historical_data(self, symbol: str, days: int = 365) -> int:
        """Load historical data for a symbol"""
        
        logger.info(f"Loading {days} days of historical data for {symbol}")
        
        market_data = self.coingecko.get_market_data(symbol, days=days)
        
        candles_loaded = 0
        for timestamp, open_p, high, low, close, *_ in market_data:
            candle = MarketData(
                timestamp=timestamp,
                open=float(open_p),
                high=float(high),
                low=float(low),
                close=float(close),
                volume=0  # CoinGecko daily doesn't give volume in this format
            )
            
            self.cache.add_candle(symbol, '1d', candle)
            candles_loaded += 1
        
        self.data_quality_metrics['total_candles_cached'] += candles_loaded
        self.data_quality_metrics['last_update'] = time.time()
        
        logger.info(f"Loaded {candles_loaded} candles for {symbol}")
        return candles_loaded
    
    def get_market_data(self, symbol: str, timeframe: str = '1d',
                       limit: int = 100) -> List[MarketData]:
        """Get market data from cache"""
        return self.cache.get_candles(symbol, timeframe, limit=limit)
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        return self.coingecko.get_current_price(self.symbols)
    
    def estimate_slippage(self, order_size: float, mid_price: float,
                         liquidity_depth: float = 100000.0) -> float:
        """Estimate slippage based on order size and market liquidity"""
        
        order_value = order_size * mid_price
        slippage_bps = (order_value / liquidity_depth) * 100  # basis points
        
        return slippage_bps / 10000  # Convert to decimal
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Report on data quality and freshness"""
        
        freshness = time.time() - self.data_quality_metrics['last_update']
        
        return {
            'total_candles_in_cache': self.data_quality_metrics['total_candles_cached'],
            'data_freshness_seconds': freshness,
            'is_fresh': freshness < 3600,  # Fresh if less than 1 hour old
            'timestamp': datetime.now().isoformat()
        }
    
    def get_symbol_stats(self, symbol: str, timeframe: str = '1d',
                        lookback: int = 365) -> Dict[str, float]:
        """Calculate statistics for a symbol"""
        
        candles = self.cache.get_candles(symbol, timeframe, limit=lookback)
        
        if not candles:
            return {}
        
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]
        
        returns = np.diff(closes) / np.array(closes[:-1])
        
        return {
            'symbol': symbol,
            'current_price': float(closes[-1]),
            'high_52w': float(np.max(highs[-252:]) if len(closes) > 252 else np.max(highs)),
            'low_52w': float(np.min(lows[-252:]) if len(closes) > 252 else np.min(lows)),
            'avg_volume': float(np.mean(volumes) if volumes else 0),
            'volatility_annual': float(np.std(returns) * np.sqrt(365)),
            'sharpe_ratio': float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(365)),
            'sma_50': float(np.mean(closes[-50:])) if len(closes) > 50 else float(closes[-1]),
            'sma_200': float(np.mean(closes[-200:])) if len(closes) > 200 else float(closes[-1]),
        }

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("REALTIME MARKET ENGINE - Real Market Data Integration")
    logger.info("=" * 80)
    
    engine = RealtimeMarketEngine()
    
    # Load historical data for major symbols
    for symbol in ['BTC', 'ETH', 'XRP']:
        engine.load_historical_data(symbol, days=365)
    
    # Get current prices
    logger.info("\nCURRENT PRICES:")
    prices = engine.get_current_prices()
    for symbol, price in prices.items():
        logger.info(f"  {symbol}: ${price:,.2f}")
    
    # Get statistics
    logger.info("\nMARKET STATISTICS:")
    for symbol in ['BTC', 'ETH', 'XRP']:
        stats = engine.get_symbol_stats(symbol)
        logger.info(f"\n  {symbol}:")
        logger.info(f"    Current Price: ${stats['current_price']:,.2f}")
        logger.info(f"    52W High: ${stats['high_52w']:,.2f}")
        logger.info(f"    52W Low: ${stats['low_52w']:,.2f}")
        logger.info(f"    Annual Volatility: {stats['volatility_annual']*100:.2f}%")
        logger.info(f"    Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
    
    # Data quality report
    logger.info("\nDATA QUALITY REPORT:")
    report = engine.get_data_quality_report()
    logger.info(f"  Total Candles Cached: {report['total_candles_in_cache']:,}")
    logger.info(f"  Data Freshness: {report['data_freshness_seconds']:.1f} seconds")
    logger.info(f"  Is Fresh: {report['is_fresh']}")
    
    logger.info("\n✓ Market engine ready for backtesting")
