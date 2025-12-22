"""Data connector for loading historical OHLCV data.

This module provides utilities to load, cache, and preprocess historical
price data from CCXT exchanges (Binance, Kraken, etc.) or local CSV files.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging
import json

logger = logging.getLogger(__name__)


class DataConnector:
    """Load and preprocess historical OHLCV data."""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_ohlcv(self, symbol: str, timeframe: str = "1h", 
                   start_date: str = "2023-01-01", 
                   end_date: str = "2024-01-01") -> np.ndarray:
        """Load OHLCV data from cache or generate synthetic data.
        
        Args:
            symbol: e.g., "BTC/USDT"
            timeframe: "1m", "5m", "1h", "4h", "1d"
            start_date: ISO format
            end_date: ISO format
        
        Returns:
            ndarray of shape (timesteps, 5+) with [open, high, low, close, volume, ...]
        """
        cache_file = self.cache_dir / f"{symbol.replace('/', '_')}_{timeframe}.npy"
        
        if cache_file.exists():
            logger.info(f"Loaded {symbol} {timeframe} from cache")
            return np.load(cache_file)
        
        # Generate synthetic data (production: call CCXT)
        data = self._generate_synthetic_ohlcv(symbol, timeframe, start_date, end_date)
        np.save(cache_file, data)
        logger.info(f"Generated synthetic data for {symbol} {timeframe}")
        
        return data
    
    def _generate_synthetic_ohlcv(self, symbol: str, timeframe: str,
                                   start_date: str, end_date: str) -> np.ndarray:
        """Generate realistic synthetic OHLCV data using random walk."""
        # Determine number of periods
        timeframe_minutes = {"1m": 1, "5m": 5, "1h": 60, "4h": 240, "1d": 1440}
        minutes = timeframe_minutes.get(timeframe, 60)
        
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        periods = int((end - start).total_seconds() / (minutes * 60))
        
        # Realistic parameters per asset
        params = {
            "BTC": {"start": 30000, "mu": 0.0001, "sigma": 0.015},
            "ETH": {"start": 1800, "mu": 0.00015, "sigma": 0.018},
            "XRP": {"start": 0.5, "mu": 0.00005, "sigma": 0.025}
        }
        base = symbol.split("/")[0]
        config = params.get(base, {"start": 100, "mu": 0.0001, "sigma": 0.02})
        
        # Generate price series
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(config["mu"], config["sigma"], periods)
        prices = config["start"] * np.cumprod(1 + returns)
        
        # Create OHLCV (simplified)
        opens = prices
        highs = prices * (1 + np.abs(np.random.normal(0, 0.005, periods)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.005, periods)))
        closes = prices
        volumes = np.random.uniform(100, 1000, periods) * 1e6
        
        ohlcv = np.column_stack([opens, highs, lows, closes, volumes])
        return ohlcv.astype(np.float32)
    
    def preprocess_for_training(self, ohlcv: np.ndarray, 
                                 lookback: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare OHLCV data for neural network training.
        
        Returns:
            (X, y) where X is (samples, lookback, features) and y is returns
        """
        closes = ohlcv[:, 3]
        volumes = ohlcv[:, 4]
        
        # Calculate features
        returns = np.diff(closes) / closes[:-1]
        volatility = np.array([np.std(returns[max(0, i-20):i+1]) 
                               for i in range(len(returns))])
        
        # Normalize
        returns_norm = (returns - np.mean(returns)) / (np.std(returns) + 1e-6)
        volatility_norm = (volatility - np.mean(volatility)) / (np.std(volatility) + 1e-6)
        volumes_norm = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-6)
        
        # Stack features: [return, volatility, volume]
        features = np.column_stack([returns_norm[:-1], volatility_norm[:-1], volumes_norm[:-1]])
        
        # Create sequences
        X = []
        y = []
        for i in range(len(features) - lookback):
            X.append(features[i:i+lookback])
            y.append(1 if returns[i+lookback] > 0 else 0)  # Binary classification
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def split_train_test(self, data: np.ndarray, 
                        train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into train/test by time."""
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]


if __name__ == "__main__":
    dc = DataConnector()
    
    # Load BTC data
    ohlcv = dc.load_ohlcv("BTC/USDT", timeframe="1h", 
                          start_date="2023-06-01", end_date="2023-12-31")
    
    print(f"Loaded {len(ohlcv)} bars")
    print(f"Price range: ${ohlcv[:, 3].min():.0f} - ${ohlcv[:, 3].max():.0f}")
    
    # Preprocess
    X, y = dc.preprocess_for_training(ohlcv, lookback=50)
    print(f"Preprocessed to X: {X.shape}, y: {y.shape}")
    print(f"Class distribution: {np.mean(y):.1%} bullish signals")
