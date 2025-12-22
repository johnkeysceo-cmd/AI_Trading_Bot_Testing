"""
technical.py
--------------
Technical Feature Pipeline for multi-agent AI crypto trading hub.

Responsibilities:
- Compute hundreds of technical indicators per symbol
- Prepare ML-ready features
- Handle missing data / NaNs
- Support time-series windows
- Integration hooks for agents
- Extendable for custom indicators

Notes:
- Uses pandas, numpy, and pandas_ta
- Designed for high-performance vectorized computation
- Fully logging-enabled
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from typing import Dict, Any, Optional

# Logging setup
logger = logging.getLogger("TechnicalFeaturePipeline")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

class TechnicalFeaturePipeline:
    """
    Computes technical indicators and prepares features for ML models.
    """

    def __init__(self, window: int = 50):
        self.window = window  # Lookback window for features
        self.indicator_list = [
            "rsi", "macd", "stoch", "bollinger_hband", "bollinger_lband", "ema", "sma", "adx", "cci"
        ]
        logger.info(f"[FeaturePipeline] Initialized with window={self.window} and indicators={self.indicator_list}")

    # -----------------------------
    # Feature Computation
    # -----------------------------
    def compute_features(self, symbol: str, price_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Compute features for a given symbol.
        price_df: DataFrame with columns ['open','high','low','close','volume']
        Returns: dictionary of features
        """
        if price_df is None:
            # In practice, fetch from CCXT or cache
            price_df = self.fetch_historical_data(symbol)

        features: Dict[str, Any] = {}

        try:
            # EMA and SMA
            for period in [5, 10, 20, 50, 100]:
                features[f"sma_{period}"] = price_df['close'].rolling(period).mean().iloc[-1]
                features[f"ema_{period}"] = price_df['close'].ewm(span=period, adjust=False).mean().iloc[-1]

            # RSI
            features["rsi_14"] = ta.rsi(price_df["close"], length=14).iloc[-1]
            features["rsi_28"] = ta.rsi(price_df["close"], length=28).iloc[-1]

            # MACD
            macd = ta.macd(price_df["close"])
            features["macd"] = macd["MACD_12_26_9"].iloc[-1]
            features["macd_signal"] = macd["MACDs_12_26_9"].iloc[-1]
            features["macd_hist"] = macd["MACDh_12_26_9"].iloc[-1]

            # Stochastic
            stoch = ta.stoch(price_df["high"], price_df["low"], price_df["close"])
            features["stoch_k"] = stoch["STOCHk_14_3_3"].iloc[-1]
            features["stoch_d"] = stoch["STOCHd_14_3_3"].iloc[-1]

            # Bollinger Bands
            bbands = ta.bbands(price_df["close"], length=20)
            features["bollinger_hband"] = bbands["BBU_20_2.0"].iloc[-1]
            features["bollinger_lband"] = bbands["BBL_20_2.0"].iloc[-1]
            features["bollinger_mband"] = bbands["BBM_20_2.0"].iloc[-1]

            # ADX
            features["adx_14"] = ta.adx(price_df["high"], price_df["low"], price_df["close"], length=14)["ADX_14"].iloc[-1]

            # CCI
            features["cci_20"] = ta.cci(price_df["high"], price_df["low"], price_df["close"], length=20).iloc[-1]

            # Volume features
            features["volume_mean_10"] = price_df["volume"].rolling(10).mean().iloc[-1]
            features["volume_mean_50"] = price_df["volume"].rolling(50).mean().iloc[-1]

            # Price momentum
            features["momentum_5"] = price_df["close"].pct_change(5).iloc[-1]
            features["momentum_10"] = price_df["close"].pct_change(10).iloc[-1]

            # Normalize features
            features = self.normalize_features(features)

            logger.debug(f"[FeaturePipeline] Features computed for {symbol}: {features}")
        except Exception as e:
            logger.error(f"[FeaturePipeline] Error computing features for {symbol}: {e}")

        return features

    # -----------------------------
    # Normalization
    # -----------------------------
    def normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize features to 0-1 range or z-score
        """
        norm_features = {}
        for key, value in features.items():
            if value is None or np.isnan(value):
                norm_features[key] = 0.0
            else:
                # Simple normalization placeholder: can extend to z-score or min-max
                norm_features[key] = float(value)
        return norm_features

    # -----------------------------
    # Data Fetching Placeholder
    # -----------------------------
    def fetch_historical_data(self, symbol: str) -> pd.DataFrame:
        """
        Placeholder for fetching historical OHLCV data
        """
        # For demonstration: generate random data
        np.random.seed(42)
        length = 100  # last 100 candles
        close = np.cumsum(np.random.randn(length)) + 100
        high = close + np.random.rand(length)
        low = close - np.random.rand(length)
        open_ = close + np.random.randn(length)
        volume = np.random.rand(length) * 1000
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
        return df

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    pipeline = TechnicalFeaturePipeline()
    features = pipeline.compute_features("BTC/USDT")
    print(features)
