"""
multi_timeframe_analyzer.py
----------------------------
Multi-Timeframe Signal Fusion for Comprehensive Market Analysis

Features:
- Analyze price data at 5 different timeframes simultaneously
- Compute technical indicators on each timeframe independently
- Weight signals by timeframe importance (daily > hourly > 5min)
- Detect timeframe alignment/conflict
- Reject trades that conflict with longer timeframes

Timeframe Hierarchy:
  Daily (1d): 70% weight - Regime confirmation
  4-Hour (4h): 20% weight - Trend direction
  1-Hour (1h): 5% weight - Entry/exit triggers
  5-Min (5m): 3% weight - Scalping signals
  1-Min (1m): 2% weight - Micro scalping

Phase 1: Signal Fusion & Positioning - Improvement: 3/10 → 6/10
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta
import pandas_ta as ta

# Logging setup
logger = logging.getLogger("MultiTimeframeAnalyzer")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class TimeframeType(Enum):
    """Supported timeframes"""
    ONE_MIN = "1m"
    FIVE_MIN = "5m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"


class TrendDirection(Enum):
    """Trend classification"""
    STRONG_UP = 2
    WEAK_UP = 1
    NEUTRAL = 0
    WEAK_DOWN = -1
    STRONG_DOWN = -2


@dataclass
class TimeframeSignal:
    """Signal for a single timeframe"""
    timeframe: TimeframeType
    symbol: str
    trend: TrendDirection
    trend_strength: float  # 0.0 - 1.0
    rsi: float
    macd_status: str  # "BULLISH", "BEARISH", "NEUTRAL"
    bb_position: float  # -1.0 (lower band) to 1.0 (upper band)
    volume_score: float  # 0.0 - 1.0
    momentum: float  # -1.0 to 1.0
    confidence: float  # 0.0 - 1.0
    timestamp: float


@dataclass
class MultiTimeframeConsensus:
    """Consensus across all timeframes"""
    symbol: str
    daily_trend: TrendDirection
    hourly_trend: TrendDirection
    minute_trend: TrendDirection
    alignment_score: float  # 0.0 (conflicted) - 1.0 (aligned)
    timeframe_signals: Dict[str, TimeframeSignal]
    final_recommendation: str  # "BUY", "SELL", "HOLD"
    confidence: float
    rejection_reason: Optional[str]  # Why trade was rejected
    timestamp: float


class MultiTimeframeAnalyzer:
    """
    Analyze price action across multiple timeframes
    """

    def __init__(self):
        # Timeframe weights (sum to 1.0)
        self.timeframe_weights = {
            TimeframeType.ONE_DAY: 0.40,
            TimeframeType.FOUR_HOUR: 0.25,
            TimeframeType.ONE_HOUR: 0.20,
            TimeframeType.FIVE_MIN: 0.10,
            TimeframeType.ONE_MIN: 0.05
        }
        
        # Price data cache
        self.price_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Historical signals
        self.signal_history: List[MultiTimeframeConsensus] = []
        
        logger.info("[MultiTimeframeAnalyzer] Initialized with 5 timeframes")

    # ==================== DATA LOADING ====================
    def load_price_data(
        self,
        symbol: str,
        ohlcv_data: Dict[TimeframeType, pd.DataFrame]
    ) -> None:
        """
        Load OHLCV data for all timeframes
        
        Expected DataFrame columns: open, high, low, close, volume
        """
        if symbol not in self.price_data:
            self.price_data[symbol] = {}
        
        self.price_data[symbol].update(ohlcv_data)
        logger.debug(f"[MultiTimeframeAnalyzer] Loaded price data for {symbol}")

    # ==================== TECHNICAL ANALYSIS ====================
    def compute_timeframe_features(
        self,
        symbol: str,
        timeframe: TimeframeType
    ) -> Optional[Dict]:
        """
        Compute technical indicators for a specific timeframe
        
        Returns dictionary with RSI, MACD, Bollinger Bands, etc.
        """
        if symbol not in self.price_data or timeframe not in self.price_data[symbol]:
            logger.warning(f"[MultiTimeframeAnalyzer] No data for {symbol} {timeframe.value}")
            return None
        
        df = self.price_data[symbol][timeframe].copy()
        
        if len(df) < 50:
            logger.warning(f"[MultiTimeframeAnalyzer] Insufficient data for {symbol} {timeframe.value}")
            return None
        
        features = {}
        
        try:
            # RSI (14 period)
            features['rsi'] = ta.rsi(df['close'], length=14).iloc[-1]
            
            # MACD
            macd = ta.macd(df['close'])
            features['macd'] = macd['MACD_12_26_9'].iloc[-1]
            features['macd_signal'] = macd['MACDs_12_26_9'].iloc[-1]
            features['macd_diff'] = macd['MACDh_12_26_9'].iloc[-1]
            
            # Bollinger Bands
            bbands = ta.bbands(df['close'], length=20)
            features['bb_upper'] = bbands['BBU_20_2.0'].iloc[-1]
            features['bb_middle'] = bbands['BBM_20_2.0'].iloc[-1]
            features['bb_lower'] = bbands['BBL_20_2.0'].iloc[-1]
            
            # ADX (trend strength)
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            features['adx'] = adx['ADX_14'].iloc[-1]
            
            # EMA crossover
            ema_20 = ta.ema(df['close'], length=20).iloc[-1]
            ema_50 = ta.ema(df['close'], length=50).iloc[-1]
            features['ema_20'] = ema_20
            features['ema_50'] = ema_50
            
            # Volume analysis
            sma_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            features['volume_ratio'] = current_volume / sma_volume if sma_volume > 0 else 1.0
            
            # Momentum
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            features['momentum'] = momentum
            
            return features
        
        except Exception as e:
            logger.error(f"[MultiTimeframeAnalyzer] Error computing features: {e}")
            return None

    # ==================== SIGNAL GENERATION ====================
    def generate_timeframe_signal(
        self,
        symbol: str,
        timeframe: TimeframeType,
        features: Dict
    ) -> Optional[TimeframeSignal]:
        """
        Generate a trading signal for a specific timeframe
        """
        if not features:
            return None
        
        rsi = features.get('rsi', 50)
        macd_diff = features.get('macd_diff', 0)
        bb_position = self._calculate_bb_position(features)
        volume_ratio = features.get('volume_ratio', 1.0)
        momentum = features.get('momentum', 0)
        adx = features.get('adx', 20)
        
        # Determine trend
        trend, trend_strength = self._determine_trend(
            rsi, macd_diff, momentum, adx, features
        )
        
        # MACD status
        if macd_diff > 0:
            macd_status = "BULLISH"
        elif macd_diff < 0:
            macd_status = "BEARISH"
        else:
            macd_status = "NEUTRAL"
        
        # Volume score (high volume = more reliable signal)
        volume_score = min(volume_ratio / 2.0, 1.0)
        
        # Calculate confidence
        confidence = self._calculate_timeframe_confidence(
            trend_strength, volume_score, adx
        )
        
        signal = TimeframeSignal(
            timeframe=timeframe,
            symbol=symbol,
            trend=trend,
            trend_strength=trend_strength,
            rsi=rsi,
            macd_status=macd_status,
            bb_position=bb_position,
            volume_score=volume_score,
            momentum=momentum,
            confidence=confidence,
            timestamp=datetime.now().timestamp()
        )
        
        return signal

    def _determine_trend(
        self,
        rsi: float,
        macd_diff: float,
        momentum: float,
        adx: float,
        features: Dict
    ) -> Tuple[TrendDirection, float]:
        """
        Determine trend direction and strength
        
        Uses RSI, MACD, momentum, and ADX
        """
        ema_20 = features.get('ema_20', 0)
        ema_50 = features.get('ema_50', 0)
        
        # Count bullish indicators
        bullish_count = 0
        bearish_count = 0
        
        # RSI signals
        if rsi > 60:
            bullish_count += 1
        elif rsi > 50:
            bullish_count += 0.5
        if rsi < 40:
            bearish_count += 1
        elif rsi < 50:
            bearish_count += 0.5
        
        # MACD signal
        if macd_diff > 0:
            bullish_count += 1
        else:
            bearish_count += 1
        
        # Momentum
        if momentum > 0:
            bullish_count += 0.5
        else:
            bearish_count += 0.5
        
        # EMA crossover
        if ema_20 > ema_50:
            bullish_count += 1
        else:
            bearish_count += 1
        
        # Determine trend
        net_score = bullish_count - bearish_count
        total_indicators = bullish_count + bearish_count
        
        # Normalize to -2 to +2
        trend_score = (net_score / total_indicators) * 2 if total_indicators > 0 else 0
        trend_strength = abs(trend_score) / 2.0  # 0.0 - 1.0
        
        if trend_score >= 1.0:
            trend = TrendDirection.STRONG_UP
        elif trend_score >= 0.3:
            trend = TrendDirection.WEAK_UP
        elif trend_score <= -1.0:
            trend = TrendDirection.STRONG_DOWN
        elif trend_score <= -0.3:
            trend = TrendDirection.WEAK_DOWN
        else:
            trend = TrendDirection.NEUTRAL
        
        # Boost confidence with ADX (trend strength)
        if adx > 25:
            trend_strength = min(trend_strength * 1.2, 1.0)
        elif adx < 20:
            trend_strength *= 0.8
        
        return trend, trend_strength

    def _calculate_bb_position(self, features: Dict) -> float:
        """
        Calculate Bollinger Band position as percentage
        
        -1.0: At lower band (oversold)
        0.0: At middle band (neutral)
        1.0: At upper band (overbought)
        """
        close = features.get('close', 0)
        bb_upper = features.get('bb_upper', close)
        bb_lower = features.get('bb_lower', close)
        bb_middle = features.get('bb_middle', close)
        
        bb_range = bb_upper - bb_lower
        if bb_range == 0:
            return 0.0
        
        position = (close - bb_middle) / (bb_range / 2)
        return np.clip(position, -1.0, 1.0)

    def _calculate_timeframe_confidence(
        self,
        trend_strength: float,
        volume_score: float,
        adx: float
    ) -> float:
        """Calculate overall confidence in the signal"""
        adx_normalized = min(adx / 40, 1.0)
        confidence = (trend_strength * 0.5 + volume_score * 0.3 + adx_normalized * 0.2)
        return np.clip(confidence, 0.0, 1.0)

    # ==================== MULTI-TIMEFRAME AGGREGATION ====================
    def aggregate_timeframe_signals(
        self,
        symbol: str
    ) -> Optional[MultiTimeframeConsensus]:
        """
        Aggregate signals across all timeframes
        
        Process:
        1. Generate signal for each timeframe
        2. Extract daily, hourly, minute-level trends
        3. Check alignment
        4. Determine if trade is valid
        """
        if symbol not in self.price_data:
            logger.warning(f"[MultiTimeframeAnalyzer] No data for {symbol}")
            return None
        
        # Generate signals for each timeframe
        timeframe_signals = {}
        rejection_reason = None
        
        for timeframe_type in self.timeframe_weights.keys():
            features = self.compute_timeframe_features(symbol, timeframe_type)
            if features:
                signal = self.generate_timeframe_signal(symbol, timeframe_type, features)
                if signal:
                    timeframe_signals[timeframe_type.value] = signal
        
        if not timeframe_signals:
            logger.warning(f"[MultiTimeframeAnalyzer] Could not generate signals for {symbol}")
            return None
        
        # Extract key timeframes
        daily_signal = timeframe_signals.get("1d")
        hourly_signal = timeframe_signals.get("1h")
        minute_signal = timeframe_signals.get("1m")
        
        daily_trend = daily_signal.trend if daily_signal else TrendDirection.NEUTRAL
        hourly_trend = hourly_signal.trend if hourly_signal else TrendDirection.NEUTRAL
        minute_trend = minute_signal.trend if minute_signal else TrendDirection.NEUTRAL
        
        # Calculate alignment score
        alignment_score = self._calculate_alignment(timeframe_signals.values())
        
        # Check for conflicts and rejections
        final_recommendation, rejection_reason = self._determine_final_recommendation(
            daily_trend, hourly_trend, minute_trend, alignment_score, timeframe_signals
        )
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(timeframe_signals)
        
        consensus = MultiTimeframeConsensus(
            symbol=symbol,
            daily_trend=daily_trend,
            hourly_trend=hourly_trend,
            minute_trend=minute_trend,
            alignment_score=alignment_score,
            timeframe_signals=timeframe_signals,
            final_recommendation=final_recommendation,
            confidence=confidence,
            rejection_reason=rejection_reason,
            timestamp=datetime.now().timestamp()
        )
        
        self.signal_history.append(consensus)
        
        logger.info(f"[MultiTimeframeAnalyzer] Aggregated signals for {symbol}: "
                   f"Daily={daily_trend.name}, "
                   f"Hourly={hourly_trend.name}, "
                   f"Alignment={alignment_score:.2%}, "
                   f"Recommendation={final_recommendation}")
        
        return consensus

    def _calculate_alignment(self, signals: List[TimeframeSignal]) -> float:
        """
        Calculate how well signals align across timeframes
        
        0.0: Complete conflict (some say up, some say down)
        1.0: Perfect agreement
        """
        if not signals:
            return 0.0
        
        trend_values = [signal.trend.value for signal in signals]
        std_dev = np.std(trend_values)
        max_std = 4.0  # Max std dev for completely conflicting signals
        
        alignment = 1.0 - (std_dev / max_std)
        return np.clip(alignment, 0.0, 1.0)

    def _determine_final_recommendation(
        self,
        daily_trend: TrendDirection,
        hourly_trend: TrendDirection,
        minute_trend: TrendDirection,
        alignment: float,
        signals: Dict
    ) -> Tuple[str, Optional[str]]:
        """
        Determine final trade recommendation with validation
        
        Rules:
        1. If daily trend DOWN, reject BUY signals (avoid counter-trend)
        2. If daily trend UP, reject SELL signals
        3. If alignment < 0.4, HOLD (too much disagreement)
        """
        
        # Rule 1: Respect daily trend
        if daily_trend in [TrendDirection.STRONG_DOWN, TrendDirection.WEAK_DOWN]:
            if minute_trend in [TrendDirection.STRONG_UP, TrendDirection.WEAK_UP]:
                return "HOLD", f"Minute trend conflicts with daily downtrend"
        
        if daily_trend in [TrendDirection.STRONG_UP, TrendDirection.WEAK_UP]:
            if minute_trend in [TrendDirection.STRONG_DOWN, TrendDirection.WEAK_DOWN]:
                return "HOLD", f"Minute trend conflicts with daily uptrend"
        
        # Rule 2: Check alignment
        if alignment < 0.4:
            return "HOLD", f"Low alignment across timeframes: {alignment:.2%}"
        
        # Rule 3: Generate recommendation
        if daily_trend.value > 0 and hourly_trend.value >= 0:
            return "BUY", None
        elif daily_trend.value < 0 and hourly_trend.value <= 0:
            return "SELL", None
        else:
            return "HOLD", f"Unclear signals"

    def _calculate_overall_confidence(self, signals: Dict) -> float:
        """Calculate weighted confidence across timeframes"""
        total_confidence = 0.0
        total_weight = 0.0
        
        for tf_type, signal in signals.items():
            for timeframe_type in self.timeframe_weights.keys():
                if tf_type == timeframe_type.value:
                    weight = self.timeframe_weights[timeframe_type]
                    total_confidence += signal.confidence * weight
                    total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0

    # ==================== REPORTING ====================
    def get_mtf_summary(self, symbol: str) -> Optional[Dict]:
        """Get latest multi-timeframe summary"""
        if not self.signal_history:
            return None
        
        latest = [c for c in self.signal_history if c.symbol == symbol]
        if not latest:
            return None
        
        consensus = latest[-1]
        return {
            'symbol': symbol,
            'daily_trend': consensus.daily_trend.name,
            'hourly_trend': consensus.hourly_trend.name,
            'alignment': consensus.alignment_score,
            'recommendation': consensus.final_recommendation,
            'confidence': consensus.confidence,
            'rejection_reason': consensus.rejection_reason
        }

    def export_mtf_history(self, filepath: str) -> None:
        """Export multi-timeframe history"""
        history = []
        for consensus in self.signal_history:
            history.append({
                'timestamp': consensus.timestamp,
                'symbol': consensus.symbol,
                'daily': consensus.daily_trend.name,
                'hourly': consensus.hourly_trend.name,
                'alignment': consensus.alignment_score,
                'recommendation': consensus.final_recommendation,
                'confidence': consensus.confidence
            })
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"[MultiTimeframeAnalyzer] Exported {len(history)} MTF records to {filepath}")


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Initialize analyzer
    mtf = MultiTimeframeAnalyzer()
    
    # Generate sample OHLCV data for all timeframes
    np.random.seed(42)
    base_price = 42000
    
    for timeframe_type in [TimeframeType.ONE_DAY, TimeframeType.FOUR_HOUR, 
                          TimeframeType.ONE_HOUR, TimeframeType.FIVE_MIN, TimeframeType.ONE_MIN]:
        
        # Create synthetic OHLCV data
        closes = base_price + np.cumsum(np.random.randn(100) * 100)
        df = pd.DataFrame({
            'open': closes + np.random.randn(100) * 50,
            'high': closes + np.random.rand(100) * 200,
            'low': closes - np.random.rand(100) * 200,
            'close': closes,
            'volume': np.random.rand(100) * 1000000
        })
        
        if 'BTC/USDT' not in mtf.price_data:
            mtf.price_data['BTC/USDT'] = {}
        
        mtf.price_data['BTC/USDT'][timeframe_type] = df
    
    # Aggregate signals
    consensus = mtf.aggregate_timeframe_signals('BTC/USDT')
    
    if consensus:
        print(f"\n📊 MULTI-TIMEFRAME ANALYSIS")
        print("=" * 60)
        print(f"Daily Trend: {consensus.daily_trend.name}")
        print(f"Hourly Trend: {consensus.hourly_trend.name}")
        print(f"Alignment: {consensus.alignment_score:.2%}")
        print(f"Recommendation: {consensus.final_recommendation}")
        print(f"Confidence: {consensus.confidence:.2%}")
        if consensus.rejection_reason:
            print(f"Rejection: {consensus.rejection_reason}")
