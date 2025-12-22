"""
PHASE 4: Real-Time Regime Detection Engine

Advanced regime detection using Hidden Markov Models, state-space models,
and Kalman filtering for dynamic market regime identification.

Continuously tracks market regimes (bull/bear/sideways) and adapts
trading strategy based on detected regime transitions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


class MarketRegime(Enum):
    """Market regime states."""
    BULL_MARKET = "bull"
    BEAR_MARKET = "bear"
    SIDEWAYS_MARKET = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    MEAN_REVERSION = "mean_revert"


@dataclass
class RegimeState:
    """Current regime state."""
    regime: MarketRegime
    probability: float
    confidence: float
    change_probability: float
    
    # Transition info
    days_in_regime: int
    expected_duration: float
    
    # Market characteristics
    volatility: float
    trend_strength: float
    autocorrelation: float
    kurtosis: float
    
    # Technical indicators
    rsi: float
    macd: float
    bollinger_position: float  # Position within bands
    vix_level: float
    
    def __str__(self):
        return f"""
Regime: {self.regime.value}
Probability: {self.probability:.4f} | Confidence: {self.confidence:.4f}
Days in Regime: {self.days_in_regime}
Expected Duration: {self.expected_duration:.1f} days

Market Characteristics:
  Volatility: {self.volatility:.4f}
  Trend Strength: {self.trend_strength:.4f}
  Autocorrelation: {self.autocorrelation:.4f}
  Kurtosis: {self.kurtosis:.4f}

Technical Signals:
  RSI: {self.rsi:.2f}
  MACD: {self.macd:.6f}
  Bollinger Position: {self.bollinger_position:.4f}
  VIX Level: {self.vix_level:.2f}
"""


class HiddenMarkovModel:
    """
    Hidden Markov Model for regime detection.
    
    Uses EM algorithm for parameter estimation and Viterbi for decoding.
    """
    
    def __init__(self, n_states: int = 4, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter = n_iter
        
        # HMM parameters
        self.transition_matrix = None
        self.emission_params = None
        self.initial_probs = None
        
        self.is_fitted = False
    
    def fit(self, observations: np.ndarray):
        """Fit HMM using Baum-Welch (EM) algorithm."""
        
        self.observations = observations.reshape(-1, 1) if observations.ndim == 1 else observations
        self.T, self.D = self.observations.shape
        
        # Initialize parameters randomly
        self._initialize_parameters()
        
        # EM iterations
        for iteration in range(self.n_iter):
            # E-step: Forward-backward algorithm
            alphas = self._forward_pass()
            betas = self._backward_pass()
            
            # Compute responsibilities
            gammas = alphas * betas
            gammas /= np.sum(gammas, axis=0, keepdims=True)
            
            # M-step: Update parameters
            self._update_parameters(gammas, alphas, betas)
            
            # Log likelihood
            log_prob = np.sum(np.log(np.sum(alphas, axis=0) + 1e-10))
            
            if iteration % 10 == 0:
                pass  # Can log progress here
        
        self.is_fitted = True
    
    def _initialize_parameters(self):
        """Initialize HMM parameters."""
        
        # Transition matrix: each row sums to 1
        self.transition_matrix = np.random.dirichlet(np.ones(self.n_states), self.n_states)
        
        # Initial probabilities
        self.initial_probs = np.random.dirichlet(np.ones(self.n_states))
        
        # Emission parameters (mean and std for each state)
        self.emission_params = {
            'means': np.random.randn(self.n_states),
            'stds': np.random.rand(self.n_states) + 0.5
        }
    
    def _forward_pass(self) -> np.ndarray:
        """Forward algorithm: compute alphas."""
        
        alphas = np.zeros((self.n_states, self.T))
        
        # Initialize
        alphas[:, 0] = self.initial_probs * self._emission_prob(0)
        
        # Recursion
        for t in range(1, self.T):
            for j in range(self.n_states):
                alphas[j, t] = np.sum(
                    alphas[:, t-1] * self.transition_matrix[:, j]
                ) * self._emission_prob(t, j)
        
        return alphas
    
    def _backward_pass(self) -> np.ndarray:
        """Backward algorithm: compute betas."""
        
        betas = np.zeros((self.n_states, self.T))
        
        # Initialize
        betas[:, -1] = 1.0
        
        # Recursion
        for t in range(self.T - 2, -1, -1):
            for i in range(self.n_states):
                betas[i, t] = np.sum(
                    self.transition_matrix[i, :] * 
                    self._emission_prob(t + 1) * 
                    betas[:, t + 1]
                )
        
        return betas
    
    def _emission_prob(self, t: int, state: Optional[int] = None) -> np.ndarray:
        """Compute emission probabilities."""
        
        obs = self.observations[t].reshape(1, -1)
        means = self.emission_params['means'].reshape(-1, 1)
        stds = self.emission_params['stds'].reshape(-1, 1)
        
        if state is not None:
            return stats.norm.pdf(obs, means[state], stds[state] + 1e-10)
        
        probs = stats.norm.pdf(obs, means, stds + 1e-10)
        return probs.ravel()
    
    def _update_parameters(self, gammas, alphas, betas):
        """Update HMM parameters in M-step."""
        
        # Update initial probabilities
        self.initial_probs = gammas[:, 0] / np.sum(gammas[:, 0])
        
        # Update transition matrix
        xi_sums = np.zeros((self.n_states, self.n_states))
        
        for t in range(self.T - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi = (
                        alphas[i, t] * 
                        self.transition_matrix[i, j] * 
                        self._emission_prob(t + 1, j) * 
                        betas[j, t + 1]
                    )
                    xi_sums[i, j] += xi
        
        for i in range(self.n_states):
            self.transition_matrix[i, :] = xi_sums[i, :] / (np.sum(xi_sums[i, :]) + 1e-10)
        
        # Update emission parameters
        for j in range(self.n_states):
            gamma_sum = np.sum(gammas[j, :])
            
            mean = np.sum(gammas[j, :] * self.observations.ravel()) / (gamma_sum + 1e-10)
            variance = np.sum(
                gammas[j, :] * (self.observations.ravel() - mean)**2
            ) / (gamma_sum + 1e-10)
            
            self.emission_params['means'][j] = mean
            self.emission_params['stds'][j] = np.sqrt(variance + 1e-10)
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Decode states using Viterbi algorithm."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        obs = observations.reshape(-1, 1) if observations.ndim == 1 else observations
        T = len(obs)
        
        # Viterbi variables
        viterbi = np.zeros((self.n_states, T))
        backpointer = np.zeros((self.n_states, T), dtype=int)
        
        # Initialize
        viterbi[:, 0] = np.log(self.initial_probs + 1e-10) + np.log(self._emission_prob(0) + 1e-10)
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                candidates = viterbi[:, t-1] + np.log(self.transition_matrix[:, j] + 1e-10)
                backpointer[j, t] = np.argmax(candidates)
                viterbi[j, t] = (
                    np.max(candidates) + np.log(self._emission_prob(t, j) + 1e-10)
                )
        
        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(viterbi[:, -1])
        
        for t in range(T - 2, -1, -1):
            states[t] = backpointer[states[t + 1], t + 1]
        
        return states
    
    def get_state_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """Get probability distribution over states."""
        
        obs = observations.reshape(-1, 1) if observations.ndim == 1 else observations
        
        alphas = np.zeros((self.n_states, len(obs)))
        alphas[:, 0] = self.initial_probs * self._emission_prob(0)
        
        for t in range(1, len(obs)):
            for j in range(self.n_states):
                alphas[j, t] = np.sum(
                    alphas[:, t-1] * self.transition_matrix[:, j]
                ) * self._emission_prob(t, j)
        
        # Normalize
        probs = alphas / np.sum(alphas, axis=0, keepdims=True)
        return probs.T


class KalmanFilterRegimeDetector:
    """
    Kalman filter-based regime detector for continuous state tracking.
    """
    
    def __init__(self, state_dim: int = 2, obs_dim: int = 1, process_noise: float = 0.01, obs_noise: float = 0.1):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # State transition matrix
        self.F = np.eye(state_dim)
        
        # Observation matrix
        self.H = np.zeros((obs_dim, state_dim))
        self.H[0, 0] = 1.0
        
        # Noise covariances
        self.Q = np.eye(state_dim) * process_noise
        self.R = np.eye(obs_dim) * obs_noise
        
        # State estimate and covariance
        self.x = np.zeros((state_dim, 1))
        self.P = np.eye(state_dim)
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state."""
        
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        return x_pred, P_pred
    
    def update(self, observation: float, x_pred: np.ndarray, P_pred: np.ndarray):
        """Update state with observation."""
        
        z = np.array([[observation]])
        
        # Innovation
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # Updated state
        self.x = x_pred + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred
    
    def filter(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter on observations."""
        
        n = len(observations)
        estimates = np.zeros((n, self.state_dim))
        variances = np.zeros((n, self.state_dim))
        
        for i, obs in enumerate(observations):
            x_pred, P_pred = self.predict()
            self.update(obs, x_pred, P_pred)
            
            estimates[i] = self.x.ravel()
            variances[i] = np.diag(self.P)
        
        return estimates, variances


class RegimeDetectionEngine:
    """
    Comprehensive regime detection engine combining multiple approaches.
    """
    
    def __init__(self, price_data: np.ndarray, window_size: int = 60):
        self.prices = price_data
        self.returns = np.diff(np.log(price_data))
        self.window_size = window_size
        
        # HMM for regime detection
        self.hmm = HiddenMarkovModel(n_states=4)
        self.hmm.fit(self.returns)
        
        # Kalman filter
        self.kf = KalmanFilterRegimeDetector()
        
        # Regime history
        self.regime_history = []
        self.state_probabilities = []
    
    def detect_regimes(self) -> np.ndarray:
        """Detect regimes using HMM."""
        
        states = self.hmm.predict(self.returns)
        self.regime_history = states
        
        return states
    
    def get_current_state_probabilities(self) -> np.ndarray:
        """Get probability distribution over regimes at current time."""
        
        probs = self.hmm.get_state_probabilities(self.returns)
        self.state_probabilities = probs
        
        return probs[-1] if len(probs) > 0 else np.ones(4) / 4
    
    def analyze_regime_characteristics(self, window: int = 20) -> Dict:
        """Analyze characteristics of detected regimes."""
        
        states = self.regime_history
        characteristics = {}
        
        for state in range(self.hmm.n_states):
            state_returns = self.returns[states == state]
            
            if len(state_returns) == 0:
                continue
            
            characteristics[f'state_{state}'] = {
                'mean_return': np.mean(state_returns),
                'volatility': np.std(state_returns),
                'sharpe': np.mean(state_returns) / (np.std(state_returns) + 1e-10),
                'skewness': stats.skew(state_returns),
                'kurtosis': stats.kurtosis(state_returns),
                'autocorr_1': np.corrcoef(state_returns[:-1], state_returns[1:])[0, 1],
                'max_drawdown': self._calculate_max_drawdown(state_returns),
                'observation_count': len(state_returns)
            }
        
        return characteristics
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown) if len(drawdown) > 0 else 0
    
    def predict_regime_transition(self, horizon: int = 5) -> Dict:
        """Predict probability of regime transition in next horizon periods."""
        
        current_state = self.regime_history[-1]
        transition_matrix = self.hmm.transition_matrix
        
        # Project forward
        probs = transition_matrix[current_state].copy()
        
        for _ in range(horizon - 1):
            probs = probs @ transition_matrix
        
        return {
            'current_state': current_state,
            'transition_probs': probs,
            'most_likely_next': np.argmax(probs),
            'confidence': np.max(probs)
        }
    
    def estimate_regime_duration(self, state: int) -> float:
        """Estimate expected duration of a regime."""
        
        # Duration is related to self-transition probability
        transition_prob = self.hmm.transition_matrix[state, state]
        
        # Expected duration = 1 / (1 - p)
        expected_duration = 1.0 / (1.0 - transition_prob + 1e-10) if transition_prob < 1 else np.inf
        
        return expected_duration
    
    def compute_regime_state(self) -> RegimeState:
        """Compute comprehensive current regime state."""
        
        current_returns = self.returns[-self.window_size:]
        current_prices = self.prices[-self.window_size:]
        
        # Volatility
        volatility = np.std(current_returns)
        
        # Trend
        log_prices = np.log(current_prices)
        trend = log_prices[-1] - log_prices[0]
        trend_strength = np.abs(trend) / (volatility * np.sqrt(self.window_size) + 1e-10)
        
        # Autocorrelation
        if len(current_returns) > 1:
            autocorr = np.corrcoef(current_returns[:-1], current_returns[1:])[0, 1]
        else:
            autocorr = 0.0
        
        # Kurtosis
        kurtosis = stats.kurtosis(current_returns)
        
        # Technical indicators
        rsi = self._calculate_rsi(current_prices)
        macd_val = self._calculate_macd(current_prices)
        bb_pos = self._calculate_bollinger_position(current_prices)
        vix_level = volatility * 100  # Annualized volatility approximation
        
        # Current regime probabilities
        current_probs = self.get_current_state_probabilities()
        current_state = np.argmax(current_probs)
        current_prob = current_probs[current_state]
        
        # Confidence
        confidence = 1.0 - np.sum(np.sort(current_probs)[-2:])  # Spread between top 2
        
        # Days in regime
        days_in_regime = 1
        for i in range(len(self.regime_history) - 2, -1, -1):
            if self.regime_history[i] == current_state:
                days_in_regime += 1
            else:
                break
        
        # Expected duration
        expected_duration = self.estimate_regime_duration(current_state)
        
        # Determine regime type based on characteristics
        regime_map = {
            0: MarketRegime.BULL_MARKET,
            1: MarketRegime.BEAR_MARKET,
            2: MarketRegime.SIDEWAYS_MARKET,
            3: MarketRegime.HIGH_VOLATILITY
        }
        
        regime = regime_map.get(current_state, MarketRegime.SIDEWAYS_MARKET)
        
        # Transition probability
        transition_prob = 1.0 - self.hmm.transition_matrix[current_state, current_state]
        
        return RegimeState(
            regime=regime,
            probability=float(current_prob),
            confidence=float(confidence),
            change_probability=float(transition_prob),
            days_in_regime=days_in_regime,
            expected_duration=float(expected_duration),
            volatility=float(volatility),
            trend_strength=float(trend_strength),
            autocorrelation=float(autocorr),
            kurtosis=float(kurtosis),
            rsi=float(rsi),
            macd=float(macd_val),
            bollinger_position=float(bb_pos),
            vix_level=float(vix_level)
        )
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_macd(self, prices: np.ndarray) -> float:
        """Calculate MACD."""
        
        if len(prices) < 26:
            return 0.0
        
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        
        return float(ema12 - ema26)
    
    def _calculate_bollinger_position(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate position within Bollinger Bands (0 = lower, 1 = upper)."""
        
        if len(prices) < period:
            return 0.5
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        current_price = prices[-1]
        position = (current_price - lower_band) / (upper_band - lower_band + 1e-10)
        
        return float(np.clip(position, 0, 1))
    
    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate exponential moving average."""
        
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])
        
        for price in prices[period:]:
            ema = price * multiplier + ema * (1 - multiplier)
        
        return ema
    
    def detect_volatility_regime(self, threshold: float = None) -> str:
        """Classify as high or low volatility."""
        
        current_vol = np.std(self.returns[-self.window_size:])
        
        if threshold is None:
            threshold = np.median([np.std(self.returns[i:i+self.window_size]) 
                                  for i in range(len(self.returns) - self.window_size)])
        
        return "high_volatility" if current_vol > threshold else "low_volatility"
    
    def adaptive_parameters(self) -> Dict:
        """Get adaptive trading parameters based on regime."""
        
        regime = self.compute_regime_state()
        
        params = {
            'position_size_multiplier': 1.0,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 5.0,
            'max_leverage': 1.0,
            'risk_per_trade': 1.0,
            'rebalance_frequency': 'daily'
        }
        
        # Adjust based on regime
        if regime.volatility > 0.03:  # High volatility
            params['position_size_multiplier'] *= 0.5
            params['stop_loss_pct'] *= 1.5
            params['max_leverage'] = 1.0
        elif regime.volatility < 0.01:  # Low volatility
            params['position_size_multiplier'] *= 1.5
            params['stop_loss_pct'] *= 0.8
            params['max_leverage'] = 2.0
        
        # Adjust based on trend
        if regime.trend_strength > 0.7:
            params['position_size_multiplier'] *= 1.3
            params['rebalance_frequency'] = 'weekly'
        elif regime.trend_strength < 0.2:
            params['position_size_multiplier'] *= 0.7
            params['rebalance_frequency'] = 'every_4_hours'
        
        return params


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate price data with regime changes
    n = 1000
    prices = np.zeros(n)
    prices[0] = 100
    
    # Bull regime
    prices[0:250] = 100 + np.cumsum(np.random.randn(250) * 0.01 + 0.002)
    
    # Bear regime
    prices[250:500] = prices[249] + np.cumsum(np.random.randn(250) * 0.01 - 0.002)
    
    # Sideways regime
    prices[500:750] = prices[499] + np.cumsum(np.random.randn(250) * 0.005)
    
    # Bull again
    prices[750:] = prices[749] + np.cumsum(np.random.randn(250) * 0.01 + 0.002)
    
    # Initialize detector
    detector = RegimeDetectionEngine(prices, window_size=20)
    
    print("=" * 80)
    print("REGIME DETECTION ENGINE")
    print("=" * 80)
    
    # Detect regimes
    states = detector.detect_regimes()
    print(f"\nDetected regimes (last 10): {states[-10:]}")
    
    # Analyze characteristics
    chars = detector.analyze_regime_characteristics()
    print(f"\nRegime characteristics:")
    for regime, chars_dict in chars.items():
        print(f"\n{regime}:")
        for key, val in chars_dict.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")
    
    # Current state
    current_state = detector.compute_regime_state()
    print(f"\nCurrent regime state:\n{current_state}")
    
    # Adaptive parameters
    params = detector.adaptive_parameters()
    print(f"\nAdaptive trading parameters:")
    for key, val in params.items():
        print(f"  {key}: {val}")
    
    # Transition prediction
    transition = detector.predict_regime_transition(horizon=5)
    print(f"\nRegime transition prediction (5 days):")
    print(f"  Current state: {transition['current_state']}")
    print(f"  Transition probabilities: {transition['transition_probs']}")
    print(f"  Most likely next: {transition['most_likely_next']}")
    print(f"  Confidence: {transition['confidence']:.4f}")
