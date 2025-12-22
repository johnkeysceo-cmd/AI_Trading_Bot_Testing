"""Realistic gym-compatible trading environment for Phase 2 RL training.

This module provides a `TradingEnv` class compatible with OpenAI gym/gymnasium.
It simulates realistic trading with:
- Market microstructure (slippage, fees, spread)
- Position management (leverage, margin calls)
- Episode termination on drawdown or margin
- Realistic reward functions (PnL - risk penalties)
"""
import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """Realistic trading environment for RL agents.
    
    State: [price, returns, volatility, RSI, MACD, portfolio_value, position_size, leverage]
    Action: {0: HOLD, 1: BUY_0.25x, 2: BUY_0.5x, 3: BUY_1.0x, ..., 8: SELL_1.0x}
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, data: np.ndarray, initial_capital: float = 10000, 
                 max_leverage: float = 3.0, fee_pct: float = 0.001,
                 slippage_pct: float = 0.0005):
        """
        Args:
            data: OHLCV array, shape (timesteps, 5+) with [open, high, low, close, volume, ...]
            initial_capital: Starting portfolio value in USD
            max_leverage: Maximum leverage allowed
            fee_pct: Trading fee as % of notional
            slippage_pct: Slippage as % of entry price
        """
        self.data = data
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        
        # Episode state
        self.current_step = 0
        self.position_size = 0.0  # in base units
        self.entry_price = 0.0
        self.leverage = 1.0
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.max_portfolio_value = initial_capital
        self.trades_closed = 0
        self.trades_pnl = []
        
        # Gym interface
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)  # 0=HOLD, 1-4=BUY, 5-8=SELL
        
    def _get_observation(self) -> np.ndarray:
        """Return current state vector."""
        if self.current_step >= len(self.data):
            return np.zeros(8, dtype=np.float32)
        
        row = self.data[self.current_step]
        price = row[3]  # close
        
        # Calculate returns and volatility from rolling window
        if self.current_step > 20:
            recent_closes = self.data[max(0, self.current_step-20):self.current_step+1, 3]
            returns = np.diff(recent_closes) / recent_closes[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0.0
            price_returns = (price - self.data[self.current_step-1, 3]) / self.data[self.current_step-1, 3]
        else:
            volatility = 0.0
            price_returns = 0.0
        
        # Simple RSI proxy (momentum)
        rsi = self._calculate_rsi(self.current_step)
        macd = self._calculate_macd(self.current_step)
        
        unrealized_pnl = 0.0
        if self.position_size != 0:
            unrealized_pnl = self.position_size * (price - self.entry_price)
        
        obs = np.array([
            price,                          # current price
            price_returns,                  # momentum
            volatility,                     # volatility
            (rsi - 50) / 50,               # RSI normalized
            macd,                          # MACD normalized
            self.portfolio_value / self.initial_capital - 1,  # total return
            self.position_size / (self.initial_capital / price + 1e-6),  # position % of capital
            self.leverage - 1.0             # leverage above 1x
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_rsi(self, step: int, period: int = 14) -> float:
        """Simple RSI calculation."""
        if step < period:
            return 50.0
        
        closes = self.data[step-period:step+1, 3]
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.where(deltas < 0, -deltas, 0).mean()
        
        if losses == 0:
            return 100.0 if gains > 0 else 50.0
        
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, step: int) -> float:
        """Simple MACD calculation."""
        if step < 26:
            return 0.0
        
        closes = self.data[max(0, step-26):step+1, 3]
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd = ema12 - ema26
        
        # Normalize
        return macd / (closes[-1] + 1e-6)
    
    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Calculate EMA of last value."""
        if len(data) < period:
            return data.mean()
        
        multiplier = 2.0 / (period + 1)
        ema = data[:period].mean()
        for val in data[period:]:
            ema = (val - ema) * multiplier + ema
        return ema
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step of environment."""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, True, {}
        
        price_before = self.data[self.current_step, 3]
        self.current_step += 1
        price_after = self.data[self.current_step, 3]
        
        reward = 0.0
        info = {"step": self.current_step}
        
        # Execute action
        if action == 0:  # HOLD
            pass
        elif action in [1, 2, 3, 4]:  # BUY
            target_size = [0.25, 0.5, 1.0, 2.0][action - 1]
            self._open_position(price_before, target_size)
            reward -= 0.01  # Action cost
        elif action in [5, 6, 7, 8]:  # SELL
            self._close_position(price_before)
            reward -= 0.01  # Action cost
        
        # Update unrealized PnL
        if self.position_size != 0:
            unrealized = self.position_size * (price_after - self.entry_price)
            reward += (unrealized / self.initial_capital) * 100  # PnL reward
            
            # Margin call check
            margin_ratio = self.cash / (self.position_size * price_after * self.leverage + 1e-6)
            if margin_ratio < 0.1:
                self._close_position(price_after)
                reward -= 5.0  # Penalty for liquidation
                info["liquidated"] = True
        
        # Update portfolio value
        self.portfolio_value = self.cash + (self.position_size * price_after if self.position_size != 0 else 0)
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        
        # Drawdown penalty
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        reward -= drawdown * 0.1
        
        # Terminal condition: exceed max drawdown
        done = drawdown > 0.2  # 20% max drawdown
        
        if done:
            reward -= 10.0
            info["reason"] = "max_drawdown"
        
        return self._get_observation(), reward, done, info
    
    def _open_position(self, price: float, size_multiple: float):
        """Open a long position."""
        if self.position_size != 0:
            return  # Already in position
        
        # Calculate position size with leverage
        notional = self.cash * self.leverage * size_multiple
        position_size = notional / price
        
        # Apply slippage and fees
        entry_price = price * (1 + self.slippage_pct)
        fee = notional * self.fee_pct
        
        self.position_size = position_size
        self.entry_price = entry_price
        self.cash -= fee
        self.leverage = min(self.max_leverage, self.leverage + 0.5)
    
    def _close_position(self, price: float):
        """Close any open position."""
        if self.position_size == 0:
            return
        
        exit_price = price * (1 - self.slippage_pct)
        pnl = self.position_size * (exit_price - self.entry_price)
        fee = self.position_size * exit_price * self.fee_pct
        
        self.cash += pnl - fee
        self.trades_pnl.append(pnl)
        self.trades_closed += 1
        
        self.position_size = 0.0
        self.entry_price = 0.0
        self.leverage = 1.0
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self.current_step = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.leverage = 1.0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        self.trades_closed = 0
        self.trades_pnl = []
        
        return self._get_observation()
    
    def render(self, mode="human"):
        """Print current state."""
        price = self.data[min(self.current_step, len(self.data) - 1), 3]
        print(f"Step {self.current_step} | Price ${price:.2f} | "
              f"Portfolio ${self.portfolio_value:.2f} | "
              f"Position {self.position_size:.4f} | "
              f"Leverage {self.leverage:.2f}x | "
              f"Trades {self.trades_closed}")


if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_steps = 1000
    prices = 100 * np.cumprod(1 + np.random.normal(0.0001, 0.01, n_steps))
    ohlcv = np.column_stack([prices, prices, prices, prices, np.ones(n_steps)])
    
    env = TradingEnv(ohlcv, initial_capital=10000)
    obs = env.reset()
    
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    
    print(f"Sample episode | Total reward: {total_reward:.2f} | "
          f"Final portfolio: ${env.portfolio_value:.2f} | "
          f"Trades: {env.trades_closed}")
