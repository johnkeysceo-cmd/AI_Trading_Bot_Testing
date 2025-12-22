"""Dynamic leverage controller for Phase 3.

Adjusts leverage (1x-3x) based on volatility, Sharpe ratio, and market
conditions to maximize risk-adjusted returns.
"""
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class LeverageMetrics:
    """Current leverage metrics."""
    current_leverage: float
    recommended_leverage: float
    volatility_percentile: float
    sharpe_ratio: float
    portfolio_return: float
    max_drawdown: float
    leverage_adjustment: float
    reason: str


class DynamicLeverageController:
    """Dynamically adjust leverage based on market conditions."""
    
    def __init__(self, min_leverage: float = 1.0, 
                 max_leverage: float = 3.0,
                 target_sharpe: float = 1.0):
        """
        Args:
            min_leverage: Minimum leverage (1x = no leverage)
            max_leverage: Maximum leverage
            target_sharpe: Target Sharpe ratio to maintain
        """
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.target_sharpe = target_sharpe
        
        self.current_leverage = 1.0
        self.historical_returns = []
        self.historical_volatility = []
        self.max_history = 252  # 1 year daily data
    
    def update_metrics(self, returns: np.ndarray, volatility: float) -> None:
        """Update historical metrics."""
        
        if len(returns) > 0:
            self.historical_returns.extend(returns[-10:])  # Last 10 returns
        
        self.historical_volatility.append(volatility)
        
        # Keep only recent history
        self.historical_returns = self.historical_returns[-self.max_history:]
        self.historical_volatility = self.historical_volatility[-self.max_history:]
    
    def calculate_leverage(self, portfolio_metrics: Dict) -> LeverageMetrics:
        """Calculate optimal leverage."""
        
        volatility = portfolio_metrics.get("volatility", 0.02)
        current_sharpe = portfolio_metrics.get("sharpe_ratio", 0.5)
        portfolio_return = portfolio_metrics.get("portfolio_return", 0.0)
        max_drawdown = portfolio_metrics.get("max_drawdown", 0.1)
        
        # Volatility-based adjustment
        volatility_percentile = self._calculate_volatility_percentile(volatility)
        volatility_leverage = self._volatility_to_leverage(volatility_percentile)
        
        # Sharpe-based adjustment
        sharpe_leverage = self._sharpe_to_leverage(current_sharpe)
        
        # Drawdown adjustment (reduce leverage on high drawdown)
        drawdown_leverage = self._drawdown_to_leverage(max_drawdown)
        
        # Composite leverage
        recommended = np.mean([volatility_leverage, sharpe_leverage, drawdown_leverage])
        recommended = np.clip(recommended, self.min_leverage, self.max_leverage)
        
        # Smooth adjustment (don't change too fast)
        adjustment = self._smooth_adjustment(self.current_leverage, recommended)
        new_leverage = self.current_leverage + adjustment
        new_leverage = np.clip(new_leverage, self.min_leverage, self.max_leverage)
        
        # Determine reason
        reason = self._determine_reason(
            volatility_percentile, current_sharpe, max_drawdown
        )
        
        metrics = LeverageMetrics(
            current_leverage=self.current_leverage,
            recommended_leverage=new_leverage,
            volatility_percentile=volatility_percentile,
            sharpe_ratio=current_sharpe,
            portfolio_return=portfolio_return,
            max_drawdown=max_drawdown,
            leverage_adjustment=new_leverage - self.current_leverage,
            reason=reason
        )
        
        self.current_leverage = new_leverage
        return metrics
    
    def _volatility_to_leverage(self, percentile: float) -> float:
        """Convert volatility percentile to leverage.
        
        Low volatility (0%ile) → High leverage (3x)
        High volatility (100%ile) → Low leverage (1x)
        """
        # Inverse relationship: high vol = low leverage
        return self.max_leverage - (percentile / 100) * (self.max_leverage - self.min_leverage)
    
    def _sharpe_to_leverage(self, sharpe: float) -> float:
        """Convert Sharpe ratio to leverage.
        
        Low Sharpe (<0.5) → Minimum leverage (1x)
        High Sharpe (>2) → Maximum leverage (3x)
        """
        if sharpe <= 0:
            return self.min_leverage
        elif sharpe >= 2:
            return self.max_leverage
        else:
            # Linear interpolation 0.5-2.0 → 1.0-3.0
            return self.min_leverage + (sharpe / 2) * (self.max_leverage - self.min_leverage)
    
    def _drawdown_to_leverage(self, max_dd: float) -> float:
        """Convert max drawdown to leverage.
        
        Small drawdown (<5%) → Maximum leverage (3x)
        Large drawdown (>20%) → Minimum leverage (1x)
        """
        if max_dd <= 0.05:
            return self.max_leverage
        elif max_dd >= 0.20:
            return self.min_leverage
        else:
            # Linear interpolation
            return self.max_leverage - (max_dd - 0.05) / 0.15 * (self.max_leverage - self.min_leverage)
    
    def _calculate_volatility_percentile(self, vol: float) -> float:
        """Calculate volatility percentile from history."""
        
        if len(self.historical_volatility) < 2:
            return 50.0  # Default to 50th percentile
        
        return np.percentile(self.historical_volatility, np.searchsorted(
            sorted(self.historical_volatility), vol
        ) / len(self.historical_volatility) * 100)
    
    def _smooth_adjustment(self, current: float, target: float, 
                          max_change: float = 0.2) -> float:
        """Smooth leverage changes to avoid whipsaws."""
        
        change = target - current
        if abs(change) > max_change:
            change = max_change if change > 0 else -max_change
        
        return change
    
    def _determine_reason(self, vol_percentile: float, sharpe: float, 
                         max_dd: float) -> str:
        """Determine why leverage is being adjusted."""
        
        reasons = []
        
        if vol_percentile > 75:
            reasons.append("high volatility")
        elif vol_percentile < 25:
            reasons.append("low volatility")
        
        if sharpe < 0.5:
            reasons.append("poor sharpe")
        elif sharpe > 2:
            reasons.append("excellent sharpe")
        
        if max_dd > 0.15:
            reasons.append("high drawdown")
        
        return ", ".join(reasons) if reasons else "normal"


class LeverageManager:
    """Manage leverage across portfolio."""
    
    def __init__(self):
        self.leverage_by_symbol = {}
        self.total_notional = 0.0
        self.total_margin_required = 0.0
        self.max_portfolio_leverage = 3.0
    
    def calculate_position_size(self, capital: float, symbol: str, 
                               leverage: float, price: float) -> float:
        """Calculate position size given leverage."""
        
        notional = capital * leverage
        position_size = notional / price
        
        return position_size
    
    def check_margin_requirements(self, positions: Dict[str, float], 
                                  prices: Dict[str, float],
                                  capital: float) -> Tuple[bool, float]:
        """Check if margin requirements are met.
        
        Returns:
            (is_valid, maintenance_ratio)
        """
        
        total_notional = sum(
            positions.get(symbol, 0) * prices.get(symbol, 0)
            for symbol in positions
        )
        
        # Maintenance margin requirement: 20% (5x leverage)
        maintenance_required = total_notional * 0.20
        maintenance_ratio = capital / maintenance_required if maintenance_required > 0 else np.inf
        
        # Fail if can't maintain 20% margin
        if maintenance_ratio < 1.0:
            return False, maintenance_ratio
        
        return True, maintenance_ratio
    
    def calculate_margin_call_level(self, capital: float, total_notional: float) -> float:
        """Calculate price level that triggers margin call."""
        
        # Margin call at 15% maintenance (can't liquidate at 20%)
        available_margin = capital * 0.15
        
        if total_notional == 0:
            return 0
        
        # Max price increase before margin call
        max_adverse_move = available_margin / total_notional
        
        return max_adverse_move


if __name__ == "__main__":
    # Demo
    controller = DynamicLeverageController()
    
    # Simulate different market conditions
    scenarios = [
        {
            "name": "Low vol, high sharpe",
            "volatility": 0.005,
            "sharpe_ratio": 2.5,
            "portfolio_return": 0.15,
            "max_drawdown": 0.03
        },
        {
            "name": "High vol, low sharpe",
            "volatility": 0.04,
            "sharpe_ratio": 0.2,
            "portfolio_return": -0.05,
            "max_drawdown": 0.25
        },
        {
            "name": "Medium vol, medium sharpe",
            "volatility": 0.015,
            "sharpe_ratio": 1.0,
            "portfolio_return": 0.08,
            "max_drawdown": 0.10
        }
    ]
    
    print("\nDynamic Leverage Controller Demo:")
    print("=" * 70)
    
    for scenario in scenarios:
        name = scenario.pop("name")
        metrics = controller.calculate_leverage(scenario)
        
        print(f"\n{name}:")
        print(f"  Volatility: {scenario['volatility']:.2%}")
        print(f"  Sharpe: {scenario['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {scenario['max_drawdown']:.1%}")
        print(f"  → Recommended Leverage: {metrics.recommended_leverage:.2f}x")
        print(f"     (Adjustment: {metrics.leverage_adjustment:+.2f}x, Reason: {metrics.reason})")
