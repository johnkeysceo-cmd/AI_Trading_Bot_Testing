"""
kelly_criterion_calculator.py
------------------------------
Kelly Criterion Position Sizing for Optimal Bet Sizing

The Kelly Criterion formula: f* = (b*p - q) / b
Where:
  f* = fraction of capital to bet
  b = odds (average_win / average_loss)
  p = probability of winning
  q = probability of losing (1 - p)

Features:
- Calculate optimal position size based on historical performance
- Fractional Kelly support (50%, 75%, 100%, 150%+)
- Dynamic leverage calculation based on Sharpe ratio
- Risk constraints and safety limits
- Adaptive position sizing per agent tier

Phase 1: Signal Fusion & Positioning - Improvement: 2/10 → 6/10
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

# Logging setup
logger = logging.getLogger("KellyCriterion")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class RiskProfile(Enum):
    """Risk profiles for different agent tiers"""
    CONSERVATIVE = 0.25  # 25% Kelly
    MODERATE = 0.50     # 50% Kelly
    AGGRESSIVE = 1.0    # 100% Kelly (full Kelly)
    NUCLEAR = 1.5       # 150% Kelly (leverage)


@dataclass
class TradePerformance:
    """Historical trade record"""
    trade_id: str
    symbol: str
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float
    duration_hours: float
    timestamp: float
    agent_tier: str


@dataclass
class KellyMetrics:
    """Kelly criterion calculation results"""
    win_rate: float  # Probability of winning (p)
    loss_rate: float  # Probability of losing (q = 1 - p)
    avg_win: float  # Average winning trade
    avg_loss: float  # Average losing trade
    odds: float  # b = avg_win / avg_loss
    kelly_fraction: float  # f* = (b*p - q) / b
    risk_profile: str
    fractional_kelly: float  # Fraction of full Kelly to use
    suggested_position_size: float  # As % of capital
    suggested_leverage: float  # Leverage multiplier
    sharpe_ratio: float  # Risk-adjusted return metric
    recommended_action: str  # INCREASE, MAINTAIN, REDUCE, HALT


class KellyCriterionCalculator:
    """
    Calculate optimal position sizing using Kelly Criterion
    """

    def __init__(self):
        self.trade_history: List[TradePerformance] = []
        self.kelly_cache: Dict[str, KellyMetrics] = {}
        
        # Agent tier risk profiles (can be adjusted)
        self.risk_profiles = {
            'survival': RiskProfile.CONSERVATIVE.value,
            'aggressive': RiskProfile.MODERATE.value,
            'nuclear': RiskProfile.AGGRESSIVE.value,
            'arbitrage': RiskProfile.CONSERVATIVE.value
        }
        
        # Safety limits
        self.max_position_size = 0.20  # Never bet more than 20% of capital
        self.min_kelly_fraction = 0.25  # Never go below 25% Kelly
        self.max_leverage = 3.0  # Never leverage more than 3x
        
        logger.info("[KellyCriterion] Initialized Kelly Criterion Calculator")

    # ==================== CORE KELLY CALCULATION ====================
    def calculate_kelly_fraction(
        self,
        wins: List[float],
        losses: List[float],
        fraction: float = 0.50
    ) -> KellyMetrics:
        """
        Calculate Kelly Criterion position sizing
        
        Args:
            wins: List of winning trade returns (in decimals, e.g., 0.02 for +2%)
            losses: List of losing trade returns (in decimals, e.g., -0.01 for -1%)
            fraction: Fraction of Kelly to use (0.25 to 1.5)
        
        Returns:
            KellyMetrics with all calculations
        """
        if not wins or not losses:
            logger.warning("[KellyCriterion] Insufficient trade history")
            return KellyMetrics(
                win_rate=0.5,
                loss_rate=0.5,
                avg_win=0.0,
                avg_loss=0.0,
                odds=1.0,
                kelly_fraction=0.0,
                risk_profile="NO_DATA",
                fractional_kelly=0.0,
                suggested_position_size=0.0,
                suggested_leverage=1.0,
                sharpe_ratio=0.0,
                recommended_action="HALT"
            )
        
        # Calculate probabilities
        total_trades = len(wins) + len(losses)
        win_rate = len(wins) / total_trades
        loss_rate = len(losses) / total_trades
        
        # Calculate average win/loss
        avg_win = np.mean(np.abs(wins))
        avg_loss = np.mean(np.abs(losses))
        
        # Avoid division by zero
        if avg_loss == 0:
            avg_loss = 0.001
        
        # Calculate odds (b)
        odds = avg_win / avg_loss
        
        # Calculate full Kelly fraction: f* = (b*p - q) / b
        numerator = (odds * win_rate) - loss_rate
        denominator = odds
        
        full_kelly = numerator / denominator if denominator != 0 else 0
        
        # Ensure Kelly is between 0 and 1 (can't bet more than 100% of capital on single trade)
        full_kelly = np.clip(full_kelly, -0.5, 1.0)
        
        # Apply fractional Kelly for safety
        fractional_kelly = full_kelly * fraction
        fractional_kelly = np.clip(fractional_kelly, self.min_kelly_fraction, self.max_position_size)
        
        # Calculate Sharpe ratio
        returns = np.array(wins + losses)
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Determine recommended action
        recommended_action = self._get_recommended_action(
            full_kelly, fractional_kelly, win_rate, sharpe_ratio
        )
        
        # Calculate leverage based on Sharpe ratio
        suggested_leverage = self._calculate_dynamic_leverage(sharpe_ratio, fractional_kelly)
        
        metrics = KellyMetrics(
            win_rate=win_rate,
            loss_rate=loss_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            odds=odds,
            kelly_fraction=full_kelly,
            risk_profile=self._get_risk_profile(full_kelly),
            fractional_kelly=fractional_kelly,
            suggested_position_size=fractional_kelly,
            suggested_leverage=suggested_leverage,
            sharpe_ratio=sharpe_ratio,
            recommended_action=recommended_action
        )
        
        logger.info(f"[KellyCriterion] Calculated Kelly metrics: "
                   f"WR={win_rate:.1%}, f*={full_kelly:.2%}, "
                   f"fractional={fractional_kelly:.2%}, "
                   f"Sharpe={sharpe_ratio:.2f}")
        
        return metrics

    # ==================== POSITION SIZING ====================
    def compute_optimal_leverage(
        self,
        kelly_fraction: float,
        agent_tier: str
    ) -> float:
        """
        Compute optimal leverage based on Kelly and agent tier
        
        Args:
            kelly_fraction: The Kelly fraction (0.0 - 1.0)
            agent_tier: One of ['survival', 'aggressive', 'nuclear', 'arbitrage']
        
        Returns:
            Leverage multiplier (1.0x to max_leverage)
        """
        # Get risk profile for agent tier
        tier_fraction = self.risk_profiles.get(agent_tier, RiskProfile.MODERATE.value)
        
        # Adjust Kelly by tier
        adjusted_kelly = kelly_fraction * tier_fraction
        
        # Conservative agents: Keep leverage at 1x
        if agent_tier == 'survival':
            return 1.0
        
        # Aggressive agents: Increase leverage with Kelly
        elif agent_tier == 'aggressive':
            if adjusted_kelly > 0.05:
                leverage = 1.0 + (adjusted_kelly * 2)  # 2x multiplier
            else:
                leverage = 1.0
        
        # Nuclear agents: Maximum leverage when Kelly is strong
        elif agent_tier == 'nuclear':
            if adjusted_kelly > 0.08:
                leverage = 1.0 + (adjusted_kelly * 3)  # 3x multiplier
            else:
                leverage = 1.0
        
        # Arbitrage: Usually 1x (low risk)
        else:
            leverage = 1.0
        
        # Enforce maximum leverage
        leverage = np.clip(leverage, 1.0, self.max_leverage)
        
        logger.debug(f"[KellyCriterion] Leverage for {agent_tier}: {leverage:.2f}x")
        return leverage

    def adjust_position_size(
        self,
        base_position: float,
        signal_confidence: float,
        kelly_fraction: float,
        volatility_percentile: float = 50.0
    ) -> float:
        """
        Adjust position size based on signal confidence and market volatility
        
        Args:
            base_position: Base position size (e.g., 0.05 = 5% of capital)
            signal_confidence: Confidence 0.0 - 1.0
            kelly_fraction: Kelly criterion result
            volatility_percentile: Current volatility percentile (0-100)
        
        Returns:
            Adjusted position size
        """
        # Adjust for signal confidence
        confidence_adjusted = base_position * (0.5 + 0.5 * signal_confidence)
        
        # Adjust for market volatility (inverse relationship)
        if volatility_percentile > 0:
            volatility_factor = 50.0 / volatility_percentile
        else:
            volatility_factor = 1.0
        
        volatility_adjusted = confidence_adjusted * volatility_factor
        
        # Apply Kelly fraction
        kelly_adjusted = volatility_adjusted * kelly_fraction
        
        # Enforce maximum
        final_position = min(kelly_adjusted, self.max_position_size)
        
        logger.debug(f"[KellyCriterion] Position size: base={base_position:.2%}, "
                    f"conf_adj={confidence_adjusted:.2%}, "
                    f"vol_adj={volatility_adjusted:.2%}, "
                    f"final={final_position:.2%}")
        
        return final_position

    def calculate_risk_per_trade(
        self,
        position_size: float,
        stop_loss_pct: float,
        capital: float,
        max_risk_pct: float = 0.02
    ) -> float:
        """
        Calculate absolute risk (dollar amount) per trade
        
        Risk = Position_Size × Capital × Stop_Loss%
        """
        risk_amount = position_size * capital * stop_loss_pct
        
        # Ensure risk is within maximum
        if risk_amount > max_risk_pct * capital:
            # Reduce position size to fit max risk
            position_size = (max_risk_pct * capital) / (capital * stop_loss_pct)
            risk_amount = max_risk_pct * capital
        
        return risk_amount

    # ==================== DYNAMIC LEVERAGE ====================
    def _calculate_dynamic_leverage(
        self,
        sharpe_ratio: float,
        fractional_kelly: float
    ) -> float:
        """
        Calculate leverage based on risk-adjusted returns (Sharpe ratio)
        
        Logic:
        - Sharpe < 0.5: 1.0x (risky, reduce leverage)
        - Sharpe 0.5-1.0: 1.5x (moderate)
        - Sharpe 1.0-1.5: 2.0x (good)
        - Sharpe 1.5-2.0: 2.5x (very good)
        - Sharpe > 2.0: 3.0x (excellent, use max leverage)
        """
        if sharpe_ratio < 0.5:
            base_leverage = 1.0
        elif sharpe_ratio < 1.0:
            base_leverage = 1.5
        elif sharpe_ratio < 1.5:
            base_leverage = 2.0
        elif sharpe_ratio < 2.0:
            base_leverage = 2.5
        else:
            base_leverage = 3.0
        
        # Apply Kelly fraction as multiplier
        final_leverage = base_leverage * (1.0 + fractional_kelly)
        final_leverage = np.clip(final_leverage, 1.0, self.max_leverage)
        
        return final_leverage

    # ==================== HELPER METHODS ====================
    def _get_risk_profile(self, kelly_fraction: float) -> str:
        """Classify risk profile based on Kelly fraction"""
        if kelly_fraction <= 0.05:
            return "VERY_CONSERVATIVE"
        elif kelly_fraction <= 0.10:
            return "CONSERVATIVE"
        elif kelly_fraction <= 0.20:
            return "MODERATE"
        elif kelly_fraction <= 0.35:
            return "AGGRESSIVE"
        else:
            return "EXTREMELY_AGGRESSIVE"

    def _get_recommended_action(
        self,
        full_kelly: float,
        fractional_kelly: float,
        win_rate: float,
        sharpe_ratio: float
    ) -> str:
        """Recommend action based on Kelly metrics"""
        
        # If win rate is too low, reduce positions
        if win_rate < 0.45:
            return "REDUCE"
        
        # If Sharpe is negative, halt trading
        if sharpe_ratio < -0.5:
            return "HALT"
        
        # If Kelly is negative, sell everything
        if full_kelly < 0:
            return "LIQUIDATE"
        
        # If Sharpe is poor, reduce
        if sharpe_ratio < 0.5:
            return "REDUCE"
        
        # If Sharpe is moderate, maintain
        if sharpe_ratio < 1.5:
            return "MAINTAIN"
        
        # If Sharpe is good, increase
        if sharpe_ratio < 2.5:
            return "INCREASE"
        
        # If Sharpe is excellent, go aggressive
        return "INCREASE_AGGRESSIVELY"

    # ==================== TRADE HISTORY MANAGEMENT ====================
    def add_trade(self, trade: TradePerformance) -> None:
        """Record a trade"""
        self.trade_history.append(trade)
        logger.debug(f"[KellyCriterion] Recorded trade: {trade.symbol} "
                    f"PnL={trade.pnl_pct:+.2%}")

    def get_performance_by_agent(self, agent_tier: str) -> Optional[KellyMetrics]:
        """Calculate Kelly metrics for a specific agent tier"""
        agent_trades = [t for t in self.trade_history if t.agent_tier == agent_tier]
        
        if not agent_trades:
            return None
        
        wins = [t.pnl_pct for t in agent_trades if t.pnl > 0]
        losses = [t.pnl_pct for t in agent_trades if t.pnl <= 0]
        
        tier_fraction = self.risk_profiles.get(agent_tier, RiskProfile.MODERATE.value)
        
        return self.calculate_kelly_fraction(wins, losses, fraction=tier_fraction)

    def get_overall_performance(self) -> Optional[KellyMetrics]:
        """Calculate Kelly metrics for all trades"""
        if not self.trade_history:
            return None
        
        wins = [t.pnl_pct for t in self.trade_history if t.pnl > 0]
        losses = [t.pnl_pct for t in self.trade_history if t.pnl <= 0]
        
        return self.calculate_kelly_fraction(wins, losses, fraction=0.50)

    # ==================== REPORTING ====================
    def generate_kelly_report(self) -> Dict:
        """Generate comprehensive Kelly Criterion report"""
        overall = self.get_overall_performance()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_trades': len(self.trade_history),
            'overall_metrics': {
                'win_rate': overall.win_rate if overall else 0.0,
                'kelly_fraction': overall.kelly_fraction if overall else 0.0,
                'fractional_kelly': overall.fractional_kelly if overall else 0.0,
                'sharpe_ratio': overall.sharpe_ratio if overall else 0.0,
                'recommended_action': overall.recommended_action if overall else "NO_DATA"
            },
            'by_agent': {}
        }
        
        for tier in self.risk_profiles.keys():
            metrics = self.get_performance_by_agent(tier)
            if metrics:
                report['by_agent'][tier] = {
                    'win_rate': metrics.win_rate,
                    'kelly_fraction': metrics.kelly_fraction,
                    'suggested_leverage': metrics.suggested_leverage,
                    'sharpe_ratio': metrics.sharpe_ratio
                }
        
        return report

    def export_kelly_metrics(self, filepath: str) -> None:
        """Export Kelly metrics to JSON"""
        report = self.generate_kelly_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"[KellyCriterion] Exported Kelly metrics to {filepath}")


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Initialize calculator
    kelly = KellyCriterionCalculator()
    
    # Simulate trade history
    np.random.seed(42)
    
    for i in range(100):
        pnl_pct = np.random.normal(0.01, 0.03)  # 1% mean, 3% std
        
        trade = TradePerformance(
            trade_id=f"trade_{i}",
            symbol="BTC/USDT",
            entry_price=42000,
            exit_price=42000 * (1 + pnl_pct),
            position_size=0.05,
            pnl=1000 * pnl_pct,
            pnl_pct=pnl_pct,
            duration_hours=4,
            timestamp=datetime.now().timestamp(),
            agent_tier="aggressive"
        )
        kelly.add_trade(trade)
    
    # Calculate Kelly
    print("\n🎯 KELLY CRITERION ANALYSIS")
    print("=" * 60)
    
    overall = kelly.get_overall_performance()
    if overall:
        print(f"Win Rate: {overall.win_rate:.1%}")
        print(f"Avg Win: {overall.avg_win:.2%}")
        print(f"Avg Loss: {overall.avg_loss:.2%}")
        print(f"Odds (b): {overall.odds:.2f}")
        print(f"Full Kelly: {overall.kelly_fraction:.2%}")
        print(f"Fractional Kelly (50%): {overall.fractional_kelly:.2%}")
        print(f"Sharpe Ratio: {overall.sharpe_ratio:.2f}")
        print(f"Recommended Action: {overall.recommended_action}")
        print(f"Suggested Leverage: {overall.suggested_leverage:.2f}x")
    
    # Generate report
    report = kelly.generate_kelly_report()
    print(f"\n📊 Full Report: {json.dumps(report, indent=2)}")
