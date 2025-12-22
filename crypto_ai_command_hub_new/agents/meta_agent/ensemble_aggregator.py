"""
ensemble_aggregator.py
-----------------------
Ensemble Signal Aggregation System for Multi-Agent Trading

Features:
- Collect signals from all 4 agent types (aggressive, conservative, nuclear, arbitrage)
- Majority voting system with confidence weighting
- Conflict detection and resolution
- Correlation analysis for systemic risk detection
- Adaptive weighting based on agent performance

Phase 1: Signal Fusion & Positioning - Improvement: 3/10 → 5/10
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

# Logging setup
logger = logging.getLogger("EnsembleAggregator")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class SignalType(Enum):
    """Signal classification"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class AgentSignal:
    """Individual agent signal"""
    agent_id: str
    agent_tier: str  # 'survival', 'aggressive', 'nuclear', 'arbitrage'
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 - 1.0
    strength: float  # -1.0 to 1.0 (-1=strong sell, 1=strong buy)
    timestamp: float
    reasoning: str = ""
    risk_score: float = 0.5  # 0.0=low risk, 1.0=high risk


@dataclass
class ConsensusSignal:
    """Aggregated consensus from all agents"""
    symbol: str
    final_signal: SignalType
    consensus_strength: float  # 0.0 - 1.0
    vote_count: Dict[str, int]  # {BUY: 2, SELL: 1, HOLD: 1}
    divergence_score: float  # 0.0=perfect agreement, 1.0=total disagreement
    weighted_confidence: float  # Weighted by agent performance
    contributing_agents: List[str]
    timestamp: float
    execution_recommended: bool
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'


class SignalAggregator:
    """
    Main aggregation engine for multi-agent signal fusion
    """

    def __init__(self, num_agents: int = 4):
        self.num_agents = num_agents
        self.agent_signals: Dict[str, AgentSignal] = {}
        self.consensus_history: List[ConsensusSignal] = []
        
        # Agent performance tracking for adaptive weighting
        self.agent_performance: Dict[str, Dict] = {
            'survival': {'wins': 0, 'losses': 0, 'weight': 0.20},
            'aggressive': {'wins': 0, 'losses': 0, 'weight': 0.25},
            'nuclear': {'wins': 0, 'losses': 0, 'weight': 0.15},
            'arbitrage': {'wins': 0, 'losses': 0, 'weight': 0.25}
        }
        
        # Correlation matrix: track how often agents agree
        self.signal_correlation_matrix: Dict[Tuple[str, str], float] = {}
        
        logger.info(f"[EnsembleAggregator] Initialized with {num_agents} agents")

    # ==================== SIGNAL COLLECTION ====================
    def add_signal(self, signal: AgentSignal) -> None:
        """
        Add a signal from an agent
        """
        self.agent_signals[signal.agent_id] = signal
        logger.debug(f"[EnsembleAggregator] Received signal from {signal.agent_id}: "
                    f"{signal.signal_type.name} ({signal.confidence:.2%}) for {signal.symbol}")

    def add_signals(self, signals: List[AgentSignal]) -> None:
        """
        Add multiple signals at once
        """
        for signal in signals:
            self.add_signal(signal)

    # ==================== VOTING MECHANISM ====================
    def compute_vote_strength(self, signals: List[AgentSignal]) -> Tuple[float, Dict[str, int]]:
        """
        Compute majority voting strength from signals
        
        Returns:
            - vote_strength: -1.0 (strong sell) to 1.0 (strong buy)
            - vote_counts: {BUY: count, SELL: count, HOLD: count}
        """
        votes = {SignalType.BUY: [], SignalType.SELL: [], SignalType.HOLD: []}
        vote_counts = {SignalType.BUY: 0, SignalType.SELL: 0, SignalType.HOLD: 0}
        
        # Collect votes with weights
        for signal in signals:
            votes[signal.signal_type].append(signal.confidence)
            vote_counts[signal.signal_type] += 1
        
        # Calculate weighted vote strength
        buy_strength = np.sum(votes[SignalType.BUY]) if votes[SignalType.BUY] else 0
        sell_strength = np.sum(votes[SignalType.SELL]) if votes[SignalType.SELL] else 0
        
        total_votes = len(signals)
        if total_votes == 0:
            return 0.0, {SignalType.BUY: 0, SignalType.SELL: 0, SignalType.HOLD: 0}
        
        # Net vote strength (-1 to 1)
        vote_strength = (buy_strength - sell_strength) / total_votes
        vote_strength = np.clip(vote_strength, -1.0, 1.0)
        
        return vote_strength, vote_counts

    def determine_consensus(self, vote_strength: float, divergence: float) -> SignalType:
        """
        Determine final signal based on vote strength and consensus level
        
        Logic:
        - Strong buy (>0.5) with high agreement (divergence < 0.3) → BUY
        - Strong sell (<-0.5) with high agreement → SELL
        - Weak or divergent → HOLD
        """
        if divergence > 0.6:  # High divergence = low confidence
            return SignalType.HOLD
        
        if abs(vote_strength) < 0.3:
            return SignalType.HOLD
        
        if vote_strength > 0.3:
            return SignalType.BUY
        elif vote_strength < -0.3:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    # ==================== DIVERGENCE & RISK ====================
    def calculate_divergence_score(self, signals: List[AgentSignal]) -> float:
        """
        Calculate how much agents disagree (0.0 = perfect agreement, 1.0 = total disagreement)
        
        Using Shannon entropy:
        - All agents agree → entropy = 0.0
        - Mixed signals → entropy = 1.0 (normalized)
        """
        if not signals:
            return 0.0
        
        vote_counts = {}
        for signal in signals:
            vote_counts[signal.signal_type] = vote_counts.get(signal.signal_type, 0) + 1
        
        # Normalize to probabilities
        probabilities = np.array([count / len(signals) for count in vote_counts.values()])
        
        # Shannon entropy (normalized to 0-1)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        max_entropy = np.log2(3)  # 3 possible signals
        
        divergence = entropy / max_entropy
        return float(divergence)

    def calculate_risk_level(self, signals: List[AgentSignal], divergence: float) -> str:
        """
        Assess overall risk level
        
        Factors:
        - Agent disagreement (high divergence = risky)
        - Average risk score from agents
        - Extreme signals from nuclear/aggressive agents
        """
        if divergence > 0.7:
            return "CRITICAL"
        
        avg_risk = np.mean([s.risk_score for s in signals])
        
        if avg_risk > 0.75 or divergence > 0.6:
            return "HIGH"
        elif avg_risk > 0.5 or divergence > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    # ==================== MAIN AGGREGATION ====================
    def aggregate_signals(self, symbol: str) -> Optional[ConsensusSignal]:
        """
        Main aggregation pipeline
        
        Process:
        1. Collect signals for symbol
        2. Compute vote strength
        3. Calculate divergence
        4. Apply adaptive weighting
        5. Determine consensus
        6. Check for conflicts
        """
        # Get all signals for this symbol
        symbol_signals = [s for s in self.agent_signals.values() if s.symbol == symbol]
        
        if not symbol_signals:
            logger.warning(f"[EnsembleAggregator] No signals received for {symbol}")
            return None
        
        # Step 1: Compute raw vote strength
        vote_strength, vote_counts = self.compute_vote_strength(symbol_signals)
        
        # Step 2: Calculate divergence
        divergence = self.calculate_divergence_score(symbol_signals)
        
        # Step 3: Apply adaptive weighting based on agent performance
        weighted_signals = self._apply_agent_weighting(symbol_signals)
        weighted_vote_strength, _ = self.compute_vote_strength(weighted_signals)
        
        # Step 4: Determine final signal
        final_signal = self.determine_consensus(weighted_vote_strength, divergence)
        
        # Step 5: Calculate consensus strength
        consensus_strength = 1.0 - divergence if divergence < 0.7 else 0.0
        
        # Step 6: Calculate weighted confidence
        weighted_confidence = np.mean([s.confidence for s in weighted_signals])
        
        # Step 7: Assess risk
        risk_level = self.calculate_risk_level(symbol_signals, divergence)
        
        # Step 8: Determine if execution is recommended
        execution_recommended = (
            final_signal != SignalType.HOLD and
            consensus_strength > 0.5 and
            weighted_confidence > 0.6 and
            risk_level != "CRITICAL"
        )
        
        # Create consensus signal
        consensus = ConsensusSignal(
            symbol=symbol,
            final_signal=final_signal,
            consensus_strength=consensus_strength,
            vote_count={
                'BUY': vote_counts.get(SignalType.BUY, 0),
                'SELL': vote_counts.get(SignalType.SELL, 0),
                'HOLD': vote_counts.get(SignalType.HOLD, 0)
            },
            divergence_score=divergence,
            weighted_confidence=weighted_confidence,
            contributing_agents=[s.agent_id for s in symbol_signals],
            timestamp=datetime.now().timestamp(),
            execution_recommended=execution_recommended,
            risk_level=risk_level
        )
        
        # Store in history
        self.consensus_history.append(consensus)
        
        # Log aggregation
        logger.info(f"[EnsembleAggregator] Aggregated signal for {symbol}: "
                   f"{final_signal.name} (strength={consensus_strength:.2%}, "
                   f"confidence={weighted_confidence:.2%}, risk={risk_level}, "
                   f"execute={execution_recommended})")
        
        return consensus

    def _apply_agent_weighting(self, signals: List[AgentSignal]) -> List[AgentSignal]:
        """
        Re-weight signals based on agent win/loss history
        
        Better-performing agents have higher influence
        """
        weighted_signals = []
        
        for signal in signals:
            performance = self.agent_performance[signal.agent_tier]
            total_trades = performance['wins'] + performance['losses']
            
            if total_trades > 0:
                win_rate = performance['wins'] / total_trades
            else:
                win_rate = 0.5  # Default if no history
            
            # Adjust confidence based on win rate
            adjusted_confidence = signal.confidence * (0.5 + win_rate)
            
            # Create weighted signal
            weighted_signal = AgentSignal(
                agent_id=signal.agent_id,
                agent_tier=signal.agent_tier,
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                confidence=adjusted_confidence,
                strength=signal.strength,
                timestamp=signal.timestamp,
                reasoning=signal.reasoning,
                risk_score=signal.risk_score
            )
            weighted_signals.append(weighted_signal)
        
        return weighted_signals

    # ==================== PERFORMANCE TRACKING ====================
    def update_agent_performance(self, agent_tier: str, outcome: str) -> None:
        """
        Update agent performance metrics after trade execution
        
        outcome: 'WIN' or 'LOSS'
        """
        if outcome.upper() == 'WIN':
            self.agent_performance[agent_tier]['wins'] += 1
        else:
            self.agent_performance[agent_tier]['losses'] += 1
        
        # Recompute adaptive weights
        self._recompute_adaptive_weights()
        
        logger.debug(f"[EnsembleAggregator] Updated {agent_tier} performance: "
                    f"{self.agent_performance[agent_tier]}")

    def _recompute_adaptive_weights(self) -> None:
        """
        Dynamically adjust agent weights based on win rates
        
        Better performers get higher weights
        """
        total_weight = 0
        win_rates = {}
        
        for tier, performance in self.agent_performance.items():
            total = performance['wins'] + performance['losses']
            if total > 0:
                win_rates[tier] = performance['wins'] / total
            else:
                win_rates[tier] = 0.5
        
        # Normalize win rates to weights (sum to 1.0)
        total_wr = sum(win_rates.values())
        for tier in self.agent_performance.keys():
            self.agent_performance[tier]['weight'] = win_rates[tier] / total_wr

    def get_agent_weights(self) -> Dict[str, float]:
        """Return current adaptive weights for all agents"""
        return {tier: perf['weight'] for tier, perf in self.agent_performance.items()}

    # ==================== CONFLICT DETECTION ====================
    def detect_conflicting_signals(self, signals: List[AgentSignal]) -> bool:
        """
        Detect if agents are severely conflicted
        
        Conflict: Some agents BUY while others SELL strongly
        """
        divergence = self.calculate_divergence_score(signals)
        return divergence > 0.6

    def resolve_conflicts(self, signals: List[AgentSignal]) -> SignalType:
        """
        Resolve conflicting signals conservatively
        
        Strategy: Default to HOLD when high divergence
        """
        if self.detect_conflicting_signals(signals):
            logger.warning("[EnsembleAggregator] High conflict detected, defaulting to HOLD")
            return SignalType.HOLD
        
        vote_strength, _ = self.compute_vote_strength(signals)
        return self.determine_consensus(vote_strength, self.calculate_divergence_score(signals))

    # ==================== REPORTING ====================
    def get_consensus_summary(self, symbol: str) -> Optional[Dict]:
        """Get latest consensus summary for a symbol"""
        if not self.consensus_history:
            return None
        
        latest = [c for c in self.consensus_history if c.symbol == symbol]
        if not latest:
            return None
        
        consensus = latest[-1]
        return {
            'symbol': consensus.symbol,
            'signal': consensus.final_signal.name,
            'strength': consensus.consensus_strength,
            'confidence': consensus.weighted_confidence,
            'divergence': consensus.divergence_score,
            'risk': consensus.risk_level,
            'votes': consensus.vote_count,
            'agents': consensus.contributing_agents,
            'execute': consensus.execution_recommended
        }

    def export_consensus_history(self, filepath: str) -> None:
        """Export consensus history to JSON for analysis"""
        history = []
        for consensus in self.consensus_history:
            history.append({
                'timestamp': consensus.timestamp,
                'symbol': consensus.symbol,
                'signal': consensus.final_signal.name,
                'strength': consensus.consensus_strength,
                'confidence': consensus.weighted_confidence,
                'divergence': consensus.divergence_score,
                'risk': consensus.risk_level,
                'execute': consensus.execution_recommended
            })
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"[EnsembleAggregator] Exported {len(history)} consensus records to {filepath}")


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    from datetime import datetime
    
    # Initialize aggregator
    agg = SignalAggregator(num_agents=4)
    
    # Simulate 4 agents sending signals for BTC
    signals = [
        AgentSignal(
            agent_id="survival_agent",
            agent_tier="survival",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence=0.75,
            strength=0.6,
            timestamp=datetime.now().timestamp(),
            reasoning="RSI oversold, MACD bullish",
            risk_score=0.3
        ),
        AgentSignal(
            agent_id="aggressive_agent",
            agent_tier="aggressive",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence=0.85,
            strength=0.8,
            timestamp=datetime.now().timestamp(),
            reasoning="Momentum breakout confirmed",
            risk_score=0.6
        ),
        AgentSignal(
            agent_id="arbitrage_agent",
            agent_tier="arbitrage",
            symbol="BTC/USDT",
            signal_type=SignalType.HOLD,
            confidence=0.5,
            strength=0.0,
            timestamp=datetime.now().timestamp(),
            reasoning="No arbitrage opportunity",
            risk_score=0.2
        ),
        AgentSignal(
            agent_id="nuclear_agent",
            agent_tier="nuclear",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence=0.7,
            strength=0.9,
            timestamp=datetime.now().timestamp(),
            reasoning="Chart looks absolutely insane",
            risk_score=0.9
        )
    ]
    
    # Add signals
    agg.add_signals(signals)
    
    # Aggregate
    consensus = agg.aggregate_signals("BTC/USDT")
    
    # Print results
    if consensus:
        print(f"\n✅ CONSENSUS SIGNAL: {consensus.final_signal.name}")
        print(f"   Strength: {consensus.consensus_strength:.2%}")
        print(f"   Confidence: {consensus.weighted_confidence:.2%}")
        print(f"   Divergence: {consensus.divergence_score:.2%}")
        print(f"   Risk Level: {consensus.risk_level}")
        print(f"   Execute: {consensus.execution_recommended}")
        print(f"   Votes: {consensus.vote_count}")
