"""
Phase 9.8: Federated Learning for Distributed Trading
Applies federated learning to cryptocurrency trading with distributed agents:
- Multi-agent model sharing
- Portfolio-level aggregation
- Trading signal consensus
- Distributed risk management
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
import logging
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Signal from a trading agent"""
    agent_id: str
    symbol: str
    action: str  # "buy", "sell", "hold"
    confidence: float
    position_size: float
    entry_price: float
    timestamp: float
    

@dataclass
class TradingMetrics:
    """Metrics for federated trading"""
    round_num: int
    num_agents: int
    total_trades: int
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    portfolio_value: float
    diversification_index: float
    signal_consensus: float
    
    def to_dict(self):
        return {
            'round': self.round_num,
            'num_agents': self.num_agents,
            'total_trades': self.total_trades,
            'win_rate': float(self.win_rate),
            'avg_return': float(self.avg_return),
            'sharpe_ratio': float(self.sharpe_ratio),
            'portfolio_value': float(self.portfolio_value),
            'diversification_index': float(self.diversification_index),
            'signal_consensus': float(self.signal_consensus)
        }


@dataclass
class FederatedTradingConfig:
    """Configuration for federated trading"""
    num_agents: int = 5
    num_trading_rounds: int = 50
    symbols: List[str] = field(default_factory=lambda: ['BTC/USDT', 'ETH/USDT', 'XRP/USDT'])
    initial_capital: float = 100000.0
    position_size_factor: float = 0.1  # 10% of capital per trade
    local_training_episodes: int = 5
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    consensus_threshold: float = 0.6
    risk_limit: float = 0.02  # Max 2% loss per trade
    

class TradingAgent:
    """Individual trading agent in federated system"""
    
    def __init__(self, agent_id: str, symbol: str, config: FederatedTradingConfig):
        self.agent_id = agent_id
        self.symbol = symbol
        self.config = config
        
        # Model parameters
        self.model_weights = np.random.normal(0, 0.1, 50)
        
        # Trading state
        self.position: Optional[Dict] = None  # Current open position
        self.pnl_history: List[float] = []
        self.trades: List[Dict] = []
        self.signal_history: List[TradingSignal] = []
        
        # Price data
        self.price_history: List[float] = []
        self.current_price: float = 100.0  # Starting price
        
    def generate_price_update(self, return_pct: float):
        """Simulate price movement"""
        self.current_price *= (1.0 + return_pct)
        self.price_history.append(self.current_price)
    
    def compute_trading_signal(self, market_features: np.ndarray) -> TradingSignal:
        """Generate trading signal based on market features"""
        
        # Linear combination of features for signal
        signal_strength = np.dot(self.model_weights[:len(market_features)], market_features)
        
        # Confidence from sigmoid
        confidence = 1.0 / (1.0 + np.exp(-signal_strength))
        
        # Determine action
        if confidence > 0.6:
            action = "buy" if signal_strength > 0 else "sell"
        else:
            action = "hold"
        
        # Position size (as fraction of capital)
        position_size = self.config.position_size_factor * abs(signal_strength)
        
        signal = TradingSignal(
            agent_id=self.agent_id,
            symbol=self.symbol,
            action=action,
            confidence=confidence,
            position_size=position_size,
            entry_price=self.current_price,
            timestamp=datetime.now().timestamp()
        )
        
        self.signal_history.append(signal)
        return signal
    
    def execute_trade(self, signal: TradingSignal, portfolio_allocation: float):
        """Execute trade based on signal"""
        
        if signal.action == "hold" or not portfolio_allocation > 0:
            return None
        
        position_value = portfolio_allocation * signal.position_size
        
        position = {
            'agent_id': self.agent_id,
            'symbol': self.symbol,
            'action': signal.action,
            'entry_price': self.current_price,
            'position_size': position_value,
            'entry_time': datetime.now().timestamp(),
            'entry_qty': position_value / self.current_price
        }
        
        self.position = position
        return position
    
    def close_trade(self) -> Optional[Dict]:
        """Close open position and compute PnL"""
        
        if not self.position:
            return None
        
        # Compute exit value
        exit_price = self.current_price
        exit_value = self.position['entry_qty'] * exit_price
        entry_value = self.position['entry_qty'] * self.position['entry_price']
        
        pnl = exit_value - entry_value
        pnl_pct = pnl / entry_value if entry_value > 0 else 0
        
        trade_result = {
            'agent_id': self.agent_id,
            'symbol': self.symbol,
            'action': self.position['action'],
            'entry_price': self.position['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'duration': datetime.now().timestamp() - self.position['entry_time']
        }
        
        self.trades.append(trade_result)
        self.pnl_history.append(pnl)
        self.position = None
        
        return trade_result
    
    def update_model(self, gradient: np.ndarray, learning_rate: float):
        """Update model weights with gradient"""
        self.model_weights -= learning_rate * gradient


class FederatedTradingServer:
    """Central server for federated trading system"""
    
    def __init__(self, config: FederatedTradingConfig):
        self.config = config
        self.global_model_weights = np.random.normal(0, 0.1, 50)
        self.agents: Dict[str, List[TradingAgent]] = defaultdict(list)
        self.trading_metrics_history: List[TradingMetrics] = []
        self.consensus_signals: List[Dict] = []
        
    def aggregate_model_updates(self, agent_updates: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """Aggregate trading model updates from all agents"""
        
        if not agent_updates:
            return self.global_model_weights.copy()
        
        # Weighted average by agent performance
        total_weight = len(agent_updates)
        aggregated = np.zeros_like(self.global_model_weights)
        
        for agent_id, weights in agent_updates:
            aggregated += weights
        
        aggregated /= total_weight
        
        self.global_model_weights = aggregated
        return aggregated
    
    def aggregate_trading_signals(self, signals: List[TradingSignal]) -> Dict:
        """Aggregate trading signals from multiple agents"""
        
        if not signals:
            return {'action': 'hold', 'consensus': 0.0}
        
        # Count votes
        buy_votes = sum(1 for s in signals if s.action == "buy")
        sell_votes = sum(1 for s in signals if s.action == "sell")
        hold_votes = len(signals) - buy_votes - sell_votes
        
        # Consensus action
        total = len(signals)
        max_votes = max(buy_votes, sell_votes, hold_votes)
        
        if buy_votes > sell_votes:
            consensus_action = "buy"
            consensus = buy_votes / total
        elif sell_votes > buy_votes:
            consensus_action = "sell"
            consensus = sell_votes / total
        else:
            consensus_action = "hold"
            consensus = hold_votes / total
        
        # Average confidence
        avg_confidence = np.mean([s.confidence for s in signals])
        
        # Average position size
        avg_position_size = np.mean([s.position_size for s in signals])
        
        return {
            'action': consensus_action,
            'consensus': consensus,
            'avg_confidence': avg_confidence,
            'avg_position_size': avg_position_size,
            'num_signals': len(signals)
        }


class FederatedTradingCoordinator:
    """Coordinates federated trading system"""
    
    def __init__(self, config: FederatedTradingConfig):
        self.config = config
        self.server = FederatedTradingServer(config)
        
        # Create agents for each symbol
        for symbol in config.symbols:
            for i in range(config.num_agents):
                agent = TradingAgent(f"agent_{i}_{symbol}", symbol, config)
                self.server.agents[symbol].append(agent)
        
        self.portfolio_value_history: List[float] = []
        self.overall_pnl: float = 0.0
        
    def run_federated_trading(self) -> Dict:
        """Execute federated trading system"""
        
        logger.info(f"Starting federated trading system")
        logger.info(f"Agents: {self.config.num_agents}, Symbols: {len(self.config.symbols)}, "
                   f"Initial capital: ${self.config.initial_capital:.2f}")
        
        for round_num in range(self.config.num_trading_rounds):
            logger.info(f"\n--- Trading Round {round_num + 1}/{self.config.num_trading_rounds} ---")
            
            # Step 1: Simulate price movements
            for symbol in self.config.symbols:
                market_return = np.random.normal(0.0001, 0.01)  # Realistic daily return
                for agent in self.server.agents[symbol]:
                    agent.generate_price_update(market_return)
            
            # Step 2: Generate signals and aggregate
            round_signals = defaultdict(list)
            round_trades = []
            
            for symbol in self.config.symbols:
                for agent in self.server.agents[symbol]:
                    # Generate signal
                    market_features = np.random.normal(0, 1, 50)
                    signal = agent.compute_trading_signal(market_features)
                    round_signals[symbol].append(signal)
                
                # Aggregate signals
                consensus = self.server.aggregate_trading_signals(round_signals[symbol])
                
                # Execute consensus trade
                portfolio_allocation = self.config.initial_capital / len(self.config.symbols)
                
                if consensus['action'] != 'hold' and consensus['consensus'] >= self.config.consensus_threshold:
                    for agent in self.server.agents[symbol]:
                        # Create consensus signal for execution
                        consensus_signal = TradingSignal(
                            agent_id="consensus",
                            symbol=symbol,
                            action=consensus['action'],
                            confidence=consensus['avg_confidence'],
                            position_size=consensus['avg_position_size'],
                            entry_price=agent.current_price,
                            timestamp=datetime.now().timestamp()
                        )
                        trade = agent.execute_trade(consensus_signal, portfolio_allocation)
                        if trade:
                            round_trades.append(trade)
                
                self.server.consensus_signals.append({
                    'round': round_num + 1,
                    'symbol': symbol,
                    'consensus': consensus
                })
            
            # Step 3: Close trades (example: close after 1 round)
            round_pnls = []
            for symbol in self.config.symbols:
                for agent in self.server.agents[symbol]:
                    if agent.position:
                        result = agent.close_trade()
                        if result:
                            round_pnls.append(result['pnl'])
            
            # Step 4: Collect metrics
            total_pnl = np.sum(round_pnls) if round_pnls else 0
            self.overall_pnl += total_pnl
            portfolio_value = self.config.initial_capital + self.overall_pnl
            
            # Win rate
            if round_pnls:
                wins = sum(1 for p in round_pnls if p > 0)
                win_rate = wins / len(round_pnls)
            else:
                win_rate = 0.5
            
            # Sharpe ratio (simplified)
            all_pnls = []
            for symbol in self.config.symbols:
                for agent in self.server.agents[symbol]:
                    all_pnls.extend(agent.pnl_history[-5:])
            
            if all_pnls:
                avg_return = np.mean(all_pnls)
                return_std = np.std(all_pnls)
                sharpe = (avg_return / (return_std + 1e-6)) if return_std > 0 else 0
            else:
                avg_return = 0
                sharpe = 0
            
            # Diversification index
            symbol_returns = []
            for symbol in self.config.symbols:
                symbol_pnl = 0
                for agent in self.server.agents[symbol]:
                    symbol_pnl += np.sum(agent.pnl_history[-1:]) if agent.pnl_history else 0
                symbol_returns.append(symbol_pnl)
            
            diversification = len(self.config.symbols) / (1 + np.std(symbol_returns) + 1e-6)
            
            # Signal consensus
            consensus_scores = [c['consensus']['consensus'] for c in self.server.consensus_signals[-len(self.config.symbols):]]
            avg_consensus = np.mean(consensus_scores) if consensus_scores else 0.5
            
            metrics = TradingMetrics(
                round_num=round_num + 1,
                num_agents=self.config.num_agents * len(self.config.symbols),
                total_trades=len(round_trades),
                win_rate=win_rate,
                avg_return=avg_return,
                sharpe_ratio=sharpe,
                portfolio_value=portfolio_value,
                diversification_index=diversification,
                signal_consensus=avg_consensus
            )
            
            self.server.trading_metrics_history.append(metrics)
            self.portfolio_value_history.append(portfolio_value)
            
            logger.info(f"Round {round_num + 1}: Portfolio=${portfolio_value:.2f}, "
                       f"Win rate={win_rate:.1%}, Sharpe={sharpe:.2f}, "
                       f"Consensus={avg_consensus:.2%}")
        
        return self.get_final_results()
    
    def get_final_results(self) -> Dict:
        """Compile final results"""
        
        returns = np.diff(self.portfolio_value_history)
        total_return = (self.portfolio_value_history[-1] - self.config.initial_capital) / self.config.initial_capital
        
        return {
            'algorithm': 'Federated-Trading-System',
            'num_agents': self.config.num_agents,
            'symbols': self.config.symbols,
            'num_rounds': len(self.server.trading_metrics_history),
            'initial_capital': float(self.config.initial_capital),
            'final_portfolio_value': float(self.portfolio_value_history[-1]) if self.portfolio_value_history else self.config.initial_capital,
            'total_return': float(total_return),
            'avg_sharpe_ratio': float(np.mean([m.sharpe_ratio for m in self.server.trading_metrics_history])),
            'avg_win_rate': float(np.mean([m.win_rate for m in self.server.trading_metrics_history])),
            'avg_signal_consensus': float(np.mean([m.signal_consensus for m in self.server.trading_metrics_history])),
            'trading_metrics': [m.to_dict() for m in self.server.trading_metrics_history],
            'portfolio_value_history': self.portfolio_value_history
        }


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 9.8: Federated Learning for Distributed Trading")
    logger.info("=" * 60)
    
    config = FederatedTradingConfig(
        num_agents=5,
        num_trading_rounds=40,
        symbols=['BTC/USDT', 'ETH/USDT', 'XRP/USDT'],
        initial_capital=100000.0,
        position_size_factor=0.1,
        learning_rate=0.01,
        consensus_threshold=0.6,
        risk_limit=0.02
    )
    
    coordinator = FederatedTradingCoordinator(config)
    results = coordinator.run_federated_trading()
    
    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Algorithm: {results['algorithm']}")
    logger.info(f"Initial capital: ${results['initial_capital']:.2f}")
    logger.info(f"Final portfolio: ${results['final_portfolio_value']:.2f}")
    logger.info(f"Total return: {results['total_return']:.2%}")
    logger.info(f"Avg Sharpe ratio: {results['avg_sharpe_ratio']:.2f}")
    logger.info(f"Avg win rate: {results['avg_win_rate']:.1%}")
    logger.info(f"Avg signal consensus: {results['avg_signal_consensus']:.2%}")
    logger.info(f"Number of agents: {results['num_agents']}")
    logger.info(f"Symbols traded: {', '.join(results['symbols'])}")
    
    # Save results
    with open('phase9_federated_trading_results.json', 'w') as f:
        json.dump(results, f, indent=2)
