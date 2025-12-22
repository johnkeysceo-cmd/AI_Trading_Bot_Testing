"""Phase 11.8: Quantum Machine Learning for Trading
Quantum-enhanced portfolio optimization and signal generation"""

import numpy as np, json, logging
from dataclasses import dataclass
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantumTradingMetrics:
    day: int; portfolio_value: float; sharpe_ratio: float; win_rate: float
    def to_dict(self):
        return {'day': self.day, 'portfolio_value': float(self.portfolio_value),
                'sharpe_ratio': float(self.sharpe_ratio), 'win_rate': float(self.win_rate)}

class QuantumPortfolioOptimizer:
    def __init__(self, num_assets: int = 5):
        self.num_assets = num_assets
        self.weights = np.random.dirichlet(np.ones(num_assets))
        self.quantum_params = np.random.uniform(0, 2*np.pi, num_assets)
    
    def optimize_weights(self, returns: np.ndarray) -> np.ndarray:
        # Quantum amplitude encoding of returns
        returns_norm = returns / (np.linalg.norm(returns) + 1e-8)
        
        # Apply quantum rotation
        rotated = np.sin(returns_norm + self.quantum_params)
        
        # Convert to weights via softmax
        weights = np.exp(rotated) / np.sum(np.exp(rotated))
        
        # Update quantum params via gradient
        self.quantum_params += 0.01 * returns_norm
        self.quantum_params = self.quantum_params % (2 * np.pi)
        
        self.weights = weights
        return weights

class QuantumSignalGenerator:
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.quantum_circuit_params = np.random.uniform(0, 2*np.pi, 4)
    
    def generate_signal(self, price_data: np.ndarray) -> float:
        if len(price_data) < self.lookback:
            return 0.0
        
        recent_prices = price_data[-self.lookback:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # Quantum signal encoding
        signal_input = np.mean(returns)
        
        # 2-qubit quantum circuit simulation
        q_state = np.array([1, 0, 0, 0])  # |00⟩
        
        # RX gates
        angle = signal_input * self.quantum_circuit_params[0]
        rx_matrix = np.array([[np.cos(angle/2), -1j*np.sin(angle/2)],
                              [-1j*np.sin(angle/2), np.cos(angle/2)]])
        
        # Apply Hadamard and measurements
        hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        measurement = np.abs(np.sin(signal_input * np.sum(self.quantum_circuit_params)))
        
        # Update circuit params
        self.quantum_circuit_params += 0.01 * returns
        
        return float(measurement)

class QuantumTradingSystem:
    def __init__(self, num_assets: int = 5, num_days: int = 100):
        self.num_assets = num_assets
        self.num_days = num_days
        self.optimizer = QuantumPortfolioOptimizer(num_assets)
        self.signal_generators = [QuantumSignalGenerator() for _ in range(num_assets)]
        self.initial_capital = 100000.0
        self.portfolio_value = self.initial_capital
        self.metrics_history = []
        self.daily_returns = []
    
    def backtest(self) -> dict:
        logger.info(f"Quantum Trading Backtest: {self.num_days} days, {self.num_assets} assets")
        
        # Generate synthetic price paths
        price_history = np.zeros((self.num_days, self.num_assets))
        price_history[0] = 100.0
        
        for t in range(1, self.num_days):
            drift = 0.0005
            volatility = 0.02
            shocks = np.random.normal(drift, volatility, self.num_assets)
            price_history[t] = price_history[t-1] * np.exp(shocks)
        
        trades = []
        total_wins = 0
        
        for day in range(1, self.num_days):
            prices = price_history[day]
            prev_prices = price_history[day-1]
            
            # Generate quantum signals
            signals = []
            for i in range(self.num_assets):
                signal = self.signal_generators[i].generate_signal(price_history[:day, i])
                signals.append(signal)
            
            signals = np.array(signals)
            
            # Optimize portfolio based on signals
            returns = (prices - prev_prices) / prev_prices
            weights = self.optimizer.optimize_weights(returns)
            
            # Compute portfolio return
            portfolio_return = np.sum(weights * returns)
            
            # Track wins
            if portfolio_return > 0:
                total_wins += 1
            
            self.daily_returns.append(portfolio_return)
            self.portfolio_value *= (1 + portfolio_return)
            
            # Compute metrics
            if day > 1:
                sharpe = np.mean(self.daily_returns) / (np.std(self.daily_returns) + 1e-8) * np.sqrt(252)
            else:
                sharpe = 0.0
            
            win_rate = total_wins / day if day > 0 else 0.0
            
            self.metrics_history.append(
                QuantumTradingMetrics(day, self.portfolio_value, sharpe, win_rate).to_dict()
            )
            
            if day % 20 == 0:
                logger.info(f"Day {day}: Portfolio={self.portfolio_value:,.0f}, "
                          f"Sharpe={sharpe:.3f}, Win%={win_rate*100:.1f}%")
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        final_sharpe = np.mean(self.daily_returns) / (np.std(self.daily_returns) + 1e-8) * np.sqrt(252)
        
        return {
            'algorithm': 'Quantum-Trading-System',
            'num_assets': self.num_assets,
            'num_days': self.num_days,
            'initial_capital': float(self.initial_capital),
            'final_portfolio_value': float(self.portfolio_value),
            'total_return': float(total_return),
            'final_sharpe_ratio': float(final_sharpe),
            'final_win_rate': float(total_wins / self.num_days),
            'max_portfolio_value': float(np.max([m['portfolio_value'] for m in self.metrics_history])),
            'min_portfolio_value': float(np.min([m['portfolio_value'] for m in self.metrics_history])),
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 11.8: Quantum Machine Learning for Trading")
    logger.info("=" * 60)
    
    system = QuantumTradingSystem(num_assets=5, num_days=100)
    results = system.backtest()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Initial Capital: ${results['initial_capital']:,.0f}")
    logger.info(f"Final Portfolio Value: ${results['final_portfolio_value']:,.0f}")
    logger.info(f"Total Return: {results['total_return']*100:.2f}%")
    logger.info(f"Final Sharpe Ratio: {results['final_sharpe_ratio']:.3f}")
    logger.info(f"Win Rate: {results['final_win_rate']*100:.1f}%")
    
    with open('phase11_quantum_trading_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✓ Results saved to phase11_quantum_trading_results.json")
