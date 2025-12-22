"""
PHASE 4: Quantum-Inspired Portfolio Optimization Engine

Advanced portfolio optimization using variational quantum algorithms,
classical heuristics, and hybrid quantum-classical approaches.

Provides real-time portfolio rebalancing with advanced risk metrics,
correlation analysis, and Pareto frontier exploration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import solve
import cvxpy as cp
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    RISK_PARITY = "risk_parity"
    EQUAL_RISK_CONTRIBUTION = "erc"
    QUANTUM_INSPIRED = "quantum_inspired"
    HIERARCHICAL = "hierarchical"
    ENTROPIC = "entropic"
    PARTICLE_SWARM = "pso"


class ConstraintType(Enum):
    """Constraint types for optimization."""
    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"
    LEVERAGE_LIMIT = "leverage_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    SECTOR_LIMITS = "sector_limits"
    FACTOR_LIMITS = "factor_limits"


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    herfindahl_index: float
    effective_number_assets: float
    turn_over: float
    expected_shortfall: float
    omega_ratio: float
    information_ratio: float
    tracking_error: float
    
    def __str__(self):
        return f"""
Portfolio Metrics:
  Return: {self.expected_return:.2%}
  Volatility: {self.volatility:.2%}
  Sharpe: {self.sharpe_ratio:.3f}
  Sortino: {self.sortino_ratio:.3f}
  VaR(95%): {self.var_95:.2%}
  CVaR(95%): {self.cvar_95:.2%}
  Max DD: {self.max_drawdown:.2%}
  Effective N: {self.effective_number_assets:.1f}
  Herfindahl: {self.herfindahl_index:.3f}
"""


@dataclass
class OptimizationResult:
    """Optimization result container."""
    weights: np.ndarray
    metrics: PortfolioMetrics
    optimization_time: float
    method_used: OptimizationMethod
    convergence_info: Dict
    constraint_violations: Dict = field(default_factory=dict)
    sensitivity_analysis: Dict = field(default_factory=dict)


class QuantumInspiredOptimizer:
    """
    Quantum-inspired portfolio optimization using variational algorithms.
    
    Implements hybrid quantum-classical optimization for portfolio construction
    using principles from quantum mechanics (superposition, entanglement) adapted
    to classical computation.
    """
    
    def __init__(self, n_assets: int, lookback_periods: int = 252):
        self.n_assets = n_assets
        self.lookback_periods = lookback_periods
        
        self.returns = None
        self.cov_matrix = None
        self.corr_matrix = None
        self.mean_returns = None
        self.volatilities = None
        
        self.scaler = StandardScaler()
        self.ledoit_wolf = LedoitWolf()
        
    def load_returns(self, returns: np.ndarray):
        """
        Load historical returns data.
        
        Args:
            returns: Shape (T, N) array of asset returns
        """
        self.returns = returns[-self.lookback_periods:].copy()
        
        # Calculate statistics
        self.mean_returns = np.mean(self.returns, axis=0)
        self.volatilities = np.std(self.returns, axis=0)
        
        # Robust covariance estimation
        self.cov_matrix, _ = self.ledoit_wolf.fit(self.returns)
        self.corr_matrix = np.corrcoef(self.returns.T)
        
        logger.info(f"Loaded returns: {self.returns.shape}, "
                   f"Condition number: {np.linalg.cond(self.cov_matrix):.2e}")
    
    def variational_quantum_optimizer(
        self,
        target_return: Optional[float] = None,
        risk_aversion: float = 1.0,
        max_iterations: int = 1000,
        learning_rate: float = 0.01,
        ansatz_depth: int = 3
    ) -> OptimizationResult:
        """
        Variational Quantum Eigensolver (VQE) inspired optimization.
        
        Uses parametrized quantum circuit analogy to find optimal portfolio weights.
        The "quantum circuit" is represented as a parameterized function that
        maps circuit parameters to portfolio weights.
        """
        
        # Initialize parameters (rotation angles in "quantum circuit")
        n_params = ansatz_depth * self.n_assets
        params = np.random.randn(n_params) * 0.1
        
        optimizer = VQEPortfolioOptimizer(
            returns=self.returns,
            cov_matrix=self.cov_matrix,
            mean_returns=self.mean_returns,
            target_return=target_return,
            risk_aversion=risk_aversion,
            ansatz_depth=ansatz_depth
        )
        
        best_loss = float('inf')
        best_params = params.copy()
        losses = []
        
        for iteration in range(max_iterations):
            # Compute gradient via parameter shift rule
            gradient = np.zeros_like(params)
            shift = np.pi / 4
            
            loss_plus_list = []
            loss_minus_list = []
            
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += shift
                loss_plus = optimizer.evaluate(params_plus)
                loss_plus_list.append(loss_plus)
                
                params_minus = params.copy()
                params_minus[i] -= shift
                loss_minus = optimizer.evaluate(params_minus)
                loss_minus_list.append(loss_minus)
                
                gradient[i] = (loss_plus - loss_minus) / (2 * np.sin(shift))
            
            # Update parameters
            params = params - learning_rate * gradient
            
            # Evaluate current loss
            loss = optimizer.evaluate(params)
            losses.append(loss)
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
            
            if iteration % 100 == 0:
                logger.debug(f"VQE Iteration {iteration}: Loss = {loss:.6f}")
            
            # Adaptive learning rate
            if iteration > 0 and losses[-1] > losses[-2]:
                learning_rate *= 0.99
        
        # Decode final weights
        weights = optimizer.decode_weights(best_params)
        weights = weights / np.sum(np.abs(weights))  # Normalize
        weights = np.clip(weights, -1, 1)  # Clip to [-1, 1]
        
        metrics = self._calculate_metrics(weights)
        
        return OptimizationResult(
            weights=weights,
            metrics=metrics,
            optimization_time=0.0,
            method_used=OptimizationMethod.QUANTUM_INSPIRED,
            convergence_info={
                'final_loss': best_loss,
                'iterations': max_iterations,
                'losses': losses,
                'final_gradient_norm': np.linalg.norm(gradient)
            }
        )
    
    def qaoa_inspired_optimizer(
        self,
        max_iterations: int = 1000,
        population_size: int = 50,
        mixing_strength: float = 0.5
    ) -> OptimizationResult:
        """
        QAOA-inspired (Quantum Approximate Optimization Algorithm) portfolio optimizer.
        
        Uses population-based evolution inspired by quantum mixing and problem Hamiltonian.
        """
        
        def portfolio_cost(weights):
            """Cost function mixing return, volatility, and mixing terms."""
            if np.sum(np.abs(weights)) < 0.1:
                return float('inf')
            
            weights = weights / np.sum(np.abs(weights))
            
            # Problem Hamiltonian component
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            
            cost = -portfolio_return + portfolio_vol
            
            # Mixing term (regularization)
            mixing_cost = mixing_strength * (np.sum(weights**4) - 1/self.n_assets)
            
            return cost + mixing_cost
        
        # Population-based optimization (differential evolution)
        bounds = [(-1, 1) for _ in range(self.n_assets)]
        
        result = differential_evolution(
            portfolio_cost,
            bounds,
            maxiter=max_iterations,
            popsize=population_size,
            seed=42,
            workers=1,
            updating='deferred',
            atol=1e-7,
            tol=1e-7
        )
        
        weights = result.x / np.sum(np.abs(result.x))
        weights = np.clip(weights, -1, 1)
        
        metrics = self._calculate_metrics(weights)
        
        return OptimizationResult(
            weights=weights,
            metrics=metrics,
            optimization_time=result.nfev * 0.001,
            method_used=OptimizationMethod.QUANTUM_INSPIRED,
            convergence_info={
                'final_cost': result.fun,
                'function_evals': result.nfev,
                'success': result.success
            }
        )
    
    def mean_variance_optimizer(
        self,
        target_return: Optional[float] = None,
        long_only: bool = True
    ) -> OptimizationResult:
        """
        Classical mean-variance optimization (Markowitz).
        """
        
        if target_return is None:
            target_return = np.mean(self.mean_returns)
        
        # Objective: minimize volatility
        weights = cp.Variable(self.n_assets)
        
        portfolio_vol = cp.quad_form(weights, self.cov_matrix)
        
        constraints = [
            cp.sum(weights) == 1,
            weights @ self.mean_returns >= target_return
        ]
        
        if long_only:
            constraints.append(weights >= 0)
        else:
            constraints.append(cp.sum(cp.abs(weights)) <= 2)  # Max leverage 2x
        
        prob = cp.Problem(cp.Minimize(portfolio_vol), constraints)
        prob.solve(verbose=False)
        
        if prob.status != 'optimal':
            logger.warning(f"Optimization status: {prob.status}")
        
        weights = np.array(weights.value).flatten()
        weights = weights / np.sum(np.abs(weights))
        
        metrics = self._calculate_metrics(weights)
        
        return OptimizationResult(
            weights=weights,
            metrics=metrics,
            optimization_time=0.0,
            method_used=OptimizationMethod.MEAN_VARIANCE,
            convergence_info={'status': prob.status}
        )
    
    def risk_parity_optimizer(
        self,
        max_iterations: int = 1000
    ) -> OptimizationResult:
        """
        Risk parity optimization: equal risk contribution from each asset.
        """
        
        def risk_parity_objective(weights):
            """
            Objective: minimize variance of risk contributions.
            """
            weights = weights / np.sum(np.abs(weights))
            
            # Calculate portfolio variance
            portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            # Marginal risk contribution
            mrc = np.dot(self.cov_matrix, weights) / portfolio_std
            
            # Risk contribution
            rc = weights * mrc
            
            # Target: equal risk (1/n)
            target_rc = portfolio_std / self.n_assets
            
            # Minimize variance of risk contributions
            variance_rc = np.sum((rc - target_rc)**2)
            
            return variance_rc
        
        # Optimize
        x0 = np.ones(self.n_assets) / self.n_assets
        bounds = [(0.001, 1) for _ in range(self.n_assets)]
        
        result = minimize(
            risk_parity_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations, 'ftol': 1e-9}
        )
        
        weights = result.x / np.sum(result.x)
        
        metrics = self._calculate_metrics(weights)
        
        return OptimizationResult(
            weights=weights,
            metrics=metrics,
            optimization_time=0.0,
            method_used=OptimizationMethod.RISK_PARITY,
            convergence_info={'success': result.success, 'iterations': result.nit}
        )
    
    def equal_risk_contribution_optimizer(
        self,
        max_iterations: int = 1000
    ) -> OptimizationResult:
        """
        ERC optimization with volatility-adjusted weights.
        """
        
        def erc_objective(weights):
            """Equal risk contribution objective."""
            weights = np.abs(weights) / np.sum(np.abs(weights))
            
            portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            mrc = np.dot(self.cov_matrix, weights) / portfolio_std
            rc = weights * mrc
            
            # ERC: each asset should contribute equally to portfolio risk
            target_contribution = portfolio_std / self.n_assets
            
            return np.sum((rc - target_contribution)**2)
        
        # Start with volatility-weighted
        weights = 1.0 / self.volatilities
        weights = weights / np.sum(weights)
        
        result = minimize(
            erc_objective,
            weights,
            method='SLSQP',
            options={'maxiter': max_iterations, 'ftol': 1e-10}
        )
        
        weights = result.x / np.sum(np.abs(result.x))
        weights = np.clip(weights, 0, 1)
        
        metrics = self._calculate_metrics(weights)
        
        return OptimizationResult(
            weights=weights,
            metrics=metrics,
            optimization_time=0.0,
            method_used=OptimizationMethod.EQUAL_RISK_CONTRIBUTION,
            convergence_info={'success': result.success}
        )
    
    def hierarchical_risk_parity(
        self,
        n_clusters: int = 3
    ) -> OptimizationResult:
        """
        Hierarchical Risk Parity (HRP) optimization.
        
        Uses hierarchical clustering on correlation matrix to construct
        a dendrogram-based portfolio allocation.
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
        
        # Convert correlation to distance
        distance = np.sqrt((1 - self.corr_matrix) / 2)
        
        # Perform hierarchical clustering
        Z = linkage(pdist(self.corr_matrix, metric='euclidean'), method='ward')
        
        # Create clusters
        from scipy.cluster.hierarchy import cut_tree
        clusters = cut_tree(Z, n_clusters=n_clusters).flatten()
        
        weights = np.ones(self.n_assets)
        
        # Allocate equally within clusters, inverse volatility within clusters
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_vols = self.volatilities[cluster_indices]
            cluster_weights = 1.0 / cluster_vols
            cluster_weights = cluster_weights / np.sum(cluster_weights)
            
            weights[cluster_indices] = cluster_weights / len(set(clusters))
        
        weights = weights / np.sum(weights)
        
        metrics = self._calculate_metrics(weights)
        
        return OptimizationResult(
            weights=weights,
            metrics=metrics,
            optimization_time=0.0,
            method_used=OptimizationMethod.HIERARCHICAL,
            convergence_info={'clusters': n_clusters}
        )
    
    def pareto_frontier(
        self,
        n_points: int = 100,
        max_return_multiplier: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Pareto frontier of efficient portfolios.
        """
        
        target_returns = np.linspace(
            np.min(self.mean_returns),
            max_return_multiplier * np.mean(self.mean_returns),
            n_points
        )
        
        frontiers = []
        volatilities = []
        
        for target_return in target_returns:
            try:
                result = self.mean_variance_optimizer(
                    target_return=target_return,
                    long_only=False
                )
                frontiers.append(result.weights)
                volatilities.append(result.metrics.volatility)
            except:
                continue
        
        returns = [np.dot(w, self.mean_returns) for w in frontiers]
        
        return np.array(returns), np.array(volatilities)
    
    def sensitivity_analysis(
        self,
        weights: np.ndarray,
        perturbation_pct: float = 0.01
    ) -> Dict:
        """
        Analyze portfolio sensitivity to input changes.
        """
        
        base_metrics = self._calculate_metrics(weights)
        sensitivities = {}
        
        # Sensitivity to mean return changes
        return_sensitivities = []
        for i in range(self.n_assets):
            perturbed_returns = self.mean_returns.copy()
            perturbed_returns[i] *= (1 + perturbation_pct)
            
            perturbed_metrics = self._calculate_metrics(
                weights,
                mean_returns=perturbed_returns
            )
            
            sensitivity = (perturbed_metrics.expected_return - 
                          base_metrics.expected_return) / perturbation_pct
            return_sensitivities.append(sensitivity)
        
        sensitivities['return_sensitivity'] = np.array(return_sensitivities)
        
        # Sensitivity to volatility changes
        vol_sensitivities = []
        for i in range(self.n_assets):
            perturbed_vols = self.volatilities.copy()
            perturbed_vols[i] *= (1 + perturbation_pct)
            
            # Reconstruct covariance with new volatilities
            perturbed_cov = self.corr_matrix * np.outer(perturbed_vols, perturbed_vols)
            
            perturbed_volatility = np.sqrt(
                np.dot(weights, np.dot(perturbed_cov, weights))
            )
            
            sensitivity = (perturbed_volatility - base_metrics.volatility) / perturbation_pct
            vol_sensitivities.append(sensitivity)
        
        sensitivities['volatility_sensitivity'] = np.array(vol_sensitivities)
        
        return sensitivities
    
    def _calculate_metrics(
        self,
        weights: np.ndarray,
        mean_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None
    ) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio metrics.
        """
        
        if mean_returns is None:
            mean_returns = self.mean_returns
        if cov_matrix is None:
            cov_matrix = self.cov_matrix
        if returns is None:
            returns = self.returns
        
        # Basic metrics
        expected_return = np.dot(weights, mean_returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        rf_rate = 0.02
        sharpe = (expected_return - rf_rate) / (volatility + 1e-10)
        
        # Sortino ratio (downside deviation)
        portfolio_returns = np.dot(returns, weights)
        downside_returns = np.minimum(portfolio_returns, 0)
        downside_variance = np.mean(downside_returns**2)
        downside_std = np.sqrt(downside_variance)
        sortino = (expected_return - rf_rate) / (downside_std + 1e-10)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        calmar = (expected_return) / (-max_drawdown + 1e-10)
        
        # Value at Risk (95%)
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Conditional Value at Risk
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        # Concentration metrics
        herfindahl = np.sum(weights**2)
        effective_n = 1.0 / herfindahl if herfindahl > 0 else 0
        
        # Turnover (relative to equal weight)
        equal_weight = np.ones(len(weights)) / len(weights)
        turnover = np.sum(np.abs(weights - equal_weight))
        
        # Expected shortfall
        expected_shortfall = np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)])
        
        # Omega ratio
        returns_above_rf = portfolio_returns - rf_rate
        omega = np.sum(returns_above_rf[returns_above_rf > 0]) / (
            -np.sum(returns_above_rf[returns_above_rf < 0]) + 1e-10
        )
        
        # Information ratio (relative to equal weight)
        active_returns = portfolio_returns - np.dot(returns, equal_weight)
        tracking_error = np.std(active_returns)
        info_ratio = np.mean(active_returns) / (tracking_error + 1e-10)
        
        return PortfolioMetrics(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            herfindahl_index=herfindahl,
            effective_number_assets=effective_n,
            turn_over=turnover,
            expected_shortfall=expected_shortfall,
            omega_ratio=omega,
            information_ratio=info_ratio,
            tracking_error=tracking_error
        )


class VQEPortfolioOptimizer:
    """Variational Quantum Eigensolver backend for portfolio optimization."""
    
    def __init__(
        self,
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        mean_returns: np.ndarray,
        target_return: Optional[float],
        risk_aversion: float,
        ansatz_depth: int
    ):
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.mean_returns = mean_returns
        self.target_return = target_return or np.mean(mean_returns)
        self.risk_aversion = risk_aversion
        self.ansatz_depth = ansatz_depth
        self.n_assets = len(mean_returns)
    
    def ansatz(self, params: np.ndarray) -> np.ndarray:
        """
        Parameterized ansatz mapping circuit parameters to probability amplitudes.
        
        Uses RY rotation layers to create superposition states.
        """
        state = np.ones(self.n_assets) / np.sqrt(self.n_assets)
        
        params = params.reshape(self.ansatz_depth, self.n_assets)
        
        for depth in range(self.ansatz_depth):
            # RY rotation
            angles = params[depth]
            for i in range(self.n_assets):
                cos_a = np.cos(angles[i])
                sin_a = np.sin(angles[i])
                state[i] = cos_a * state[i] + sin_a
            
            state = state / (np.linalg.norm(state) + 1e-10)
        
        return state
    
    def decode_weights(self, params: np.ndarray) -> np.ndarray:
        """
        Decode circuit parameters to portfolio weights.
        
        Weights are probabilities from the quantum state.
        """
        state = self.ansatz(params)
        weights = np.abs(state)**2
        return weights
    
    def evaluate(self, params: np.ndarray) -> float:
        """
        Evaluate cost function for given parameters.
        """
        weights = self.decode_weights(params)
        weights = weights / np.sum(weights)
        
        # Portfolio return and volatility
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Objective: maximize Sharpe ratio with target return constraint
        cost = portfolio_vol - (portfolio_return / self.risk_aversion)
        
        # Return constraint penalty
        if self.target_return is not None:
            return_penalty = max(0, self.target_return - portfolio_return)**2 * 100
            cost += return_penalty
        
        return cost


class PortfolioRebalancer:
    """Real-time portfolio rebalancing with transaction cost optimization."""
    
    def __init__(
        self,
        initial_weights: np.ndarray,
        assets: List[str],
        transaction_cost_bps: float = 10,  # Basis points
        min_position_size: float = 0.001  # 0.1%
    ):
        self.current_weights = initial_weights.copy()
        self.target_weights = initial_weights.copy()
        self.assets = assets
        self.transaction_cost_bps = transaction_cost_bps / 10000
        self.min_position_size = min_position_size
        
        self.rebalance_history = []
    
    def calculate_trades(
        self,
        new_weights: np.ndarray,
        current_prices: np.ndarray,
        portfolio_value: float
    ) -> Dict:
        """
        Calculate trades needed to rebalance to new weights.
        
        Accounts for transaction costs and minimum position sizes.
        """
        
        # Clean new weights
        new_weights = np.maximum(new_weights, 0)
        new_weights = new_weights / np.sum(new_weights)
        
        # Enforce minimum position size
        small_positions = new_weights < self.min_position_size
        new_weights[small_positions] = 0
        new_weights = new_weights / np.sum(new_weights)
        
        # Calculate position values
        current_values = self.current_weights * portfolio_value
        target_values = new_weights * portfolio_value
        
        # Calculate shares to trade
        current_shares = current_values / current_prices
        target_shares = target_values / current_prices
        
        shares_to_trade = target_shares - current_shares
        
        # Transaction costs
        transaction_costs = np.abs(shares_to_trade * current_prices) * self.transaction_cost_bps
        total_transaction_cost = np.sum(transaction_costs)
        
        # Net proceeds/costs
        proceeds = -np.sum(shares_to_trade * current_prices)
        
        trades = {
            'assets': self.assets,
            'current_shares': current_shares,
            'target_shares': target_shares,
            'shares_to_trade': shares_to_trade,
            'current_weights': self.current_weights.copy(),
            'target_weights': new_weights.copy(),
            'transaction_costs': transaction_costs,
            'total_transaction_cost': total_transaction_cost,
            'net_proceeds': proceeds,
            'implementation_shortfall': total_transaction_cost / portfolio_value
        }
        
        return trades
    
    def execute_rebalance(self, trades: Dict):
        """Execute portfolio rebalance."""
        self.current_weights = trades['target_weights'].copy()
        self.target_weights = trades['target_weights'].copy()
        
        self.rebalance_history.append({
            'timestamp': np.datetime64('now'),
            'trades': trades,
            'weights': self.current_weights.copy()
        })
    
    def calculate_tracking_error(
        self,
        returns: np.ndarray,
        benchmark_weights: np.ndarray
    ) -> float:
        """Calculate tracking error relative to benchmark."""
        
        portfolio_returns = np.dot(returns, self.current_weights)
        benchmark_returns = np.dot(returns, benchmark_weights)
        
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(active_returns)
        
        return tracking_error


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample returns
    n_assets = 20
    n_periods = 252
    
    mu = np.random.uniform(0.0001, 0.0005, n_assets)
    L = np.random.randn(n_assets, n_assets)
    sigma = np.dot(L, L.T)
    
    returns = np.random.multivariate_normal(mu, sigma, n_periods)
    
    # Optimize
    optimizer = QuantumInspiredOptimizer(n_assets)
    optimizer.load_returns(returns)
    
    print("=" * 80)
    print("QUANTUM-INSPIRED PORTFOLIO OPTIMIZATION")
    print("=" * 80)
    
    # Mean-Variance
    mv_result = optimizer.mean_variance_optimizer()
    print("\nMean-Variance Optimization:")
    print(mv_result.metrics)
    
    # Risk Parity
    rp_result = optimizer.risk_parity_optimizer()
    print("\nRisk Parity Optimization:")
    print(rp_result.metrics)
    
    # VQE
    vqe_result = optimizer.variational_quantum_optimizer(max_iterations=500)
    print("\nVariational Quantum Optimization:")
    print(vqe_result.metrics)
    
    # Pareto frontier
    returns_frontier, vols_frontier = optimizer.pareto_frontier(n_points=50)
    print(f"\nPareto Frontier Generated: {len(returns_frontier)} points")
