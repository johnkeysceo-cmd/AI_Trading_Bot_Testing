"""
PHASE 4: Causal Inference Engine for Alpha Identification

Advanced causal inference framework for identifying true alpha sources
and causal relationships in market data.

Uses propensity score matching, instrumental variables, DAG-based analysis,
and Granger causality tests to isolate genuine trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.stats import zscore, shapiro, jarque_bera
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from itertools import combinations
import logging
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CausalMethod(Enum):
    """Causal inference methodologies."""
    GRANGER_CAUSALITY = "granger"
    PROPENSITY_SCORE = "propensity"
    INSTRUMENTAL_VARIABLE = "iv"
    DIFFERENCE_IN_DIFFERENCES = "did"
    REGRESSION_DISCONTINUITY = "rd"
    SYNTHETIC_CONTROL = "synthetic"
    DAG_ANALYSIS = "dag"
    CAUSAL_FOREST = "forest"


@dataclass
class CausalEffect:
    """Causal effect estimation result."""
    treatment_variable: str
    outcome_variable: str
    method: CausalMethod
    
    # Effect estimates
    ate: float  # Average treatment effect
    att: float  # Average treatment effect on treated
    ate_std_error: float
    att_std_error: float
    
    # Statistical tests
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    
    # Model diagnostics
    r_squared: float
    adjusted_r_squared: float
    residual_std_error: float
    
    # Balance statistics
    standardized_bias: Dict[str, float] = field(default_factory=dict)
    max_bias: float = 0.0
    
    # Robustness
    sensitivity_parameter: float = 1.0
    is_robust: bool = True
    bounds_ci_lower: float = 0.0
    bounds_ci_upper: float = 0.0
    
    def __str__(self):
        return f"""
Causal Effect Analysis: {self.treatment_variable} → {self.outcome_variable}
Method: {self.method.value}
─────────────────────────────────────────────
ATE: {self.ate:.6f} ± {self.ate_std_error:.6f}
ATT: {self.att:.6f} ± {self.att_std_error:.6f}
95% CI: [{self.ci_lower:.6f}, {self.ci_upper:.6f}]

t-statistic: {self.t_statistic:.4f}
p-value: {self.p_value:.6f}
R²: {self.r_squared:.4f}

Max Standardized Bias: {self.max_bias:.4f}
Robust to Hidden Bias (Γ): {self.sensitivity_parameter:.2f}
Is Causal: {self.is_robust and self.p_value < 0.05}
"""


@dataclass
class GrangerCausalityResult:
    """Granger causality test result."""
    cause_variable: str
    effect_variable: str
    lags: int
    
    f_statistic: float
    p_value: float
    test_statistic: float
    critical_values: Dict[str, float]
    
    is_causal: bool
    strength_of_causality: float  # F-statistic scaled
    temporal_lag: int  # Optimal lag
    
    def __str__(self):
        sig = "*" if self.is_causal else ""
        return f"{self.cause_variable} → {self.effect_variable}: F={self.f_statistic:.4f}, p={self.p_value:.4f} {sig}"


class CausalInferenceEngine:
    """
    Comprehensive causal inference engine for identifying true alpha sources.
    """
    
    def __init__(self, data: pd.DataFrame, min_obs: int = 100):
        """
        Initialize causal inference engine.
        
        Args:
            data: DataFrame with all variables (market data, signals, outcomes)
            min_obs: Minimum observations required for analysis
        """
        self.data = data.copy()
        self.variables = list(data.columns)
        self.n_obs = len(data)
        self.min_obs = min_obs
        
        self.scaler = StandardScaler()
        self.data_scaled = self.scaler.fit_transform(data)
        
        logger.info(f"Initialized causal engine with {self.n_obs} observations, "
                   f"{len(self.variables)} variables")
    
    def granger_causality_test(
        self,
        cause: str,
        effect: str,
        max_lags: int = 10,
        significance_level: float = 0.05
    ) -> GrangerCausalityResult:
        """
        Test Granger causality between two time series.
        
        H0: Cause does NOT Granger-cause Effect
        H1: Cause DOES Granger-cause Effect
        """
        
        if cause not in self.variables or effect not in self.variables:
            raise ValueError(f"Variables {cause} or {effect} not found")
        
        x = self.data[cause].values
        y = self.data[effect].values
        
        best_lag = 1
        best_f_stat = 0
        best_p_value = 1.0
        
        for lag in range(1, max_lags + 1):
            # Prepare data
            valid_idx = lag
            x_lagged = x[:-lag]
            y_current = y[lag:]
            
            # Restricted model: y ~ y_lagged
            y_lagged = y[:-lag]
            X_restricted = np.column_stack([np.ones_like(y_lagged), y_lagged])
            
            # Unrestricted model: y ~ y_lagged + x_lagged
            X_unrestricted = np.column_stack([np.ones_like(y_current), y_current, x_lagged])
            
            # Fit models
            try:
                beta_r = np.linalg.lstsq(X_restricted, y_lagged, rcond=None)[0]
                residuals_r = y_lagged - X_restricted @ beta_r
                rss_r = np.sum(residuals_r**2)
                
                beta_u = np.linalg.lstsq(X_unrestricted, y_current, rcond=None)[0]
                residuals_u = y_current - X_unrestricted @ beta_u
                rss_u = np.sum(residuals_u**2)
                
                # F-statistic
                n = len(y_current)
                q = 1  # Number of restrictions
                f_stat = ((rss_r - rss_u) / q) / (rss_u / (n - 3))
                p_value = 1 - stats.f.cdf(f_stat, q, n - 3)
                
                if f_stat > best_f_stat:
                    best_f_stat = f_stat
                    best_p_value = p_value
                    best_lag = lag
            
            except:
                continue
        
        is_causal = best_p_value < significance_level
        
        critical_values = {
            '10%': stats.f.ppf(0.90, 1, self.n_obs - 3),
            '5%': stats.f.ppf(0.95, 1, self.n_obs - 3),
            '1%': stats.f.ppf(0.99, 1, self.n_obs - 3)
        }
        
        return GrangerCausalityResult(
            cause_variable=cause,
            effect_variable=effect,
            lags=best_lag,
            f_statistic=best_f_stat,
            p_value=best_p_value,
            test_statistic=best_f_stat,
            critical_values=critical_values,
            is_causal=is_causal,
            strength_of_causality=best_f_stat / critical_values['5%'],
            temporal_lag=best_lag
        )
    
    def propensity_score_matching(
        self,
        treatment: str,
        outcome: str,
        confounders: List[str],
        caliper: float = 0.1,
        estimand: str = "ATE"
    ) -> CausalEffect:
        """
        Estimate causal effect using propensity score matching.
        
        Matches treated units with similar control units based on propensity scores.
        """
        
        # Extract variables
        T = (self.data[treatment] > self.data[treatment].median()).astype(int).values
        Y = self.data[outcome].values
        X = self.data[confounders].values
        
        # Estimate propensity scores
        ps_model = LinearRegression()
        ps_model.fit(X, T)
        propensity_scores = ps_model.predict(X)
        propensity_scores = np.clip(propensity_scores, 0.01, 0.99)
        
        # Matching
        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]
        
        matched_pairs = []
        matched_treated = set()
        matched_control = set()
        
        # Nearest neighbor matching
        for t_idx in treated_idx:
            ps_t = propensity_scores[t_idx]
            
            # Find closest control
            distances = np.abs(propensity_scores[control_idx] - ps_t)
            closest_control = control_idx[np.argmin(distances)]
            
            if distances.min() < caliper and closest_control not in matched_control:
                matched_pairs.append((t_idx, closest_control))
                matched_treated.add(t_idx)
                matched_control.add(closest_control)
        
        matched_pairs = np.array(matched_pairs)
        
        # Calculate treatment effects
        treated_outcomes = Y[matched_pairs[:, 0]]
        control_outcomes = Y[matched_pairs[:, 1]]
        
        individual_effects = treated_outcomes - control_outcomes
        ate = np.mean(individual_effects)
        ate_std = np.std(individual_effects) / np.sqrt(len(matched_pairs))
        
        att = ate  # With matching, ATE and ATT are similar
        att_std = ate_std
        
        # Statistical test
        t_stat = ate / (ate_std + 1e-10)
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), len(matched_pairs) - 1))
        
        # Confidence interval
        ci = stats.t.ppf(0.975, len(matched_pairs) - 1) * ate_std
        
        # Check balance
        standardized_bias = {}
        for i, confounder in enumerate(confounders):
            treated_mean = X[matched_pairs[:, 0], i].mean()
            control_mean = X[matched_pairs[:, 1], i].mean()
            pooled_std = np.sqrt(
                (np.std(X[matched_pairs[:, 0], i])**2 + 
                 np.std(X[matched_pairs[:, 1], i])**2) / 2
            )
            
            bias = (treated_mean - control_mean) / (pooled_std + 1e-10)
            standardized_bias[confounder] = bias
        
        max_bias = max(np.abs(list(standardized_bias.values())))
        
        # Model diagnostics
        X_matched_treated = X[matched_pairs[:, 0]]
        X_matched_control = X[matched_pairs[:, 1]]
        X_matched = np.vstack([X_matched_treated, X_matched_control])
        y_matched = np.hstack([np.ones(len(matched_pairs)), np.zeros(len(matched_pairs))])
        
        reg = LinearRegression()
        reg.fit(X_matched, y_matched)
        r2 = reg.score(X_matched, y_matched)
        
        residuals = y_matched - reg.predict(X_matched)
        residual_std = np.std(residuals)
        
        n = len(matched_pairs)
        k = len(confounders)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        
        return CausalEffect(
            treatment_variable=treatment,
            outcome_variable=outcome,
            method=CausalMethod.PROPENSITY_SCORE,
            ate=ate,
            att=att,
            ate_std_error=ate_std,
            att_std_error=att_std,
            t_statistic=t_stat,
            p_value=p_value,
            ci_lower=ate - ci,
            ci_upper=ate + ci,
            r_squared=r2,
            adjusted_r_squared=adj_r2,
            residual_std_error=residual_std,
            standardized_bias=standardized_bias,
            max_bias=max_bias,
            is_robust=max_bias < 0.1 and p_value < 0.05
        )
    
    def instrumental_variable_estimation(
        self,
        treatment: str,
        outcome: str,
        instrument: str,
        confounders: List[str]
    ) -> CausalEffect:
        """
        Estimate causal effect using instrumental variables.
        
        Two-stage least squares (2SLS) estimation.
        """
        
        # Stage 1: Predict treatment using instrument
        X = self.data[confounders + [instrument]].values
        T = self.data[treatment].values
        
        stage1_model = LinearRegression()
        stage1_model.fit(X, T)
        T_predicted = stage1_model.predict(X)
        
        # Stage 2: Predict outcome using predicted treatment
        Y = self.data[outcome].values
        X_stage2 = np.column_stack([np.ones_like(T_predicted), T_predicted, X[:, :-1]])
        
        stage2_model = LinearRegression()
        stage2_model.fit(X_stage2, Y)
        
        # Extract treatment effect (second coefficient)
        ate = stage2_model.coef_[1]
        
        # Calculate standard errors via bootstrap
        n = len(T)
        bootstrap_effects = []
        
        for _ in range(100):
            idx = np.random.choice(n, n, replace=True)
            
            X_boot = X[idx]
            T_boot = T[idx]
            Y_boot = Y[idx]
            
            s1 = LinearRegression()
            s1.fit(X_boot, T_boot)
            T_pred_boot = s1.predict(X_boot)
            
            X_s2_boot = np.column_stack([np.ones_like(T_pred_boot), T_pred_boot, X_boot[:, :-1]])
            s2 = LinearRegression()
            s2.fit(X_s2_boot, Y_boot)
            
            bootstrap_effects.append(s2.coef_[1])
        
        bootstrap_effects = np.array(bootstrap_effects)
        ate_std = np.std(bootstrap_effects)
        
        # Test
        t_stat = ate / (ate_std + 1e-10)
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - len(stage2_model.coef_)))
        
        ci = 1.96 * ate_std
        
        return CausalEffect(
            treatment_variable=treatment,
            outcome_variable=outcome,
            method=CausalMethod.INSTRUMENTAL_VARIABLE,
            ate=ate,
            att=ate,  # IV gives LATE
            ate_std_error=ate_std,
            att_std_error=ate_std,
            t_statistic=t_stat,
            p_value=p_value,
            ci_lower=ate - ci,
            ci_upper=ate + ci,
            r_squared=stage2_model.score(X_stage2, Y),
            adjusted_r_squared=0.0,
            residual_std_error=np.std(Y - stage2_model.predict(X_stage2)),
            is_robust=p_value < 0.05
        )
    
    def directed_acyclic_graph_analysis(
        self,
        relationships: Dict[str, List[str]]
    ) -> nx.DiGraph:
        """
        Analyze causal structure using DAGs.
        
        relationships: {'X': ['Y', 'Z']} means X causes Y and Z
        """
        
        dag = nx.DiGraph()
        
        for cause, effects in relationships.items():
            dag.add_node(cause)
            for effect in effects:
                dag.add_node(effect)
                dag.add_edge(cause, effect)
        
        # Analyze paths and colliders
        cycles = list(nx.simple_cycles(dag))
        if cycles:
            logger.warning(f"Cycles detected in DAG: {cycles}")
        
        # Find d-connected paths
        for node in dag.nodes():
            ancestors = nx.ancestors(dag, node)
            descendants = nx.descendants(dag, node)
            
            logger.debug(f"{node}: ancestors={ancestors}, descendants={descendants}")
        
        return dag
    
    def synthetic_control_method(
        self,
        treated_unit: str,
        control_units: List[str],
        pre_period: Tuple[int, int],
        post_period: Tuple[int, int],
        outcome: str
    ) -> Dict:
        """
        Estimate treatment effect using synthetic control method.
        """
        
        # Extract data
        pre_start, pre_end = pre_period
        post_start, post_end = post_period
        
        treated_pre = self.data.loc[pre_start:pre_end, treated_unit].values
        control_pre = self.data.loc[pre_start:pre_end, control_units].values
        
        treated_post = self.data.loc[post_start:post_end, treated_unit].values
        control_post = self.data.loc[post_start:post_end, control_units].values
        
        # Find optimal weights for synthetic control
        def synthetic_control_loss(weights):
            weights = np.abs(weights) / np.sum(np.abs(weights))
            synthetic_pre = np.dot(control_pre, weights)
            
            return np.sum((treated_pre - synthetic_pre)**2)
        
        from scipy.optimize import minimize as scipy_minimize
        
        x0 = np.ones(len(control_units)) / len(control_units)
        bounds = [(0, 1) for _ in range(len(control_units))]
        
        result = scipy_minimize(
            synthetic_control_loss,
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        optimal_weights = np.abs(result.x) / np.sum(np.abs(result.x))
        
        # Calculate synthetic control
        synthetic_post = np.dot(control_post, optimal_weights)
        
        # Treatment effect
        ate = np.mean(treated_post - synthetic_post)
        
        return {
            'weights': dict(zip(control_units, optimal_weights)),
            'ate': ate,
            'treated_path': treated_post,
            'synthetic_path': synthetic_post,
            'individual_effects': treated_post - synthetic_post
        }
    
    def placebo_test(
        self,
        treatment_indicator: np.ndarray,
        outcome: str,
        n_placebos: int = 100
    ) -> Tuple[float, List[float]]:
        """
        Conduct placebo test to assess validity of causal inference.
        
        Randomly assigns "treatment" to control units and checks if
        estimated effects are consistent.
        """
        
        Y = self.data[outcome].values
        n = len(Y)
        
        # True effect
        treated = treatment_indicator == 1
        true_effect = np.mean(Y[treated]) - np.mean(Y[~treated])
        
        # Placebo effects
        placebo_effects = []
        
        for _ in range(n_placebos):
            placebo_treatment = np.random.choice(
                [0, 1],
                size=n,
                p=[1 - treated.mean(), treated.mean()]
            )
            
            placebo_effect = np.mean(Y[placebo_treatment == 1]) - np.mean(Y[placebo_treatment == 0])
            placebo_effects.append(placebo_effect)
        
        return true_effect, placebo_effects
    
    def rosenbaum_sensitivity_analysis(
        self,
        effect_estimate: float,
        std_error: float,
        gamma: float = 1.5
    ) -> Dict:
        """
        Rosenbaum sensitivity analysis for hidden bias.
        
        Analyzes how robust causal conclusion is to unobserved confounding.
        """
        
        # Critical gamma where inference changes
        z_critical = 1.96  # 95% confidence
        
        Gamma_range = np.linspace(1, gamma, 50)
        bounds = []
        
        for Gamma in Gamma_range:
            # Lower bound on treatment effect under unobserved confounding
            lower = (effect_estimate - z_critical * std_error) / Gamma
            # Upper bound
            upper = (effect_estimate + z_critical * std_error) * Gamma
            
            bounds.append({'Gamma': Gamma, 'lower': lower, 'upper': upper})
        
        return {
            'bounds': bounds,
            'is_robust': all(b['lower'] > 0 or b['upper'] < 0 for b in bounds),
            'critical_gamma': gamma,
            'interpretation': f"Inference is robust if Gamma < {gamma}"
        }
    
    def multiple_testing_correction(
        self,
        p_values: List[float],
        method: str = "benjamini_hochberg"
    ) -> Dict:
        """
        Adjust p-values for multiple comparisons.
        """
        
        p_values = np.array(p_values)
        n = len(p_values)
        
        if method == "bonferroni":
            adjusted = np.minimum(p_values * n, 1.0)
        
        elif method == "benjamini_hochberg":
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            
            threshold = np.arange(1, n + 1) * 0.05 / n
            adjusted_sorted = sorted_p < threshold
            
            adjusted = np.zeros_like(p_values)
            adjusted[sorted_idx] = adjusted_sorted
        
        else:
            adjusted = p_values
        
        return {
            'original_p_values': list(p_values),
            'adjusted_p_values': list(adjusted),
            'method': method,
            'significant_after_correction': np.sum(adjusted < 0.05)
        }
    
    def find_causal_graph(
        self,
        max_lag: int = 5,
        significance_level: float = 0.05
    ) -> nx.DiGraph:
        """
        Auto-discover causal graph using Granger causality tests.
        """
        
        dag = nx.DiGraph()
        
        for var in self.variables:
            dag.add_node(var)
        
        # Test all pairs
        for cause in self.variables:
            for effect in self.variables:
                if cause != effect:
                    result = self.granger_causality_test(
                        cause,
                        effect,
                        max_lags=max_lag,
                        significance_level=significance_level
                    )
                    
                    if result.is_causal:
                        dag.add_edge(cause, effect, weight=result.strength_of_causality)
        
        return dag


class AlphaDecomposition:
    """Decompose trading returns into causal components."""
    
    def __init__(self, returns: np.ndarray, signal_components: Dict[str, np.ndarray]):
        self.returns = returns
        self.signal_components = signal_components
        self.n_components = len(signal_components)
    
    def decompose_returns(self) -> Dict:
        """
        Decompose returns into contributions from each signal.
        """
        
        # Stack signal components
        X = np.column_stack(list(self.signal_components.values()))
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, self.returns)
        
        # Get contributions
        contributions = {}
        for i, name in enumerate(self.signal_components.keys()):
            contributions[name] = model.coef_[i]
        
        # Residual (unexplained alpha)
        residuals = self.returns - model.predict(X)
        contributions['residual'] = np.mean(residuals)
        
        # R-squared
        r2 = model.score(X, self.returns)
        
        return {
            'contributions': contributions,
            'r_squared': r2,
            'residual_std': np.std(residuals),
            'model': model
        }
    
    def analyze_alpha_persistence(self) -> Dict:
        """Analyze if alpha is persistent (causal) or random."""
        
        # Split into two periods
        n = len(self.returns)
        split = n // 2
        
        X_1 = np.column_stack(list(self.signal_components.values()))[:split]
        X_2 = X_1[split:]
        
        y_1 = self.returns[:split]
        y_2 = self.returns[split:]
        
        # Fit on first period
        model_1 = LinearRegression()
        model_1.fit(X_1, y_1)
        
        # Test on second period
        pred_2 = model_1.predict(X_2)
        r2_2 = 1 - np.sum((y_2 - pred_2)**2) / np.sum((y_2 - np.mean(y_2))**2)
        
        # If R² drops significantly, alpha was not causal
        is_persistent = r2_2 > 0.3
        
        return {
            'r2_period_1': model_1.score(X_1, y_1),
            'r2_period_2': r2_2,
            'is_persistent': is_persistent,
            'persistence_ratio': r2_2 / (model_1.score(X_1, y_1) + 1e-10)
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    n = 500
    data = pd.DataFrame({
        'signal_1': np.random.randn(n),
        'signal_2': np.random.randn(n),
        'confounder': np.random.randn(n),
        'outcome': np.random.randn(n)
    })
    
    engine = CausalInferenceEngine(data)
    
    print("=" * 80)
    print("CAUSAL INFERENCE ENGINE")
    print("=" * 80)
    
    # Granger causality
    gc = engine.granger_causality_test('signal_1', 'outcome', max_lags=5)
    print(f"\nGranger Causality:\n{gc}")
    
    # Propensity score matching
    treatment_data = pd.DataFrame({
        'treatment': (np.random.randn(n) > 0).astype(int),
        'outcome': np.random.randn(n),
        'confounder_1': np.random.randn(n),
        'confounder_2': np.random.randn(n)
    })
    
    ps_engine = CausalInferenceEngine(treatment_data)
    ps_result = ps_engine.propensity_score_matching(
        'treatment',
        'outcome',
        ['confounder_1', 'confounder_2']
    )
    print(f"\nPropensity Score Matching:\n{ps_result}")
    
    # DAG analysis
    relationships = {
        'signal_1': ['outcome'],
        'signal_2': ['outcome'],
        'confounder': ['signal_1', 'signal_2']
    }
    
    dag = engine.directed_acyclic_graph_analysis(relationships)
    print(f"\nDAG nodes: {list(dag.nodes())}")
    print(f"DAG edges: {list(dag.edges())}")
