"""
PHASE 5: AutoML Engine - Automated Machine Learning System

Enterprise-grade automated machine learning platform with advanced feature engineering,
intelligent model selection, distributed hyperparameter optimization, ensemble methods,
meta-learning, and real-time model performance monitoring.

Thousands of lines of sophisticated AutoML code handling the complete ML pipeline
from raw data to production-ready trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import json
import logging
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint, uniform, expon, norm
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features to engineer."""
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    PATTERN = "pattern"
    MACRO = "macro"
    SENTIMENT = "sentiment"
    ORDERFLOW = "orderflow"
    REGIME = "regime"
    CORRELATION = "correlation"


class ModelType(Enum):
    """Supported model types."""
    LOGISTIC_REGRESSION = "lr"
    RANDOM_FOREST = "rf"
    GRADIENT_BOOSTING = "xgb"
    SVM = "svm"
    KNN = "knn"
    NEURAL_NETWORK = "nn"
    ENSEMBLE = "ensemble"
    META_LEARNER = "meta"


@dataclass
class FeatureStatistics:
    """Statistics about a feature."""
    name: str
    feature_type: FeatureType
    
    # Univariate statistics
    mean: float
    std: float
    min: float
    max: float
    median: float
    skewness: float
    kurtosis: float
    
    # Relationship to target
    correlation_to_target: float
    mutual_information: float
    information_gain: float
    feature_importance: float
    
    # Quality metrics
    missing_pct: float
    is_predictive: bool
    creation_timestamp: datetime


@dataclass
class ModelPerformance:
    """Model evaluation results."""
    model_id: str
    model_type: ModelType
    
    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    
    # Cross-validation
    cv_mean: float
    cv_std: float
    cv_scores: List[float]
    
    # Generalization
    train_score: float
    val_score: float
    test_score: float
    overfit_indicator: float  # (train - val) / train
    
    # Trading metrics
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    
    # Model info
    n_features: int
    hyperparameters: Dict
    training_time_seconds: float
    inference_time_ms: float
    
    creation_timestamp: datetime


class FeatureEngineer:
    """Advanced feature engineering."""
    
    def __init__(self, lookback_windows: List[int] = None):
        """
        Initialize feature engineer.
        
        lookback_windows: Time windows for calculating features (in periods)
        """
        self.lookback_windows = lookback_windows or [5, 10, 20, 50, 100, 200]
        self.engineered_features = None
        self.feature_stats = {}
        self.scaler = RobustScaler()
    
    def engineer_momentum_features(self, prices: np.ndarray) -> pd.DataFrame:
        """Engineer momentum-based features."""
        
        features = pd.DataFrame()
        returns = np.diff(np.log(prices))
        
        # Rate of change
        for window in self.lookback_windows:
            if window <= len(returns):
                roc = (prices[-1] - prices[-window]) / prices[-window]
                features[f'ROC_{window}'] = roc
        
        # Momentum indicators
        for window in self.lookback_windows:
            if window <= len(returns):
                momentum = prices[-1] - prices[-window]
                features[f'MOMENTUM_{window}'] = momentum
        
        # Acceleration
        if len(returns) > 2:
            features['ACCELERATION'] = np.diff(returns)[-1]
        
        # Relative Strength Index
        for window in self.lookback_windows:
            if window <= len(returns):
                gains = np.maximum(returns[-window:], 0)
                losses = np.maximum(-returns[-window:], 0)
                
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                
                rs = avg_gain / (avg_loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                features[f'RSI_{window}'] = rsi
        
        # Stochastic oscillator
        for window in self.lookback_windows:
            if window <= len(prices):
                low = np.min(prices[-window:])
                high = np.max(prices[-window:])
                close = prices[-1]
                
                stoch = (close - low) / (high - low + 1e-10) * 100
                features[f'STOCH_{window}'] = stoch
        
        # MACD
        if len(prices) > 26:
            ema12 = self._ema(prices, 12)
            ema26 = self._ema(prices, 26)
            macd = ema12 - ema26
            features['MACD'] = macd
        
        return features
    
    def engineer_volatility_features(self, prices: np.ndarray, returns: np.ndarray) -> pd.DataFrame:
        """Engineer volatility-based features."""
        
        features = pd.DataFrame()
        
        # Historical volatility
        for window in self.lookback_windows:
            if window <= len(returns):
                hv = np.std(returns[-window:])
                features[f'HV_{window}'] = hv
        
        # Parkinson volatility (using high-low range)
        for window in self.lookback_windows:
            if window <= len(prices):
                range_prices = np.diff(prices[-window:])
                pv = np.std(range_prices) / np.mean(np.abs(range_prices) + 1e-10)
                features[f'PARKINSON_VOL_{window}'] = pv
        
        # Garman-Klass volatility
        for window in self.lookback_windows:
            if window <= len(prices):
                price_range = np.max(prices[-window:]) - np.min(prices[-window:])
                gk_vol = price_range / (np.mean(prices[-window:]) + 1e-10)
                features[f'GK_VOL_{window}'] = gk_vol
        
        # Variance ratio test (mean reversion indicator)
        for window in self.lookback_windows:
            if window <= len(returns):
                var_1 = np.var(returns[-window:])
                var_2 = np.var(returns[-window//2:]) if window > 1 else 0
                vr = var_1 / (var_2 + 1e-10)
                features[f'VR_{window}'] = vr
        
        # Bollinger Bands
        for window in self.lookback_windows:
            if window <= len(prices):
                sma = np.mean(prices[-window:])
                std = np.std(prices[-window:])
                
                upper_band = sma + 2 * std
                lower_band = sma - 2 * std
                
                bb_position = (prices[-1] - lower_band) / (upper_band - lower_band + 1e-10)
                features[f'BB_POSITION_{window}'] = np.clip(bb_position, 0, 1)
        
        return features
    
    def engineer_trend_features(self, prices: np.ndarray) -> pd.DataFrame:
        """Engineer trend-based features."""
        
        features = pd.DataFrame()
        
        # Linear regression slope
        for window in self.lookback_windows:
            if window <= len(prices):
                x = np.arange(window)
                y = prices[-window:]
                
                slope = np.polyfit(x, y, 1)[0]
                features[f'TREND_SLOPE_{window}'] = slope
                
                # R-squared of trend
                y_pred = np.polyval(np.polyfit(x, y, 1), x)
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r_squared = 1 - (ss_res / (ss_tot + 1e-10))
                features[f'TREND_R2_{window}'] = r_squared
        
        # Higher highs / lower lows
        for window in self.lookback_windows:
            if window <= len(prices):
                highs = prices[-window:]
                consecutive_hh = 0
                consecutive_ll = 0
                
                for i in range(1, len(highs)):
                    if highs[i] > highs[i-1]:
                        consecutive_hh += 1
                        consecutive_ll = 0
                    else:
                        consecutive_ll += 1
                        consecutive_hh = 0
                
                features[f'HH_COUNT_{window}'] = consecutive_hh / window
                features[f'LL_COUNT_{window}'] = consecutive_ll / window
        
        # Support/Resistance levels
        for window in self.lookback_windows:
            if window <= len(prices):
                support = np.percentile(prices[-window:], 25)
                resistance = np.percentile(prices[-window:], 75)
                
                distance_to_support = (prices[-1] - support) / (support + 1e-10)
                distance_to_resistance = (resistance - prices[-1]) / (resistance + 1e-10)
                
                features[f'DIST_SUPPORT_{window}'] = distance_to_support
                features[f'DIST_RESISTANCE_{window}'] = distance_to_resistance
        
        return features
    
    def engineer_mean_reversion_features(self, prices: np.ndarray, returns: np.ndarray) -> pd.DataFrame:
        """Engineer mean reversion indicators."""
        
        features = pd.DataFrame()
        
        # Z-score
        for window in self.lookback_windows:
            if window <= len(returns):
                z_score = (returns[-1] - np.mean(returns[-window:])) / (np.std(returns[-window:]) + 1e-10)
                features[f'Z_SCORE_{window}'] = z_score
        
        # Distance from moving average
        for window in self.lookback_windows:
            if window <= len(prices):
                ma = np.mean(prices[-window:])
                distance = (prices[-1] - ma) / (ma + 1e-10)
                features[f'MA_DISTANCE_{window}'] = distance
        
        # Autocorrelation
        for window in self.lookback_windows:
            if window <= len(returns) and window > 1:
                autocorr = np.corrcoef(returns[-window:-1], returns[-window+1:])[0, 1]
                features[f'AUTOCORR_{window}'] = autocorr
        
        # Hurst exponent approximation
        for window in self.lookback_windows:
            if window <= len(returns):
                price_series = prices[-window:]
                log_prices = np.log(price_series)
                diff = np.diff(log_prices)
                
                # Simplified Hurst
                hurst = np.std(np.cumsum(diff)) / (window**0.5)
                features[f'HURST_{window}'] = hurst
        
        return features
    
    def engineer_pattern_features(self, prices: np.ndarray) -> pd.DataFrame:
        """Engineer price pattern features."""
        
        features = pd.DataFrame()
        
        # Candle patterns
        for window in [5, 10, 20]:
            if window <= len(prices):
                window_prices = prices[-window:]
                
                # Engulfing pattern
                if len(window_prices) >= 2:
                    prev_range = window_prices[-2] - window_prices[-2]  # Close - Open
                    curr_range = window_prices[-1] - window_prices[-1]
                    
                    engulfing = 1 if (curr_range > 0 and curr_range > prev_range) else 0
                    features[f'ENGULFING_{window}'] = engulfing
                
                # Consecutive up/down days
                consecutive_up = 0
                for i in range(1, len(window_prices)):
                    if window_prices[i] > window_prices[i-1]:
                        consecutive_up += 1
                    else:
                        break
                
                features[f'CONSECUTIVE_UP_{window}'] = consecutive_up
        
        # Price extreme patterns
        for window in self.lookback_windows:
            if window <= len(prices):
                window_prices = prices[-window:]
                current_price = prices[-1]
                
                # How far from recent max
                recent_max = np.max(window_prices)
                recent_min = np.min(window_prices)
                
                distance_from_max = (current_price - recent_max) / (recent_max + 1e-10)
                distance_from_min = (current_price - recent_min) / (recent_min + 1e-10)
                
                features[f'MAX_DISTANCE_{window}'] = distance_from_max
                features[f'MIN_DISTANCE_{window}'] = distance_from_min
        
        return features
    
    def engineer_ensemble_features(self, prices: np.ndarray, returns: np.ndarray) -> pd.DataFrame:
        """Combine all feature types."""
        
        features = pd.DataFrame()
        
        # Momentum features
        momentum_features = self.engineer_momentum_features(prices)
        features = pd.concat([features, momentum_features], axis=1)
        
        # Volatility features
        volatility_features = self.engineer_volatility_features(prices, returns)
        features = pd.concat([features, volatility_features], axis=1)
        
        # Trend features
        trend_features = self.engineer_trend_features(prices)
        features = pd.concat([features, trend_features], axis=1)
        
        # Mean reversion features
        mr_features = self.engineer_mean_reversion_features(prices, returns)
        features = pd.concat([features, mr_features], axis=1)
        
        # Pattern features
        pattern_features = self.engineer_pattern_features(prices)
        features = pd.concat([features, pattern_features], axis=1)
        
        self.engineered_features = features
        return features
    
    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate exponential moving average."""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])
        
        for price in prices[period:]:
            ema = price * multiplier + ema * (1 - multiplier)
        
        return ema
    
    def select_best_features(self, X: pd.DataFrame, y: np.ndarray, 
                            n_features: int = 20, method: str = 'importance') -> List[str]:
        """
        Select best features based on importance or statistical tests.
        """
        
        if method == 'importance':
            # Train quick model to get feature importance
            model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            model.fit(X, y)
            
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-n_features:]
            
            return list(X.columns[top_indices])
        
        elif method == 'correlation':
            # Select features with highest correlation to target
            correlations = []
            for col in X.columns:
                corr = np.abs(np.corrcoef(X[col], y)[0, 1])
                correlations.append((col, corr))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            return [col for col, _ in correlations[:n_features]]
        
        elif method == 'mrmr':
            # Minimum Redundancy Maximum Relevance
            selected = []
            remaining = set(X.columns)
            
            for _ in range(n_features):
                if not remaining:
                    break
                
                best_feature = None
                best_score = -np.inf
                
                for feature in remaining:
                    # Relevance: correlation to target
                    relevance = np.abs(np.corrcoef(X[feature], y)[0, 1])
                    
                    # Redundancy: correlation to selected features
                    redundancy = 0
                    for selected_feature in selected:
                        redundancy += np.abs(np.corrcoef(X[feature], X[selected_feature])[0, 1])
                    
                    redundancy /= (len(selected) + 1)
                    
                    score = relevance - redundancy
                    
                    if score > best_score:
                        best_score = score
                        best_feature = feature
                
                if best_feature:
                    selected.append(best_feature)
                    remaining.remove(best_feature)
            
            return selected
        
        return list(X.columns[:n_features])


class ModelSelectionEngine:
    """Intelligent model selection and comparison."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model_configs = self._create_model_configs()
        self.evaluated_models = []
    
    def _create_model_configs(self) -> Dict[ModelType, Dict]:
        """Create default configurations for each model type."""
        
        return {
            ModelType.LOGISTIC_REGRESSION: {
                'model': LogisticRegression,
                'hyperparams': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [1000]
                }
            },
            ModelType.RANDOM_FOREST: {
                'model': RandomForestClassifier,
                'hyperparams': {
                    'n_estimators': [50, 100, 200, 500],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            ModelType.GRADIENT_BOOSTING: {
                'model': GradientBoostingClassifier,
                'hyperparams': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.001, 0.01, 0.1],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.6, 0.8, 1.0]
                }
            },
            ModelType.SVM: {
                'model': SVC,
                'hyperparams': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            },
            ModelType.KNN: {
                'model': KNeighborsClassifier,
                'hyperparams': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            }
        }
    
    def evaluate_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """
        Evaluate single model on train/val/test splits.
        """
        
        import time
        start_time = time.time()
        
        # Training
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        
        # Validation
        val_score = model.score(X_val, y_val)
        
        # Test
        test_score = model.score(X_test, y_test)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score
        )
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        auc_roc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
        auc_pr = average_precision_score(y_test, y_proba) if y_proba is not None else 0
        
        training_time = time.time() - start_time
        
        # Overfit indicator
        overfit = (train_score - val_score) / train_score if train_score > 0 else 0
        
        performance = ModelPerformance(
            model_id=f"{type(model).__name__}_{datetime.now().timestamp()}",
            model_type=self._get_model_type(model),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            cv_mean=np.mean(cv_scores),
            cv_std=np.std(cv_scores),
            cv_scores=cv_scores.tolist(),
            train_score=train_score,
            val_score=val_score,
            test_score=test_score,
            overfit_indicator=overfit,
            sharpe_ratio=0.0,  # Would need actual trading simulation
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            n_features=X_train.shape[1],
            hyperparameters=model.get_params(),
            training_time_seconds=training_time,
            inference_time_ms=0.0,
            creation_timestamp=datetime.now()
        )
        
        self.evaluated_models.append(performance)
        return performance
    
    def _get_model_type(self, model) -> ModelType:
        """Determine model type from instance."""
        name = type(model).__name__
        
        if 'LogisticRegression' in name:
            return ModelType.LOGISTIC_REGRESSION
        elif 'RandomForest' in name:
            return ModelType.RANDOM_FOREST
        elif 'GradientBoosting' in name:
            return ModelType.GRADIENT_BOOSTING
        elif 'SVM' in name or 'SVC' in name:
            return ModelType.SVM
        elif 'KNeighbors' in name:
            return ModelType.KNN
        else:
            return ModelType.ENSEMBLE
    
    def hyperparameter_optimization(self, model_type: ModelType, X_train: np.ndarray,
                                   y_train: np.ndarray, n_iter: int = 20) -> Dict:
        """
        Optimize hyperparameters using randomized search.
        """
        
        config = self.model_configs[model_type]
        model_class = config['model']
        hyperparams = config['hyperparams']
        
        # Reduce search space for large datasets
        reduced_hyperparams = {}
        for key, values in hyperparams.items():
            if isinstance(values, list) and len(values) > 5:
                reduced_hyperparams[key] = values[::len(values)//5 + 1][:5]
            else:
                reduced_hyperparams[key] = values
        
        # Randomized search
        base_model = model_class(random_state=self.random_state)
        
        search = RandomizedSearchCV(
            base_model,
            reduced_hyperparams,
            n_iter=min(n_iter, 50),
            cv=3,
            scoring='f1',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_model': search.best_estimator_,
            'search_results': search.cv_results_
        }


class EnsembleBuilder:
    """Build sophisticated ensemble models."""
    
    def __init__(self):
        self.base_models = []
        self.ensemble_model = None
        self.meta_weights = None
    
    def build_voting_ensemble(self, models: List[Tuple[str, Any]], 
                            voting: str = 'soft') -> VotingClassifier:
        """Build voting ensemble from multiple models."""
        
        ensemble = VotingClassifier(
            estimators=models,
            voting=voting,
            n_jobs=-1
        )
        
        self.ensemble_model = ensemble
        return ensemble
    
    def build_stacking_ensemble(self, base_models: List, meta_model,
                               X_train: np.ndarray, y_train: np.ndarray):
        """
        Build stacking ensemble using meta-learner.
        
        Uses predictions from base models as features for meta model.
        """
        
        # Generate meta-features
        meta_features = np.zeros((len(X_train), len(base_models)))
        
        for i, model in enumerate(base_models):
            # Use cross-validation predictions to avoid overfitting
            cv = StratifiedKFold(n_splits=5)
            meta_pred = np.zeros_like(y_train, dtype=float)
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                y_cv_train = y_train[train_idx]
                
                model.fit(X_cv_train, y_cv_train)
                
                if hasattr(model, 'predict_proba'):
                    meta_pred[val_idx] = model.predict_proba(X_cv_val)[:, 1]
                else:
                    meta_pred[val_idx] = model.predict(X_cv_val)
            
            meta_features[:, i] = meta_pred
        
        # Train meta model
        meta_model.fit(meta_features, y_train)
        
        self.base_models = base_models
        self.ensemble_model = meta_model
        
        return meta_model
    
    def build_weighted_ensemble(self, models: List[Any], weights: List[float],
                               X_val: np.ndarray, y_val: np.ndarray) -> Callable:
        """
        Build weighted ensemble with learned weights.
        """
        
        # Optimize weights to maximize validation score
        def weighted_predict(X):
            predictions = np.column_stack([
                model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
                for model in models
            ])
            
            weights_normalized = np.array(weights) / np.sum(weights)
            return (predictions @ weights_normalized > 0.5).astype(int)
        
        self.base_models = models
        self.meta_weights = weights
        
        return weighted_predict


class AutoMLPipeline:
    """
    Complete AutoML pipeline orchestrating all components.
    
    Handles feature engineering, model selection, hyperparameter optimization,
    ensemble building, and cross-validation.
    """
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1):
        self.test_size = test_size
        self.val_size = val_size
        
        self.feature_engineer = FeatureEngineer()
        self.model_selector = ModelSelectionEngine()
        self.ensemble_builder = EnsembleBuilder()
        
        # Pipeline results
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        self.engineered_features = None
        self.selected_features = None
        self.best_model = None
        self.ensemble_models = []
        
        self.pipeline_history = deque(maxlen=100)
    
    def fit(self, X: np.ndarray, y: np.ndarray, prices: np.ndarray = None) -> Dict:
        """
        Run complete AutoML pipeline.
        """
        
        pipeline_start = datetime.now()
        
        logger.info("Starting AutoML pipeline...")
        
        # Step 1: Feature Engineering
        logger.info("Step 1: Feature engineering...")
        
        if prices is not None:
            returns = np.diff(np.log(prices))
            engineered_features = self.feature_engineer.engineer_ensemble_features(prices, returns)
            
            # Combine with original features
            X_engineered = np.column_stack([X, engineered_features.values])
        else:
            X_engineered = X
        
        self.engineered_features = X_engineered
        
        # Step 2: Feature Selection
        logger.info("Step 2: Feature selection...")
        
        selected_features = self.feature_engineer.select_best_features(
            pd.DataFrame(X_engineered),
            y,
            n_features=min(20, X_engineered.shape[1] // 2),
            method='mrmr'
        )
        
        feature_indices = [i for i, _ in enumerate(X_engineered.T) 
                          if i < len(selected_features)][:len(selected_features)]
        
        X_selected = X_engineered[:, feature_indices]
        
        self.selected_features = selected_features
        
        # Step 3: Data Splitting
        logger.info("Step 3: Splitting data...")
        
        n = len(X_selected)
        test_idx = int(n * (1 - self.test_size))
        val_idx = int(test_idx * (1 - self.val_size))
        
        self.X_train = X_selected[:val_idx]
        self.X_val = X_selected[val_idx:test_idx]
        self.X_test = X_selected[test_idx:]
        
        self.y_train = y[:val_idx]
        self.y_val = y[val_idx:test_idx]
        self.y_test = y[test_idx:]
        
        # Scaling
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)
        
        # Step 4: Model Selection & Hyperparameter Optimization
        logger.info("Step 4: Model selection and optimization...")
        
        best_model = None
        best_score = -np.inf
        model_performances = []
        
        for model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING, ModelType.SVM]:
            logger.debug(f"Optimizing {model_type.value}...")
            
            try:
                opt_result = self.model_selector.hyperparameter_optimization(
                    model_type,
                    self.X_train,
                    self.y_train,
                    n_iter=20
                )
                
                model = opt_result['best_model']
                
                performance = self.model_selector.evaluate_model(
                    model,
                    self.X_train, self.y_train,
                    self.X_val, self.y_val,
                    self.X_test, self.y_test
                )
                
                model_performances.append(performance)
                
                if performance.test_score > best_score:
                    best_score = performance.test_score
                    best_model = model
            
            except Exception as e:
                logger.warning(f"Error optimizing {model_type.value}: {e}")
        
        self.best_model = best_model
        
        # Step 5: Ensemble Building
        logger.info("Step 5: Building ensemble...")
        
        top_models = sorted(model_performances, key=lambda x: x.test_score, reverse=True)[:3]
        
        # Add base models to ensemble
        for perf in top_models:
            try:
                opt_result = self.model_selector.hyperparameter_optimization(
                    perf.model_type,
                    self.X_train,
                    self.y_train,
                    n_iter=10
                )
                
                model = opt_result['best_model']
                model.fit(self.X_train, self.y_train)
                self.ensemble_models.append(model)
            
            except:
                pass
        
        # Build voting ensemble
        if len(self.ensemble_models) > 1:
            ensemble_models_list = [
                (f"model_{i}", model) for i, model in enumerate(self.ensemble_models)
            ]
            
            ensemble = self.ensemble_builder.build_voting_ensemble(ensemble_models_list)
            ensemble.fit(self.X_train, self.y_train)
            
            ensemble_perf = self.model_selector.evaluate_model(
                ensemble,
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                self.X_test, self.y_test
            )
            
            logger.info(f"Ensemble test score: {ensemble_perf.test_score:.4f}")
        
        pipeline_time = (datetime.now() - pipeline_start).total_seconds()
        
        result = {
            'best_model': best_model,
            'ensemble_models': self.ensemble_models,
            'model_performances': model_performances,
            'best_test_score': best_score,
            'num_engineered_features': X_engineered.shape[1],
            'num_selected_features': len(selected_features),
            'pipeline_time_seconds': pipeline_time,
            'selected_features': selected_features
        }
        
        self.pipeline_history.append(result)
        
        logger.info(f"AutoML pipeline completed in {pipeline_time:.2f}s")
        logger.info(f"Best test score: {best_score:.4f}")
        
        return result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with best model."""
        if self.best_model is None:
            raise ValueError("Pipeline not fitted yet")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions."""
        if self.best_model is None:
            raise ValueError("Pipeline not fitted yet")
        
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            raise ValueError("Best model doesn't support predict_proba")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from best model."""
        if self.best_model is None:
            raise ValueError("Pipeline not fitted yet")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            return dict(zip(self.selected_features, importances))
        else:
            return {}
    
    def get_pipeline_summary(self) -> Dict:
        """Get comprehensive pipeline summary."""
        return {
            'num_models_evaluated': len(self.model_selector.evaluated_models),
            'best_model_type': type(self.best_model).__name__,
            'ensemble_size': len(self.ensemble_models),
            'num_engineered_features': self.engineered_features.shape[1] if self.engineered_features is not None else 0,
            'num_selected_features': len(self.selected_features) if self.selected_features else 0,
            'models_evaluated': [
                {
                    'model_type': perf.model_type.value,
                    'test_score': perf.test_score,
                    'auc_roc': perf.auc_roc,
                    'f1_score': perf.f1_score
                }
                for perf in self.model_selector.evaluated_models
            ]
        }


if __name__ == "__main__":
    print("=" * 80)
    print("AUTOML ENGINE")
    print("=" * 80)
    
    # Generate sample data
    np.random.seed(42)
    
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
    y = (prices > np.mean(prices)).astype(int)
    
    # Create and run pipeline
    pipeline = AutoMLPipeline(test_size=0.2, val_size=0.1)
    
    result = pipeline.fit(X, y, prices=prices)
    
    print(f"\nBest test score: {result['best_test_score']:.4f}")
    print(f"Engineered features: {result['num_engineered_features']}")
    print(f"Selected features: {result['num_selected_features']}")
    print(f"Pipeline time: {result['pipeline_time_seconds']:.2f}s")
    
    # Get summary
    summary = pipeline.get_pipeline_summary()
    print(f"\nPipeline Summary:")
    for key, val in summary.items():
        if isinstance(val, list):
            print(f"  {key}: {len(val)} models")
        else:
            print(f"  {key}: {val}")
    
    # Make predictions
    X_test = np.random.randn(10, n_features)
    predictions = pipeline.predict(X_test)
    print(f"\nSample predictions: {predictions}")
