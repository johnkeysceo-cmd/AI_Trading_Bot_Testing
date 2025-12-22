"""
models/classical_ml/lightgbm_model.py
-------------------------------------
LightGBM model for crypto trading predictions.

Features:
- Handles OHLCV + technical indicators
- Multi-agent prediction outputs
- Training, validation, and model evaluation
- Model persistence and checkpointing
- Hyperparameter tuning support via Optuna
- Thread-safe for multi-agent use
"""

import os
import json
import logging
import threading
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import optuna

# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger("LightGBMModel")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# -----------------------------
# Feature Engineering Utilities
# -----------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features for LightGBM model.
    Adds technical indicators and rolling stats.
    """
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log1p(df['return'])
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['std_10'] = df['close'].rolling(window=10).std()
    df['momentum'] = df['close'] - df['close'].shift(5)
    df['rsi'] = 100 - 100 / (1 + (df['return'].rolling(14).mean() / df['return'].rolling(14).std()))
    df = df.dropna()
    return df

def generate_labels(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Generate classification labels: 1 if price goes up, 0 otherwise
    """
    df = df.copy()
    future_return = df['close'].shift(-horizon) - df['close']
    labels = (future_return > 0).astype(int)
    return labels[:-horizon]

# -----------------------------
# LightGBM Model Class
# -----------------------------
class LightGBMTradingModel:
    """
    Wrapper for LightGBM model in crypto trading.
    Supports classification and regression.
    """
    def __init__(self,
                 model_type: str = "classification",
                 model_dir: str = "./data/models/lightgbm",
                 seed: int = 42):
        self.model_type = model_type
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.seed = seed
        self.lock = threading.Lock()
        self.model = None
        self.feature_columns = []
        logger.info(f"[LightGBMTradingModel] Initialized ({model_type})")

    # -----------------------------
    # Training
    # -----------------------------
    def train(self, df: pd.DataFrame, horizon: int = 1):
        """
        Train the LightGBM model
        """
        df = create_features(df)
        labels = generate_labels(df, horizon)
        df = df.iloc[:-horizon]
        self.feature_columns = [c for c in df.columns if c not in ['close', 'return', 'log_return']]

        X = df[self.feature_columns].values
        y = labels.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed, shuffle=False)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            'objective': 'binary' if self.model_type == "classification" else 'regression',
            'metric': 'binary_logloss' if self.model_type == "classification" else 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'seed': self.seed,
            'verbose': -1
        }

        logger.info("[LightGBMTradingModel] Starting model training...")
        self.model = lgb.train(params,
                               train_data,
                               valid_sets=[train_data, valid_data],
                               num_boost_round=500,
                               early_stopping_rounds=50,
                               verbose_eval=50)

        y_pred = np.round(self.model.predict(X_test))

        if self.model_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            logger.info(f"[LightGBMTradingModel] Training complete. Accuracy: {acc:.4f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            logger.info(f"[LightGBMTradingModel] Training complete. MSE: {mse:.6f}")

        self.save_model("lightgbm_trading_model.pkl")

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict using trained LightGBM model
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")
        df = create_features(df)
        X = df[self.feature_columns].values
        with self.lock:
            preds = np.round(self.model.predict(X))
        return preds

    # -----------------------------
    # Save / Load
    # -----------------------------
    def save_model(self, filename: str):
        path = os.path.join(self.model_dir, filename)
        with self.lock:
            joblib.dump(self.model, path)
        logger.info(f"[LightGBMTradingModel] Model saved to {path}")

    def load_model(self, filename: str):
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        with self.lock:
            self.model = joblib.load(path)
        logger.info(f"[LightGBMTradingModel] Model loaded from {path}")

# -----------------------------
# Hyperparameter Optimization
# -----------------------------
class LightGBMHPO:
    """
    Optuna-based hyperparameter tuning
    """
    def __init__(self, df: pd.DataFrame, horizon: int = 1, n_trials: int = 20):
        self.df = create_features(df)
        self.labels = generate_labels(self.df, horizon)
        self.df = self.df.iloc[:-horizon]
        self.X = self.df[[c for c in self.df.columns if c not in ['close', 'return', 'log_return']]].values
        self.y = self.labels.values
        self.n_trials = n_trials

    def objective(self, trial):
        num_leaves = trial.suggest_int("num_leaves", 20, 100)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        n_estimators = trial.suggest_int("n_estimators", 50, 500)

        model = lgb.LGBMClassifier(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, shuffle=False)
        model.fit(X_train, y_train)
        preds = np.round(model.predict(X_test))
        acc = accuracy_score(y_test, preds)
        return 1 - acc

    def run(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        logger.info(f"[LightGBMHPO] Best parameters: {study.best_params}")
        return study.best_params

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    df = pd.read_csv("./data/raw/sample_candles.csv")
    model = LightGBMTradingModel()
    model.train(df)
    preds = model.predict(df)
    print(f"Predictions: {preds[-10:]}")
