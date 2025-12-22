"""
models/classical_ml/xgboost_model.py
------------------------------------
XGBoost model for feature-based crypto trading predictions.

Features:
- Handles OHLCV + technical indicator features
- Multi-agent prediction outputs
- Training, validation, and model evaluation
- Model persistence and checkpointing
- Hyperparameter tuning support via Optuna (optional)
- Thread-safe for multi-agent use
"""

import os
import json
import logging
import threading
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import optuna

# Logging setup
logger = logging.getLogger("XGBoostModel")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# -----------------------------
# Utility functions
# -----------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features from OHLCV and technical indicators.
    This can be expanded to include RSI, MACD, Bollinger Bands, etc.
    """
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log1p(df['return'])
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['volatility'] = df['log_return'].rolling(window=10).std()
    df['momentum'] = df['close'] - df['close'].shift(5)
    df = df.dropna()
    return df

def generate_labels(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Generates target labels: 1 for price up, 0 for price down
    """
    df = df.copy()
    future_return = df['close'].shift(-horizon) - df['close']
    labels = (future_return > 0).astype(int)
    return labels[:-horizon]  # Drop last horizon rows

# -----------------------------
# XGBoost Trading Model
# -----------------------------
class XGBoostTradingModel:
    """
    XGBoost model wrapper for crypto trading.
    Supports classification (up/down) and regression (price prediction)
    """

    def __init__(self,
                 model_type: str = "classification",
                 model_dir: str = "./data/models/xgboost",
                 seed: int = 42):
        self.model_type = model_type
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.seed = seed
        self.lock = threading.Lock()
        self.model = None
        self.feature_columns = []
        logger.info(f"[XGBoostTradingModel] Initialized ({model_type})")

    # -----------------------------
    # Model training
    # -----------------------------
    def train(self, df: pd.DataFrame, horizon: int = 1):
        """
        Train the XGBoost model
        """
        df = create_features(df)
        labels = generate_labels(df, horizon)
        df = df.iloc[:-horizon]
        self.feature_columns = [c for c in df.columns if c not in ['close', 'return', 'log_return']]

        X = df[self.feature_columns].values
        y = labels.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed, shuffle=False)

        if self.model_type == "classification":
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=self.seed,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=self.seed
            )

        logger.info("[XGBoostTradingModel] Starting model training...")
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        if self.model_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            logger.info(f"[XGBoostTradingModel] Training complete. Accuracy: {acc:.4f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            logger.info(f"[XGBoostTradingModel] Training complete. MSE: {mse:.6f}")

        # Save model
        self.save_model("xgboost_trading_model.pkl")

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts the next move using the trained model
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded yet")

        df = create_features(df)
        X = df[self.feature_columns].values
        with self.lock:
            preds = self.model.predict(X)
        return preds

    # -----------------------------
    # Save / Load
    # -----------------------------
    def save_model(self, filename: str):
        path = os.path.join(self.model_dir, filename)
        with self.lock:
            joblib.dump(self.model, path)
        logger.info(f"[XGBoostTradingModel] Model saved to {path}")

    def load_model(self, filename: str):
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        with self.lock:
            self.model = joblib.load(path)
        logger.info(f"[XGBoostTradingModel] Model loaded from {path}")

# -----------------------------
# Hyperparameter Optimization (Optional)
# -----------------------------
class XGBoostHPO:
    """
    Optuna-based hyperparameter optimization for XGBoost
    """

    def __init__(self, df: pd.DataFrame, horizon: int = 1, n_trials: int = 20):
        self.df = create_features(df)
        self.labels = generate_labels(self.df, horizon)
        self.df = self.df.iloc[:-horizon]
        self.X = self.df[[c for c in self.df.columns if c not in ['close', 'return', 'log_return']]].values
        self.y = self.labels.values
        self.n_trials = n_trials

    def objective(self, trial):
        max_depth = trial.suggest_int("max_depth", 3, 10)
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        model = XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, shuffle=False)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return 1 - acc  # minimize error

    def run(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        logger.info(f"[XGBoostHPO] Best parameters: {study.best_params}")
        return study.best_params

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    df = pd.read_csv("./data/raw/sample_candles.csv")
    model = XGBoostTradingModel()
    model.train(df)
    preds = model.predict(df)
    print(f"Predictions: {preds[-10:]}")
