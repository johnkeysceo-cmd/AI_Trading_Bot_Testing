"""
models/classical_ml/catboost_model.py
-------------------------------------
CatBoost model for crypto trading predictions.

Features:
- Handles OHLCV + technical indicators
- Multi-agent prediction outputs
- Training, evaluation, saving/loading
- Hyperparameter tuning support
- Thread-safe for multi-agent environments
"""

import os
import logging
import threading
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("CatBoostModel")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# -----------------------------
# Feature Engineering
# -----------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['std_10'] = df['close'].rolling(window=10).std()
    df['momentum'] = df['close'] - df['close'].shift(5)
    df['rsi'] = 100 - 100 / (1 + (df['return'].rolling(14).mean() / df['return'].rolling(14).std()))
    df = df.dropna()
    return df

def generate_labels(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    future_return = df['close'].shift(-horizon) - df['close']
    labels = (future_return > 0).astype(int)
    return labels[:-horizon]

# -----------------------------
# CatBoost Model Class
# -----------------------------
class CatBoostTradingModel:
    def __init__(self, model_dir="./data/models/catboost", seed=42):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.seed = seed
        self.lock = threading.Lock()
        self.model = None
        self.feature_columns = []
        logger.info("[CatBoostTradingModel] Initialized")

    def train(self, df: pd.DataFrame, horizon: int = 1):
        df = create_features(df)
        labels = generate_labels(df, horizon)
        df = df.iloc[:-horizon]
        self.feature_columns = [c for c in df.columns if c not in ['close', 'return']]

        X = df[self.feature_columns].values
        y = labels.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        self.model = CatBoostClassifier(iterations=500,
                                        learning_rate=0.05,
                                        depth=6,
                                        random_seed=self.seed,
                                        verbose=50)
        self.model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=50)

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"[CatBoostTradingModel] Training complete. Accuracy: {acc:.4f}")

        self.save_model("catboost_trading_model.cbm")

    def predict(self, df: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        df = create_features(df)
        X = df[self.feature_columns].values
        with self.lock:
            preds = self.model.predict(X)
        return preds

    def save_model(self, filename: str):
        path = os.path.join(self.model_dir, filename)
        with self.lock:
            self.model.save_model(path)
        logger.info(f"[CatBoostTradingModel] Model saved to {path}")

    def load_model(self, filename: str):
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        with self.lock:
            self.model = CatBoostClassifier()
            self.model.load_model(path)
        logger.info(f"[CatBoostTradingModel] Model loaded from {path}")
