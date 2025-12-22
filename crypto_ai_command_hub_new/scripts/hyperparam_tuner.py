#!/usr/bin/env python3
"""Hyperparameter tuning scaffold using Optuna.

This file provides a minimal Optuna objective that runs short paper
simulations with the `ULTIMATE_10_10_TradingBot` to evaluate candidate
hyperparameters. It includes a guardrail to prune trials whose simulated
maximum drawdown exceeds a threshold.

Run locally after installing `optuna`.
"""

import optuna
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger('hyperparam_tuner')

from models.ULTIMATE_10_10_TRADING_BOT import ULTIMATE_10_10_TradingBot


def simulate_with_params(params, days=7):
    """Run a short simulation with given params and return performance metrics."""
    bot = ULTIMATE_10_10_TradingBot(
        initial_capital=200.0,
        use_real_data=False,
        lookback_days=days,
        entry_threshold=params.get('entry_threshold', 0.1),
        min_confidence=params.get('min_confidence', 0.0),
        max_positions=int(params.get('max_positions', 3))
    )

    bot.run_simulation()

    # compute simple metrics
    final = bot.capital
    pnl = final - bot.initial_capital
    # approximate drawdown: max peak-to-trough using closed trades sequence
    peak = bot.initial_capital
    max_dd = 0.0
    equity = bot.initial_capital
    for t in bot.closed_trades:
        equity += t.get('pnl', 0)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return {'pnl': pnl, 'max_drawdown': max_dd, 'trades': len(bot.closed_trades)}


def objective(trial: optuna.Trial):
    # Suggest hyperparameters
    entry_threshold = trial.suggest_float('entry_threshold', 0.05, 0.5)
    min_confidence = trial.suggest_float('min_confidence', 0.0, 0.6)
    max_positions = trial.suggest_int('max_positions', 1, 5)

    params = {
        'entry_threshold': entry_threshold,
        'min_confidence': min_confidence,
        'max_positions': max_positions
    }

    metrics = simulate_with_params(params, days=7)

    # Guardrail: prune if drawdown exceeds 30%
    if metrics['max_drawdown'] > 0.30:
        raise optuna.TrialPruned()

    # Objective: maximize pnl (Optuna minimizes by default, so return negative)
    return -metrics['pnl']


def run_study(n_trials=20, storage: str = None, study_name: str = 'ultimate_tuning'):
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler, study_name=study_name)
    study.optimize(objective, n_trials=n_trials)
    logger.info(f'Best trial: {study.best_trial.params} -> value={study.best_value}')
    return study


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--trials', '-n', type=int, default=20)
    args = p.parse_args()

    run_study(n_trials=args.trials)
