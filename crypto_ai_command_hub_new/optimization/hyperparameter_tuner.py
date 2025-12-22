# hyperparameter_tuner.py
# -----------------------
# Unified hyperparameter tuning interface using Optuna, Ray Tune, and NNI
# Optimizes confidence thresholds, risk fractions, and model hyperparameters
# for all agents.

import optuna
from ray import tune
from nni.experiment import Experiment
from typing import Dict, Any
from agents.meta_agent.agent_selector import agents, dispatch_signals, CAPITAL_POOL, ALLOCATION_TIERS
from agents.external._common import TradeSignal

# -------------------------------
# Objective for Optuna
# -------------------------------
def optuna_objective(trial: optuna.Trial) -> float:
    """
    Optimize confidence threshold and risk fraction per agent
    """
    # Example: confidence threshold per tier
    thresholds = {
        tier: trial.suggest_float(f"{tier}_conf_threshold", 0.2, 0.9)
        for tier in ALLOCATION_TIERS
    }
    risk_fractions = {
        tier: trial.suggest_float(f"{tier}_risk_fraction", 0.01, 0.2)
        for tier in ALLOCATION_TIERS
    }

    # Simulate signals
    total_reward = 0
    for tier, adapter in agents.items():
        signals: list[TradeSignal] = adapter.generate_signals()
        for s in signals:
            if s.confidence >= thresholds[tier]:
                # Simplified reward = confidence * allocation
                capital = CAPITAL_POOL * risk_fractions[tier]
                total_reward += s.confidence * capital
    # Negative because Optuna minimizes by default
    return -total_reward

def run_optuna_study(n_trials: int = 50):
    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=n_trials)
    print("Best hyperparameters:", study.best_params)
    return study.best_params

# -------------------------------
# Objective for Ray Tune
# -------------------------------
def ray_tune_objective(config: Dict[str, float]):
    reward = 0
    for tier, adapter in agents.items():
        threshold = config[f"{tier}_conf_threshold"]
        risk_fraction = config[f"{tier}_risk_fraction"]
        signals = adapter.generate_signals()
        for s in signals:
            if s.confidence >= threshold:
                capital = CAPITAL_POOL * risk_fraction
                reward += s.confidence * capital
    return {"score": reward}

def run_ray_tune(n_samples: int = 20):
    config = {
        f"{tier}_conf_threshold": tune.uniform(0.2, 0.9)
        for tier in ALLOCATION_TIERS
    }
    config.update({
        f"{tier}_risk_fraction": tune.uniform(0.01, 0.2)
        for tier in ALLOCATION_TIERS
    })
    analysis = tune.run(ray_tune_objective, config=config, num_samples=n_samples)
    print("Best config:", analysis.get_best_config(metric="score"))
    return analysis.get_best_config(metric="score")

# -------------------------------
# NNI Integration
# -------------------------------
def nni_trial(params: Dict[str, Any]):
    reward = 0
    for tier, adapter in agents.items():
        threshold = params[f"{tier}_conf_threshold"]
        risk_fraction = params[f"{tier}_risk_fraction"]
        signals = adapter.generate_signals()
        for s in signals:
            if s.confidence >= threshold:
                capital = CAPITAL_POOL * risk_fraction
                reward += s.confidence * capital
    # Report result to NNI
    import nni
    nni.report_final_result(reward)
    return reward
