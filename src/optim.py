import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
import gym 

from typing import Optional, Dict

from .environment import MyStocksEnv

def _make_eval_callback(
    env: MyStocksEnv, 
    eval_freq: int, 
    max_no_improvement_evals: int, 
    min_evals: int) -> EvalCallback:
    """Create an evaluation callback for the given environment and configuration."""

    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=max_no_improvement_evals, 
        min_evals=min_evals, 
        verbose=1)
    
    eval_callback = EvalCallback(
        Monitor(env), 
        eval_freq=eval_freq, 
        callback_after_eval=stop_train_callback, 
        verbose=1)
    
    return eval_callback

def train(model: BaseAlgorithm, env: MyStocksEnv, cfg: Dict, n_steps_per_episode: Optional[int]=None): 

    eval_callback = _make_eval_callback(env, cfg["eval_freq"], cfg["max_no_improvement_evals"], cfg["min_evals"])

    if n_steps_per_episode is None:
        total_timesteps = cfg["n_episodes"] * env.n_steps_per_episode
    else:
        total_timesteps = cfg["n_episodes"] * n_steps_per_episode

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True) 
    model.save(cfg["save_path"])