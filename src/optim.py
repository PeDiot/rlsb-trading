from typing import Optional, Dict

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym


def _make_eval_callback(
    env: gym.Env, 
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

def train(model: BaseAlgorithm, env_eval: gym.Env, cfg: Dict, n_steps_per_episode: int=-1): 
    """Train policy with the given configuration and evaluate on env.
    
    Args:
        model: Model to train.
        env_eval: Environment to evaluate policy on.
        cfg: Configuration for training.
        n_steps_per_episode: Number of steps per episode. Defaults to the number of steps per episode of the environment."""

    max_n_steps_per_episode = model.env.envs[0].n_steps_per_episode

    if n_steps_per_episode == -1:
        n_steps_per_episode = max_n_steps_per_episode

    elif n_steps_per_episode > max_n_steps_per_episode:
        raise ValueError(f"n_steps_per_episode must be less than or equal to {max_n_steps_per_episode}.")

    cfg_optim = cfg["optim"]
    total_timesteps = cfg_optim["n_episodes"] * n_steps_per_episode
    eval_freq = n_steps_per_episode

    eval_callback = _make_eval_callback(env_eval, eval_freq, cfg_optim["max_no_improvement_evals"], cfg_optim["min_evals"])

    model.learn(
        total_timesteps=total_timesteps, 
        callback=eval_callback, 
        reset_num_timesteps=False,
        progress_bar=True)