from typing import Optional, Dict, Callable


from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    Args:
        initial_value: Initial learning rate.
    
    Returns: schedule that compute current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0."""
        return progress_remaining * initial_value

    return func

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

def train(model: BaseAlgorithm, env: gym.Env, cfg: Dict, n_steps_per_episode: Optional[int]=None): 
    """Train policy with the given configuration and evaluate on env.
    
    Args:
        model: Model to train.
        env: Environment to evaluate policy on.
        cfg: Configuration for training.
        n_steps_per_episode: Number of steps per episode. Defaults to the number of steps per episode of the environment."""

    if n_steps_per_episode is None:
        n_steps_per_episode = env.n_steps_per_episode

    total_timesteps = cfg["n_episodes"] * n_steps_per_episode

    eval_callback = _make_eval_callback(env, n_steps_per_episode, cfg["max_no_improvement_evals"], cfg["min_evals"])

    model.learn(
        total_timesteps=total_timesteps, 
        callback=eval_callback, 
        progress_bar=True) 
    
    model.save(cfg["save_path"])