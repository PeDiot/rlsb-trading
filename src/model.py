from typing import Dict, Callable

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3 import PPO
from sb3_contrib.ppo_recurrent import RecurrentPPO

from torch import nn
import gymnasium as gym

from src.environment import vectorize_env

def _process_activation_fn(cfg: Dict): 

    if cfg["activation_fn"] == "ReLU": 
        cfg["activation_fn"] = nn.ReLU
    elif cfg["activation_fn"] == "LeakyReLU":
        cfg["activation_fn"] = nn.LeakyReLU
    elif cfg["activation_fn"] == "PreLU":
        cfg["activation_fn"] = nn.PReLU
    elif cfg["activation_fn"] == "Tanh":
        cfg["activation_fn"] = nn.Tanh
    else: 
        raise NotImplementedError
    
def _linear_schedule(initial_value: float) -> Callable[[float], float]:
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

def prepare_ppo(cfg: Dict, env: gym.Env, initial_lr: float=.01) -> BasePolicy: 
    """Prepare PPO model from config file.
    
    Args:
        cfg: config file
        env: vectorized environment
        initial_lr: initial learning rate
        
    Returns:
        PPO model (PPO or RecurrentPPO)."""

    cfg_policy = cfg["policy"].copy()
    n_lstm_layers = cfg_policy["n_lstm_layers"]

    _process_activation_fn(cfg_policy)

    scheduler = _linear_schedule(initial_value=initial_lr)

    env_vec = vectorize_env(env)

    if n_lstm_layers == 0: 
        cfg_policy.pop("lstm_hidden_size")
        cfg_policy.pop("n_lstm_layers")

        ppo = PPO(policy="MlpPolicy", env=env_vec, policy_kwargs=cfg_policy, learning_rate=scheduler, **cfg["ppo"])

    else: 
        ppo = RecurrentPPO(policy="MlpLstmPolicy", env=env_vec, policy_kwargs=cfg_policy, learning_rate=scheduler, **cfg["ppo"])

    return ppo