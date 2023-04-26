from typing import Dict, Tuple, List, Optional

import yaml
from yaml import Loader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

from stable_baselines3.common.base_class import BaseAlgorithm

from src.environment import MyStocksEnv


def load_config(cfg_path: str) -> Dict:
    """Load YAML config file."""

    with open(cfg_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader)

    return cfg

def get_model_path(cfg: Dict) -> str: 
    """Get model path."""

    cfg_env = cfg["env"]
    n_indicators = len(cfg_env["indicators"])

    file_name = f"{cfg_env['ticker']}_{cfg_env['period']}_{cfg_env['interval']}_{n_indicators}_indicators/" 
    file_path = "models/" + file_name

    return file_path

def save_model(model: BaseAlgorithm, cfg: Dict): 
    save_path = get_model_path(cfg)
    model.save(save_path)
    print(f"Model saved to {save_path}")

def display_env(env: MyStocksEnv, fig_dims: Tuple=(8, 4)): 
    """Display environment (prices and positions)."""

    plt.figure(figsize=fig_dims, facecolor="w")
    plt.cla()
    env.render_all()
    plt.show()

def plot_results(rewards: List, profits: List, fig: Optional[plt.Figure]=None):
    """Plot cumulative rewards and profits."""

    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor="w") 

    n_actions = [t+1 for t in range(len(rewards))]
    cum_rewards = np.cumsum(rewards)

    sns.lineplot(x=n_actions, y=cum_rewards, ax=axes[0])
    sns.lineplot(x=n_actions, y=profits, ax=axes[1], color="orange")

    axes[0].set_ylabel("Cumulative Rewards")
    axes[1].set_ylabel("Cumulative Profits")  

    axes[0].set_title(f"Total Rewards: {cum_rewards[-1]:.2f}")
    axes[1].set_title(f"Total Profits: {profits[-1]:.2f}")