from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from typing import Dict

def make_eval_callback(env: DummyVecEnv, config: Dict) -> EvalCallback:
    """Create an evaluation callback for the given environment and configuration."""

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=config["reward_threshold"], verbose=1)

    eval_callback = EvalCallback(
        eval_env=env,
        callback_on_new_best=stop_callback,
        eval_freq=config["eval_freq"],
        best_model_save_path=config["save_path"],
        verbose=1)
    
    return eval_callback