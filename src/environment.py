import numpy as np 
from gym_anytrading.envs import StocksEnv
import yfinance as yf

from pandas.core.frame import DataFrame
from typing import Tuple, List, Dict

from .indicators import add_indicators


class MyStocksEnv(StocksEnv):
    """Custom stock trading environment with technical indicators."""
    def __init__(self, prices: np.ndarray, signal_features: np.ndarray, **kwargs):
        self._prices = prices
        self._signal_features = signal_features
        super().__init__(**kwargs)

    def _process_data(self):
        return self._prices, self._signal_features
    
def _preprocess_data(df: DataFrame, indicators: List, window_size: int, frame_bound: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Build signal features using selected technical indicators."""

    start = frame_bound[0] - window_size
    end = frame_bound[1]

    prices = df.Close.to_numpy()[start:end]
    signal_features = add_indicators(df, indicators)[indicators].to_numpy()[start:end]
    
    return prices, signal_features

def _get_trading_env_args(df: DataFrame, train_prop: float, window_size: int) -> Dict: 
    """Get train and test trading environment arguments."""
    n = len(df)
    train_size = int(n * train_prop)
    
    args = {
        "train": {
            "df": df, 
            "frame_bound": (window_size, train_size),
            "window_size": window_size
        }, 
        "test": {
            "df": df,
            "frame_bound": (train_size+window_size+1, n),
            "window_size": window_size
        }
    }    
    return args

def prepare_trading_env(cfg_env: Dict) -> Tuple[MyStocksEnv, MyStocksEnv]: 
    """Load data from Yahoo API and make train and test trading environment from config file."""    
    df_args = {
        "tickers": cfg_env["ticker"], 
        "interval": cfg_env["interval"], 
        "period": cfg_env["period"] 
    }

    df = yf.download(**df_args)

    train_prop = cfg_env["train_prop"]   
    window_size = cfg_env["window_size"]
    selected_indicators = cfg_env["indicators"]

    env_args = _get_trading_env_args(df, train_prop, window_size)

    prices_train, signal_features_train = _preprocess_data(indicators=selected_indicators, **env_args["train"])
    prices_test, signal_features_test = _preprocess_data(indicators=selected_indicators, **env_args["test"])

    env_train = MyStocksEnv(prices_train, signal_features_train, **env_args["train"])
    env_test = MyStocksEnv(prices_test, signal_features_test, **env_args["test"])

    return env_train, env_test