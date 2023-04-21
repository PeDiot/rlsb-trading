from ta.momentum import rsi
from ta.trend import macd
from ta.volume import (
    on_balance_volume, 
    money_flow_index, 
    chaikin_money_flow, 
)
from ta.volatility import average_true_range

from pandas.core.series import Series
from pandas.core.frame import DataFrame
from typing import List


INDICATORS = {
    "MOM": lambda df: _momentum_indicator(df.Close), 
    "MACD": lambda df: macd(df.Close), 
    "MFI": lambda df: money_flow_index(df.High, df.Low, df.Close, df.Volume), 
    "RSI": lambda df: rsi(df.Close), 
    "ATR": lambda df: average_true_range(df.High, df.Low, df.Close), 
    "CO": lambda df: chaikin_money_flow(df.High, df.Low, df.Close, df.Volume), 
    "OBV": lambda df: on_balance_volume(close=df.Close, volume=df.Volume), 
}

def _momentum_indicator(close: Series, window: int=14) -> Series:
    """Description. Calculate momentum indicator."""
    close_shifted = close.shift(periods=window)

    return 100 * close / close_shifted

def add_indicators(df: DataFrame, indicators: List) -> DataFrame: 
    """Description. Add technical indicators to financial dataset."""

    for indicator in indicators: 
        if indicator not in list(INDICATORS.keys()): 
            raise ValueError(f"Indicator {indicator} not supported.")
        else:
            fun = INDICATORS[indicator]
            df.loc[:, indicator] = fun(df)

    df.fillna(0, inplace=True)
    return df