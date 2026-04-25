import numpy as np

from config import WEIGHT_TREND, WEIGHT_MEAN, WEIGHT_MOMENTUM, SIGNAL_THRESHOLD


def calc_signal(df):
    """加权合成信号值"""
    trend = df['factor_trend'].fillna(0).values
    mean = df['factor_mean'].fillna(0).values
    momentum = df['factor_momentum'].fillna(0).values

    signal = WEIGHT_TREND * trend + WEIGHT_MEAN * mean + WEIGHT_MOMENTUM * momentum
    return signal


def signal_to_action(signal_val):
    """信号值 → 交易动作"""
    if signal_val > SIGNAL_THRESHOLD:
        return 'long'
    elif signal_val < -SIGNAL_THRESHOLD:
        return 'short'
    return 'flat'
