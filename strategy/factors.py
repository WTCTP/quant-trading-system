import numpy as np

from config import EMA_SHORT, EMA_LONG, ZSCORE_PERIOD, MOMENTUM_PERIOD


def calc_ema_trend(df):
    """EMA趋势因子：短期 vs 长期EMA差值，归一化到[-1, 1]"""
    ema_short = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
    ema_long = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
    raw = (ema_short - ema_long) / df['close']
    result = raw / (raw.std() + 1e-10)
    return np.tanh(result)


def calc_zscore(df):
    """均值回归因子：价格偏离均值的Z-score取反，归一化到[-1, 1]"""
    mean = df['close'].rolling(ZSCORE_PERIOD).mean()
    std = df['close'].rolling(ZSCORE_PERIOD).std()
    zscore = (df['close'] - mean) / (std + 1e-10)
    mean_reversion = -np.tanh(zscore / 2)
    return mean_reversion


def calc_momentum(df):
    """动量因子：周期收益率，归一化到[-1, 1]"""
    momentum = df['close'].pct_change(MOMENTUM_PERIOD)
    return np.tanh(momentum / (momentum.std() + 1e-10))
