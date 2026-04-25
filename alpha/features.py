import numpy as np
import pandas as pd

from config import ZSCORE_PERIOD, VOL_PERIOD


def build_features(df):
    """构建行为特征（市场微结构 + 资金行为），返回 (X, feature_names)"""
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']
    ret = c.pct_change()

    features = pd.DataFrame(index=df.index)

    # 1. 成交量异常：当前成交量偏离历史均值的程度
    vol_mean = v.rolling(50).mean()
    vol_std = v.rolling(50).std()
    features['vol_zscore'] = (v - vol_mean) / (vol_std + 1e-10)

    # 2. 价格冲击：单位成交量的价格变动
    amihud = abs(ret) / (v + 1e-10)
    features['impact'] = amihud.rolling(VOL_PERIOD).mean()

    # 3. 波动率状态：当前波动率 vs 长期波动率
    vol_short = ret.rolling(VOL_PERIOD).std()
    vol_long = ret.rolling(100).std()
    features['vol_regime'] = vol_short / (vol_long + 1e-10)

    # 4. 突破强度：标准化后的价格位置
    high_20 = h.rolling(20).max()
    low_20 = l.rolling(20).min()
    atr = (h - l).rolling(20).mean()
    features['breakout_strength'] = (c - low_20) / (atr + 1e-10)

    # 5. 趋势连续性（惯性）
    up = (ret > 0).astype(int)
    down = (ret < 0).astype(int)
    features['up_streak'] = up.rolling(5).sum()
    features['down_streak'] = down.rolling(5).sum()

    # 6. 流动性压缩：近期振幅 vs 历史振幅
    daily_range = h - l
    range_mean = daily_range.rolling(50).mean()
    features['compression'] = daily_range / (range_mean + 1e-10)

    return features


def build_label(df, forward_bars=5):
    """分类标签：未来N根K线是否上涨"""
    future_close = df['close'].shift(-forward_bars)
    y = (future_close > df['close']).astype(int)
    return y
